"""Command line: fetch data, run the replay benchmark, inspect a replay, serve MCP."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer

from vcmem.config import data_dir, locomo_path

app = typer.Typer(
    help="Virtual context memory: paged agent memory and a LoCoMo replay benchmark.",
    no_args_is_help=True,
    add_completion=False,
)

EmbedderOpt = Annotated[
    Literal["auto", "minilm", "hashing"],
    typer.Option(help="auto = MiniLM if fetched, else hashing."),
]
CounterOpt = Annotated[
    Literal["auto", "wordpiece", "regex"],
    typer.Option(help="Token counter for budgets; auto = WordPiece if fetched."),
]


def _command() -> str:
    return " ".join(["vcmem", *sys.argv[1:]])


@app.command()
def fetch() -> None:
    """Download LoCoMo and the MiniLM ONNX model into $VCMEM_DATA_DIR and verify sha256."""
    from vcmem.assets import fetch as do_fetch

    do_fetch(data_dir())


@app.command()
def bench(
    budget: Annotated[
        list[int] | None, typer.Option(help="Token budget; repeat for several.")
    ] = None,
    method: Annotated[list[str] | None, typer.Option(help="Method; repeat. Default: all.")] = None,
    conversations: Annotated[int | None, typer.Option(help="Use the first N.")] = None,
    questions: Annotated[int | None, typer.Option(help="First N questions each.")] = None,
    embedder: EmbedderOpt = "minilm",
    counter: CounterOpt = "wordpiece",
    flush_to: float = 0.75,
    page_size: int = 5,
    out: Annotated[Path, typer.Option(help="Results JSON path.")] = Path(
        "bench/results/locomo-replay.json"
    ),
    qa_base_url: Annotated[
        str | None, typer.Option(help="OpenAI-compatible URL to also score answers.")
    ] = None,
    qa_model: str | None = None,
    agent_base_url: Annotated[
        str | None, typer.Option(help="OpenAI-compatible URL to drive paging with an LLM.")
    ] = None,
    agent_model: str | None = None,
) -> None:
    """Replay LoCoMo turn by turn and measure evidence recall per method and budget."""
    from vcmem.bench import METHODS, RunConfig, run_benchmark, write_results
    from vcmem.bench.report import render_report
    from vcmem.build import make_counter, make_embedder
    from vcmem.providers import OpenAICompatibleProvider

    path = locomo_path()
    if not path.exists():
        typer.echo(f"{path} not found; run `vcmem fetch` first", err=True)
        raise typer.Exit(2)
    qa_provider = OpenAICompatibleProvider(qa_base_url, qa_model or "") if qa_base_url else None
    agent_provider = (
        OpenAICompatibleProvider(agent_base_url, agent_model or "") if agent_base_url else None
    )
    config = RunConfig(
        budgets=tuple(budget or (1024, 2048, 4096)),
        methods=tuple(method or METHODS),
        max_conversations=conversations,
        max_questions=questions,
        flush_to=flush_to,
        page_size=page_size,
        qa_provider=qa_provider,
        agent_provider=agent_provider,
    )
    results = run_benchmark(
        config,
        path,
        make_embedder(embedder),
        make_counter(counter),
        command=_command(),
        repo=Path.cwd(),
        log=lambda line: typer.echo(line, err=True),
    )
    write_results(results, out)
    typer.echo(render_report(results))
    typer.echo(f"\nwrote {out} in {results['runtime_seconds']} s")


@app.command()
def report(path: Path) -> None:
    """Print the markdown tables for a results JSON."""
    from vcmem.bench.report import render_report

    typer.echo(render_report(json.loads(path.read_text(encoding="utf-8"))))


@app.command()
def inspect(
    sample: Annotated[str, typer.Option(help="LoCoMo sample_id, e.g. conv-26.")] = "conv-26",
    question: Annotated[int, typer.Option(help="Index into the sample's QA list.")] = 0,
    policy: str = "arc",
    budget: int = 2048,
    embedder: EmbedderOpt = "auto",
    counter: CounterOpt = "auto",
) -> None:
    """Replay one LoCoMo conversation and print the context assembled for one question."""
    from vcmem.agent import HeartbeatAgent
    from vcmem.build import make_counter, make_embedder
    from vcmem.locomo import CATEGORY_NAMES, load_locomo
    from vcmem.manager import MemoryManager
    from vcmem.scoring import HeuristicImportance
    from vcmem.scripted import ScriptedMemoryAgent
    from vcmem.stores.sqlite import Database

    convs = {c.sample_id: c for c in load_locomo(locomo_path())}
    if sample not in convs:
        typer.echo(f"unknown sample {sample!r}; have {sorted(convs)}", err=True)
        raise typer.Exit(2)
    conv = convs[sample]
    qa = conv.qa[question]
    manager = MemoryManager(
        Database(),
        budget=budget,
        policy=policy,
        embedder=make_embedder(embedder),
        counter=make_counter(counter),
        importance=HeuristicImportance(),
        core_blocks=conv.speakers,
    )
    agent = HeartbeatAgent(manager, ScriptedMemoryAgent(manager, HeuristicImportance()))
    for message in conv.messages():
        agent.observe(message)
    trace = agent.ask(qa.question)
    context = manager.assemble()
    present = context.full_sources()
    typer.echo(context.render())
    typer.echo("")
    typer.echo(f"question  : {qa.question} [{CATEGORY_NAMES.get(qa.category, qa.category)}]")
    typer.echo(f"gold      : {qa.answer}")
    typer.echo(f"tool calls: {[c.name for c in trace.calls]}")
    typer.echo(f"tokens    : {context.tokens}/{budget}")
    for ev in qa.evidence:
        typer.echo(f"evidence  : {ev} {'IN CONTEXT' if ev in present else 'missing'}")
    typer.echo(f"manager   : {manager.stats}")


@app.command()
def demo(budget: int = 320) -> None:
    """Run every policy on a short synthetic chat (no downloads needed)."""
    from vcmem.demo import run_demo

    rows = run_demo(budget)
    typer.echo(f"{'policy':<12}{'evidence in context':>22}{'evictions':>11}{'page-ins':>10}")
    for r in rows:
        typer.echo(f"{r.policy:<12}{f'{r.found}/{r.total}':>22}{r.evictions:>11}{r.page_ins:>10}")


@app.command()
def serve(
    transport: Literal["stdio", "http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
    db: Annotated[Path | None, typer.Option(help="SQLite file (default in data dir).")] = None,
    budget: int = 4096,
    policy: str = "arc",
    embedder: EmbedderOpt = "auto",
    counter: CounterOpt = "auto",
) -> None:
    """Serve the memory tools over MCP (stdio, or streamable HTTP at /mcp with /healthz)."""
    from vcmem.build import make_counter, make_embedder
    from vcmem.manager import MemoryManager
    from vcmem.mcp_server import create_server
    from vcmem.scoring import HeuristicImportance
    from vcmem.stores.sqlite import Database

    db_path = db or data_dir() / "memory.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    manager = MemoryManager(
        Database(db_path),
        budget=budget,
        policy=policy,
        embedder=make_embedder(embedder),
        counter=make_counter(counter),
        importance=HeuristicImportance(),
    )
    server = create_server(manager)
    if transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=host, port=port)
