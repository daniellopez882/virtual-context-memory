from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from vcmem.bench import RunConfig, run_benchmark, write_results
from vcmem.bench.runner import replay_agent
from vcmem.build import build_manager
from vcmem.cli import app
from vcmem.demo import QUESTIONS, demo_messages
from vcmem.embed import HashingEmbedder
from vcmem.scoring import HeuristicImportance
from vcmem.scripted import ScriptedMemoryAgent
from vcmem.tokens import RegexTokenCounter

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()


def test_demo_runs_every_policy() -> None:
    result = runner.invoke(app, ["demo", "--budget", "320"])
    assert result.exit_code == 0, result.output
    for policy in ("fifo", "lru", "arc", "importance"):
        assert policy in result.output


def test_demo_questions_point_at_real_turns() -> None:
    ids = {m.id for m in demo_messages()}
    assert all(set(evidence) <= ids for _, evidence in QUESTIONS)


def test_report_command_prints_tables(mini_locomo: Path, tmp_path: Path) -> None:
    results = run_benchmark(
        RunConfig(budgets=(400,), methods=("window", "memgpt-lru")),
        mini_locomo,
        HashingEmbedder(dim=32),
        RegexTokenCounter(),
        command="test",
        repo=tmp_path,
    )
    path = tmp_path / "r.json"
    write_results(results, path)
    out = runner.invoke(app, ["report", str(path)])
    assert out.exit_code == 0, out.output
    assert "| window |" in out.output
    assert json.loads(path.read_text(encoding="utf-8"))["config"]["budgets"] == [400]


def test_inspect_replays_with_the_benchmark_settings(
    mini_locomo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "locomo"
    data.mkdir()
    shutil.copy(mini_locomo, data / "locomo10.json")
    monkeypatch.setenv("VCMEM_DATA_DIR", str(tmp_path))
    args = ["inspect", "--sample", "fixture-1", "--question", "0", "--budget", "330"]
    out = runner.invoke(app, [*args, "--embedder", "hashing", "--counter", "regex"])
    assert out.exit_code == 0, out.output
    assert "evidence  :" in out.output and "tokens    :" in out.output


def test_replay_agent_uses_the_run_config() -> None:
    manager = build_manager(budget=400)
    agent = replay_agent(manager, RunConfig(archive_threshold=0.3, page_size=2))
    assert isinstance(agent.provider, ScriptedMemoryAgent)
    assert agent.provider.archive_threshold == 0.3
    # the library default and the benchmark default are one setting
    standalone = ScriptedMemoryAgent(manager, HeuristicImportance())
    assert standalone.archive_threshold == RunConfig().archive_threshold
    assert standalone.question_fill == RunConfig().question_fill


def test_bench_requires_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VCMEM_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["bench", "--conversations", "1"])
    assert result.exit_code == 2
    assert "vcmem fetch" in result.output
