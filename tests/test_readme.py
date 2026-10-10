"""The README's Python examples must run as written (the LLM example needs a server),
and its result tables must be what `vcmem report` prints from the committed results."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from vcmem.bench.report import (
    bootstrap_table,
    category_table,
    latency_table,
    recall_table,
    work_table,
)

README = Path(__file__).parent.parent / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.S)
OFFLINE = [b for b in BLOCKS if "OpenAICompatibleProvider(" not in b]


def test_readme_has_examples() -> None:
    assert len(OFFLINE) >= 2


@pytest.mark.parametrize("code", OFFLINE, ids=[f"block{i}" for i in range(len(OFFLINE))])
def test_readme_example_runs(code: str, capsys: pytest.CaptureFixture[str]) -> None:
    exec(compile(code, str(README), "exec"), {"__name__": "readme_example"})
    out = capsys.readouterr().out
    if "print(page.paged_in)" in code:
        assert "True False" in out
        assert "['t0']" in out


RESULTS = sorted((README.parent / "bench" / "results").glob("*.json"))
MAIN_RESULTS = README.parent / "bench" / "results" / "locomo-replay.json"


def test_readme_tables_are_the_reports_of_the_committed_results() -> None:
    # The tables are pasted from `vcmem report`; regenerating them from the committed
    # results must give the same text, so a re-run that is not pasted fails here.
    text = README.read_text(encoding="utf-8")
    results = json.loads(MAIN_RESULTS.read_text(encoding="utf-8"))
    budgets = results["config"]["budgets"]
    for table in (
        recall_table(results, "final"),
        recall_table(results, "fresh"),
        category_table(results, "final", budgets[len(budgets) // 2]),
        work_table(results),
    ):
        assert table in text
    assert results["git_sha"][:7] in text
    assert not results["git_dirty"]
    rows = set(bootstrap_table(results).splitlines())
    quoted = [line for line in text.splitlines() if line.startswith(("| final |", "| fresh |"))]
    assert quoted and set(quoted) <= rows
    # latency may come from another committed run; it must be quoted whole from one
    latency = [json.loads(p.read_text(encoding="utf-8")) for p in RESULTS]
    assert any(latency_table(r) in text for r in latency)


def test_readme_links_only_committed_results() -> None:
    linked = set(re.findall(r"bench/results/([\w.-]+\.json)", README.read_text(encoding="utf-8")))
    assert linked and linked <= {p.name for p in RESULTS}
