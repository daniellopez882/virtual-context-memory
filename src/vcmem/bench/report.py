"""Markdown tables from a results JSON, so README numbers are pasted, never retyped."""

from __future__ import annotations

from typing import Any

from vcmem.locomo import CATEGORY_NAMES


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _rows(results: dict[str, Any], probe: str) -> dict[tuple[str, int], dict[str, Any]]:
    return {(r["method"], r["budget"]): r for r in results["results"] if r["probe"] == probe}


def recall_table(results: dict[str, Any], probe: str) -> str:
    rows = _rows(results, probe)
    budgets = results["config"]["budgets"]
    lines = [
        "| method | " + " | ".join(f"{b} tokens" for b in budgets) + " |",
        "|---|" + "---:|" * len(budgets),
    ]
    for method in results["config"]["methods"]:
        cells = []
        for b in budgets:
            r = rows.get((method, b))
            cells.append(
                "n/a"
                if r is None
                else f"{_pct(r['evidence_recall'])} ({_pct(r['evidence_recall_any'])})"
            )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def category_table(results: dict[str, Any], probe: str, budget: int) -> str:
    rows = _rows(results, probe)
    cats = list(CATEGORY_NAMES.values())
    lines = ["| method | " + " | ".join(cats) + " |", "|---|" + "---:|" * len(cats)]
    for method in results["config"]["methods"]:
        r = rows.get((method, budget))
        if r is None:
            continue
        cells = [
            _pct(r["by_category"][c]["evidence_recall"]) if c in r["by_category"] else "n/a"
            for c in cats
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cost_table(results: dict[str, Any]) -> str:
    lines = [
        "| method | budget | turn p50 ms | turn p95 ms | question p50 ms | question p95 ms "
        "| SQLite KiB per conversation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results["results"]:
        if r["probe"] != "final":
            continue
        turn50 = f"{r['turn_ms_p50']:.2f}" if "turn_ms_p50" in r else "-"
        turn95 = f"{r['turn_ms_p95']:.2f}" if "turn_ms_p95" in r else "-"
        storage = f"{r['storage_kib_mean']:.1f}" if "storage_kib_mean" in r else "-"
        lines.append(
            f"| {r['method']} | {r['budget']} | {turn50} | {turn95} | "
            f"{r['question_ms_p50']:.2f} | {r['question_ms_p95']:.2f} | {storage} |"
        )
    return "\n".join(lines)


def render_report(results: dict[str, Any]) -> str:
    m = results["machine"]
    budgets = results["config"]["budgets"]
    mid = budgets[len(budgets) // 2]
    return "\n".join(
        [
            f"Measured on {m['cpu']}, {m['logical_cores']} logical cores, {m['ram_gb']} GB RAM, "
            f"{m['os']}, Python {m['python']}, {results['date'][:10]}, commit "
            f"{results['git_sha'][:10]}.",
            "",
            "Evidence recall after the whole conversation (final probe), %. In brackets: "
            "also counting evidence linked from summary, core-memory and archival lines.",
            "",
            recall_table(results, "final"),
            "",
            "Evidence recall right after the session holding the last evidence (fresh probe), %",
            "",
            recall_table(results, "fresh"),
            "",
            f"Final-probe evidence recall by question category at {mid} tokens, %",
            "",
            category_table(results, "final", mid),
            "",
            "Cost (latency per replayed turn and per question; SQLite size after replay)",
            "",
            cost_table(results),
        ]
    )
