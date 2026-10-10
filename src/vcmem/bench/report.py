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


def _fmt(row: dict[str, Any], key: str, spec: str) -> str:
    return format(row[key], spec) if key in row else "-"


def cost_table(results: dict[str, Any]) -> str:
    """Latency per replayed turn and per final-probe question, the MiniLM texts each one
    asks for (served from the precomputed cache, so not inside the latency), and size."""
    lines = [
        "| method | budget | turn p50 ms | turn p95 ms | MiniLM texts per turn "
        "| question p50 ms | question p95 ms | MiniLM texts per question "
        "| SQLite KiB per conversation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results["results"]:
        if r["probe"] != "final":
            continue
        cells = [
            _fmt(r, "turn_ms_p50", ".2f"),
            _fmt(r, "turn_ms_p95", ".2f"),
            _fmt(r, "embed_texts_per_turn", ".2f"),
            _fmt(r, "question_ms_p50", ".2f"),
            _fmt(r, "question_ms_p95", ".2f"),
            _fmt(r, "embed_texts_per_question", ".2f"),
            _fmt(r, "storage_kib_mean", ".1f"),
        ]
        lines.append(f"| {r['method']} | {r['budget']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bootstrap_table(results: dict[str, Any]) -> str:
    """Paired differences in evidence recall, percentage points, with 95% intervals."""
    lines = [
        "| probe | budget | method | compared with | difference | 95% interval |",
        "|---|---:|---|---|---:|---|",
    ]
    for r in results.get("paired_bootstrap", {}).get("rows", []):
        lo, hi = r["ci95"]
        lines.append(
            f"| {r['probe']} | {r['budget']} | {r['a']} | {r['b']} | {100 * r['diff']:+.1f} "
            f"| {100 * lo:+.1f} to {100 * hi:+.1f} |"
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
            "Cost (latency per replayed turn and per final-probe question, MiniLM texts "
            "requested per turn and per question, SQLite size after replay)",
            "",
            cost_table(results),
            "",
            "Paired differences in evidence recall, percentage points; 95% intervals from "
            "resampling whole conversations",
            "",
            bootstrap_table(results),
        ]
    )
