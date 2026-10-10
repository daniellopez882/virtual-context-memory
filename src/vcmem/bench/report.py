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


def _column_table(
    results: dict[str, Any], columns: list[tuple[str, str, str]], probe: str = "final"
) -> str:
    """One row per method and budget; columns no row has are left out, and so are rows
    with none of the remaining columns (older results files lack some fields)."""
    rows = [r for r in results["results"] if r["probe"] == probe]
    present = [(h, k, f) for h, k, f in columns if any(k in r for r in rows)]
    if not present:
        return ""
    lines = [
        "| method | budget | " + " | ".join(h for h, _, _ in present) + " |",
        "|---|---:|" + "---:|" * len(present),
    ]
    for r in rows:
        if not any(k in r for _, k, _ in present):
            continue
        cells = [format(r[k], f) if k in r else "-" for _, k, f in present]
        lines.append(f"| {r['method']} | {r['budget']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def work_table(results: dict[str, Any]) -> str:
    """Deterministic work per replay: MiniLM texts a live deployment would encode per
    turn and per final-probe question, and the SQLite file size."""
    return _column_table(
        results,
        [
            ("MiniLM texts per turn", "embed_texts_per_turn", ".2f"),
            ("MiniLM texts per final-probe question", "embed_texts_per_question", ".2f"),
            ("SQLite KiB per conversation", "storage_kib_mean", ".1f"),
        ],
    )


def latency_table(results: dict[str, Any]) -> str:
    """Latency per replayed turn and per final-probe question. The replay serves vectors
    from a cache filled before it starts, so these leave out the encoder; the estimate
    columns price the texts in ``work_table`` at the run's measured cost per text."""
    return _column_table(
        results,
        [
            ("turn p50 ms", "turn_ms_p50", ".2f"),
            ("turn p95 ms", "turn_ms_p95", ".2f"),
            ("encoder ms per turn (est.)", "encoder_ms_per_turn_est", ".2f"),
            ("final-probe question p50 ms", "question_ms_p50", ".2f"),
            ("final-probe question p95 ms", "question_ms_p95", ".2f"),
            ("encoder ms per question (est.)", "encoder_ms_per_question_est", ".2f"),
        ],
    )


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
    sections = [
        (
            "Evidence recall after the whole conversation (final probe), %. In brackets: "
            "also counting evidence linked from summary, core-memory and archival lines.",
            recall_table(results, "final"),
        ),
        (
            "Evidence recall right after the session holding the last evidence (fresh probe), %",
            recall_table(results, "fresh"),
        ),
        (
            f"Final-probe evidence recall by question category at {mid} tokens, %",
            category_table(results, "final", mid),
        ),
        ("Work per replay (deterministic counts)", work_table(results)),
        (
            "Latency per replayed turn and per final-probe question; encoder estimates at "
            f"{results.get('embedding', {}).get('ms_per_text', '?')} ms per text",
            latency_table(results),
        ),
    ]
    if results.get("paired_bootstrap", {}).get("rows"):
        sections.append(
            (
                "Paired differences in evidence recall, percentage points; 95% intervals "
                "from resampling whole conversations",
                bootstrap_table(results),
            )
        )
    out = [
        f"Measured on {m['cpu']}, {m['logical_cores']} logical cores, {m['ram_gb']} GB RAM, "
        f"{m['os']}, Python {m['python']}, {results['date'][:10]}, commit "
        f"{results['git_sha'][:10]}."
    ]
    for title, table in sections:
        if table:
            out += ["", title, "", table]
    return "\n".join(out)
