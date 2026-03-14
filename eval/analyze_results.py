"""Analyze eval results JSONL and report all problematic cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval._utils import truncate

JUDGE_QUALITY_THRESHOLD = 4.0
SNIPPET_TRUNCATE_LEN = 120
COMMENT_TRUNCATE_LEN = 280


def fmt(v: object, decimals: int = 4) -> str:
    """Format a numeric value as a fixed-point string.

    Args:
        v: The value to format. Any non-numeric value is returned as str(v).
        decimals: Number of decimal places.

    Returns:
        A formatted string, or ``'None'`` when ``v`` is ``None``.

    """
    if v is None:
        return "None"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def parse_metrics(results: list) -> dict:
    """Convert the nested results list into a flat ``{key: kv_dict}`` mapping.

    Args:
        results: The raw ``evaluators.results.value`` list from one case.

    Returns:
        A dict mapping each metric key to its full key-value pair dict.

    """
    metrics: dict = {}
    for item in results:
        kv = {pair[0]: pair[1] for pair in item}
        k = kv.get("key")
        if k:
            metrics[k] = kv
    return metrics


def parse_json_value(raw: object) -> object:
    """Safely parse a JSON-encoded ``value`` field.

    Args:
        raw: A string, list, dict, or ``None`` from the ``value`` field.

    Returns:
        The parsed Python object, the original ``raw`` if it is not a string,
        or ``None`` when ``raw`` is ``None``.

    """
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def detect_problems(metrics: dict) -> list[tuple]:
    """Return problem tuples for a single case based on the three criteria.

    Criteria checked:
    - ``info_reference_subset_pass_rate`` < 1.0 (and the metric is not skipped).
    - ``judge_consumer_quality`` < 4.0.
    - ``judge_grounding_faithfulness`` == 0.0.

    Args:
        metrics: Flat metric dict produced by :func:`parse_metrics`.

    Returns:
        A list of ``(metric_name, score, turns, failures)`` tuples for each
        triggered criterion. ``turns`` and ``failures`` are ``None`` for the
        judge metrics.

    """
    problems: list[tuple] = []

    info_skipped = (metrics.get("info_reference_subset_skipped") or {}).get("score")
    info_pass_rate = (
        metrics.get("info_reference_subset_pass_rate") or {}
    ).get("score")
    info_turns = (metrics.get("info_reference_subset_turns") or {}).get("score")
    info_failures_raw = (
        metrics.get("info_reference_subset_failures") or {}
    ).get("value")
    info_failures = parse_json_value(info_failures_raw) or []

    if info_skipped != 1.0 and info_pass_rate is not None and info_pass_rate < 1.0:
        problems.append(
            ("info_reference_subset_pass", info_pass_rate, info_turns, info_failures),
        )

    judge_quality = (metrics.get("judge_consumer_quality") or {}).get("score")
    if judge_quality is not None and judge_quality < JUDGE_QUALITY_THRESHOLD:
        problems.append(("judge_consumer_quality", judge_quality, None, None))

    judge_faith = (metrics.get("judge_grounding_faithfulness") or {}).get("score")
    if judge_faith is not None and judge_faith == 0.0:
        problems.append(("judge_grounding_faithfulness", judge_faith, None, None))

    return problems


def build_case_record(case_id: str, metrics: dict, problems: list[tuple]) -> dict:
    """Assemble a structured record for one problematic case.

    Args:
        case_id: The case identifier string (e.g. ``'case_0055'``).
        metrics: Flat metric dict from :func:`parse_metrics`.
        problems: Problem tuples from :func:`detect_problems`.

    Returns:
        A dict with all relevant scores, comments, RAG per-turn data, and the
        detected problem list.

    """
    def score(key: str) -> object:
        return (metrics.get(key) or {}).get("score")

    def comment(key: str) -> str:
        return (metrics.get(key) or {}).get("comment") or ""

    rag_per_turn_raw = (metrics.get("rag_per_turn") or {}).get("value")
    rag_per_turn = parse_json_value(rag_per_turn_raw) or []

    info_failures_raw = (
        metrics.get("info_reference_subset_failures") or {}
    ).get("value")
    info_failures = parse_json_value(info_failures_raw) or []

    return {
        "case_id": case_id,
        "problems": problems,
        "info_pass_rate": score("info_reference_subset_pass_rate"),
        "info_turns": score("info_reference_subset_turns"),
        "info_failures": info_failures,
        "info_skipped": score("info_reference_subset_skipped"),
        "judge_quality": score("judge_consumer_quality"),
        "judge_quality_comment": comment("judge_consumer_quality"),
        "judge_faith": score("judge_grounding_faithfulness"),
        "judge_faith_comment": comment("judge_grounding_faithfulness"),
        "rag_faith": score("rag_faithfulness_mean"),
        "rag_relevancy": score("rag_answer_relevancy_mean"),
        "rag_per_turn": rag_per_turn,
        "rag_skipped": score("rag_metrics_skipped"),
    }


def _render_problems(p: dict, out: list[str]) -> None:
    """Append the Problems section lines to ``out``.

    Args:
        p: Case record dict from :func:`build_case_record`.
        out: Accumulator list of output lines.

    """
    out.append("  Problems:")
    for prob in p["problems"]:
        metric = prob[0]
        if metric == "info_reference_subset_pass":
            rate, turns, failures = prob[1], prob[2], prob[3]
            turns_str = int(turns) if turns else "?"
            out.append(
                f"    - info_reference_subset_pass: pass_rate={fmt(rate)} "
                f"across {turns_str} turns",
            )
            for fail in (failures or []):
                ti = fail.get("turn_index")
                missing = fail.get("missing_snippets", [])
                expected = fail.get("expected_snippets", [])
                n_miss = len(missing)
                n_exp = len(expected)
                out.append(
                    f"      Turn {ti}: {n_miss}/{n_exp} snippets missing",
                )
                out.extend(
                    f'        MISSING: "{truncate(s, SNIPPET_TRUNCATE_LEN)}"'
                    for s in missing
                )
        elif metric == "judge_consumer_quality":
            out.append(
                f"    - judge_consumer_quality: {prob[1]} "
                f"(threshold < {JUDGE_QUALITY_THRESHOLD})",
            )
        elif metric == "judge_grounding_faithfulness":
            out.append(
                f"    - judge_grounding_faithfulness: {prob[1]} (== 0.0, flagged)",
            )


def _render_info_score(p: dict, out: list[str]) -> None:
    """Append the info_reference_subset score line(s) to ``out``.

    Args:
        p: Case record dict.
        out: Accumulator list of output lines.

    """
    if p["info_skipped"] == 1.0:
        out.append(
            "    - info_reference_subset_pass: "
            "SKIPPED (no turns with reference snippets)",
        )
    else:
        turns_str = int(p["info_turns"]) if p["info_turns"] else "?"
        out.append(
            f"    - info_reference_subset_pass_rate: {fmt(p['info_pass_rate'])} "
            f"(over {turns_str} turns)",
        )


def _render_judge_scores(p: dict, out: list[str]) -> None:
    """Append judge metric score lines to ``out``.

    Args:
        p: Case record dict.
        out: Accumulator list of output lines.

    """
    out.append(f"    - judge_consumer_quality: {p['judge_quality']}")
    if p["judge_quality_comment"]:
        cmt = truncate(p["judge_quality_comment"], COMMENT_TRUNCATE_LEN)
        out.append(f"      -> {cmt}")

    out.append(f"    - judge_grounding_faithfulness: {p['judge_faith']}")
    if p["judge_faith_comment"]:
        cmt = truncate(p["judge_faith_comment"], COMMENT_TRUNCATE_LEN)
        out.append(f"      -> {cmt}")


def _render_rag_scores(p: dict, out: list[str]) -> None:
    """Append RAG metric score lines to ``out``.

    Args:
        p: Case record dict.
        out: Accumulator list of output lines.

    """
    if p["rag_skipped"] == 1.0:
        out.append("    - rag_answer_relevancy: SKIPPED (no RAG/info turns)")
        out.append("    - rag_faithfulness: SKIPPED (no RAG/info turns)")
        return

    out.append(f"    - rag_answer_relevancy_mean: {fmt(p['rag_relevancy'])}")
    out.append(f"    - rag_faithfulness_mean: {fmt(p['rag_faith'])}")
    if p["rag_per_turn"]:
        out.append("    - rag per-turn breakdown:")
        for t in p["rag_per_turn"]:
            ti = t.get("turn_index")
            ar = t.get("answer_relevancy")
            fa = t.get("faithfulness")
            cp = t.get("context_precision")
            cr = t.get("context_recall")
            out.append(
                f"      turn_{ti}: faithfulness={fmt(fa)}, "
                f"answer_relevancy={fmt(ar)}, "
                f"context_precision={fmt(cp)}, "
                f"context_recall={fmt(cr)}",
            )


def render_case(p: dict) -> list[str]:
    """Render a single problematic case as a list of report lines.

    Args:
        p: Case record dict from :func:`build_case_record`.

    Returns:
        A list of strings (one per output line) for this case.

    """
    out: list[str] = []
    out.append("=" * 70)
    out.append(f"CASE: {p['case_id']}")
    _render_problems(p, out)
    out.append("  All metric scores:")
    _render_info_score(p, out)
    _render_judge_scores(p, out)
    _render_rag_scores(p, out)
    out.append("")
    return out


def main() -> None:
    """Parse the eval results JSONL and write a structured report to stdout.

    Reads the JSONL file at the path given on the command line, flags cases
    that fail any of the three problem criteria, and prints per-case scores
    alongside an overall summary.
    """
    parser = argparse.ArgumentParser(
        description="Analyze eval results JSONL and report all problematic cases.",
    )
    parser.add_argument(
        "results",
        metavar="PATH",
        help="Path to results.jsonl produced by run_experiment.py.",
    )
    args = parser.parse_args()
    results_path = Path(args.results)

    all_problems: list[dict] = []
    total = 0

    with results_path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            total += 1
            case_id = d["case_id"]
            results = d["evaluators"]["results"]["value"]
            metrics = parse_metrics(results)
            problems = detect_problems(metrics)
            if problems:
                record = build_case_record(case_id, metrics, problems)
                all_problems.append(record)

    all_problems.sort(key=lambda x: x["case_id"])

    lines: list[str] = []
    lines.append(f"Total cases examined: {total}")
    lines.append(f"Problematic cases found: {len(all_problems)}")
    lines.append("")

    for p in all_problems:
        lines.extend(render_case(p))

    lines.append("=" * 70)
    lines.append("FINAL SUMMARY")
    lines.append(f"  Total cases examined: {total}")
    lines.append(f"  Problematic cases:    {len(all_problems)}")

    n_info = sum(
        1 for p in all_problems
        if any(x[0] == "info_reference_subset_pass" for x in p["problems"])
    )
    n_quality = sum(
        1 for p in all_problems
        if any(x[0] == "judge_consumer_quality" for x in p["problems"])
    )
    n_faith = sum(
        1 for p in all_problems
        if any(x[0] == "judge_grounding_faithfulness" for x in p["problems"])
    )
    lines.append(f"  - info_reference_subset_pass < 1.0:    {n_info} cases")
    lines.append(f"  - judge_consumer_quality < 4.0:        {n_quality} cases")
    lines.append(f"  - judge_grounding_faithfulness == 0.0: {n_faith} cases")

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
