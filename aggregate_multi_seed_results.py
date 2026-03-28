from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_report_utils import (
    PAPER_MULTI_SEED,
    apply_multiple_comparison_correction,
    build_pairwise_comparison,
    ensure_group_is_paper_grade_ready,
    expand_run_dirs,
    parse_group_spec,
    render_markdown_report,
    summarize_experiment_group,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed gpu_stream experiment runs into a statistical summary."
    )
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help="Experiment group spec in the form NAME=GLOB_PATTERN. Reuse the same seed set across groups.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=list(PAPER_MULTI_SEED),
        help="Fixed seed set required for each experiment group.",
    )
    parser.add_argument(
        "--pairwise-mode",
        choices=["adjacent", "all"],
        default="adjacent",
        help="Whether to compare only adjacent groups or every ordered pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    for spec in args.group:
        name, pattern = parse_group_spec(spec)
        run_dirs = expand_run_dirs(pattern)
        group = summarize_experiment_group(
            name,
            run_dirs,
            confidence=float(args.confidence),
            expected_seeds=[int(seed) for seed in args.expected_seeds],
        )
        ensure_group_is_paper_grade_ready(group)
        groups.append(group)

    pairwise = []
    if str(args.pairwise_mode) == "all":
        for left_idx, baseline in enumerate(groups):
            for candidate in groups[left_idx + 1 :]:
                pairwise.append(
                    build_pairwise_comparison(
                        baseline,
                        candidate,
                        confidence=float(args.confidence),
                        comparison_scope="exploratory",
                    )
                )
        pairwise = apply_multiple_comparison_correction(pairwise, method="holm_bonferroni")
    else:
        for baseline, candidate in zip(groups, groups[1:]):
            pairwise.append(
                build_pairwise_comparison(
                    baseline,
                    candidate,
                    confidence=float(args.confidence),
                    comparison_scope="confirmatory",
                )
            )

    summary_payload = {
        "confidence": float(args.confidence),
        "pairwise_mode": str(args.pairwise_mode),
        "expected_seeds": [int(seed) for seed in args.expected_seeds],
        "groups": groups,
    }
    pairwise_payload = {
        "confidence": float(args.confidence),
        "pairwise_mode": str(args.pairwise_mode),
        "expected_seeds": [int(seed) for seed in args.expected_seeds],
        "comparisons": pairwise,
    }
    (output_dir / "multi_seed_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "pairwise_significance.json").write_text(
        json.dumps(pairwise_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "multi_seed_summary.md").write_text(
        render_markdown_report(groups, pairwise, confidence=float(args.confidence)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
