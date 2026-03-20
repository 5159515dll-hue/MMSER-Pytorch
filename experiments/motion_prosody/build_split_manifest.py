from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()

from experiments.motion_prosody.manifest_utils import build_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a split manifest for motion_prosody experiments")
    parser.add_argument("--data-root", type=Path, default=Path("databases"))
    parser.add_argument("--xlsx", type=Path, default=Path("databases/video_databases.xlsx"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/motion_prosody/splits/default_split_manifest.json"),
    )
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="stratified_random_by_label",
        choices=["stratified_random_by_label", "stratified_sequential_by_label"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy = "stratified_random_by_label"
    if args.split_strategy == "stratified_sequential_by_label":
        strategy = "stratified_sequential_by_label"

    manifest = build_split_manifest(
        data_root=args.data_root.expanduser(),
        xlsx=args.xlsx.expanduser(),
        train_split=float(args.train_split),
        seed=int(args.seed),
        split_strategy=strategy,
    )
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = manifest.get("summary", {})
    print(f"Wrote split manifest -> {output_path}")
    print(
        json.dumps(
            {
                "manifest_sha256": manifest.get("manifest_sha256"),
                "usable_rows": summary.get("usable_rows"),
                "raw_usable_rows": summary.get("raw_usable_rows"),
                "split_counts": summary.get("split_counts"),
                "fatal_confound": summary.get("fatal_confound"),
                "speaker_group_split_feasible": summary.get("speaker_group_split_feasible"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
