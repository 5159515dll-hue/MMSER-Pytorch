"""构建主线 split manifest 的命令行入口。

主线训练与推理不再直接扫描磁盘数据集，而是先把样本、标签、路径和划分
规则固定为一份 manifest。这个脚本负责生成那份“训练事实来源”。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    """在脚本直跑时把仓库根目录加入 `sys.path`。"""

    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()

from manifest_utils import DATASET_KINDS, build_split_manifest
from path_utils import default_databases_dir, default_xlsx_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build a split manifest for motion_prosody experiments")
    parser.add_argument("--dataset-kind", type=str, default="mongolian_xlsx", choices=list(DATASET_KINDS))
    parser.add_argument("--data-root", type=Path, default=default_databases_dir(repo_root))
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help="Optional metadata root for datasets whose annotations live outside --data-root (e.g. MELD CSV files).",
    )
    parser.add_argument("--xlsx", type=Path, default=default_xlsx_path(repo_root, "video_databases.xlsx"))
    parser.add_argument(
        "--audio-cache-root",
        type=Path,
        default=Path("outputs/datasets/meld/audio_cache"),
        help="Expected audio sidecar root for datasets that require audio extraction (used by MELD manifests).",
    )
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
    """构建 manifest 并把摘要打印到终端。

    这个入口的作用不是训练，而是先把数据划分和元数据事实固定下来，
    让后续训练/推理都围绕同一份 manifest 运行。
    """

    args = parse_args()
    strategy = "stratified_random_by_label"
    if args.split_strategy == "stratified_sequential_by_label":
        strategy = "stratified_sequential_by_label"

    manifest = build_split_manifest(
        dataset_kind=str(args.dataset_kind),
        data_root=args.data_root.expanduser(),
        metadata_root=args.metadata_root.expanduser() if args.metadata_root is not None else None,
        xlsx=args.xlsx.expanduser() if args.dataset_kind == "mongolian_xlsx" else None,
        audio_cache_root=args.audio_cache_root.expanduser() if args.dataset_kind == "meld" else None,
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
                "dataset_kind": manifest.get("dataset_kind"),
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
