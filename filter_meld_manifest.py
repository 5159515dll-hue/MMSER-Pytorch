"""过滤 MELD manifest 中缺失媒体的样本。

当某些视频或音频 sidecar 缺失时，主线训练不应把它们继续带入数据集。
这个脚本会删除无效样本，并重算过滤后的摘要和哈希。
"""

from __future__ import annotations

import argparse
import hashlib
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

from manifest_utils import _summarize_manifest_items, load_split_manifest


def parse_args() -> argparse.Namespace:
    """解析过滤 MELD manifest 所需参数。"""

    parser = argparse.ArgumentParser(
        description="Filter a MELD manifest down to items whose media sidecars actually exist"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/benchmarks/meld/splits/default_manifest.json"),
        help="Source MELD manifest JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmarks/meld/splits/default_manifest.filtered.json"),
        help="Filtered manifest JSON to write",
    )
    parser.add_argument(
        "--allow-missing-video",
        action="store_true",
        help="Keep rows whose video_path is missing. Default behavior is to drop them.",
    )
    parser.add_argument(
        "--allow-missing-audio",
        action="store_true",
        help="Keep rows whose audio_path is missing. Default behavior is to drop them.",
    )
    return parser.parse_args()


def _path_exists(raw: object) -> bool:
    """判断 manifest 中记录的媒体路径是否真实存在。"""

    path = Path(str(raw or "")).expanduser()
    return bool(str(raw or "").strip()) and path.exists()


def main() -> None:
    """过滤缺失媒体的 MELD manifest，并重算摘要与哈希。"""

    args = parse_args()
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()

    manifest = load_split_manifest(input_path)
    items = manifest.get("items", [])

    require_video = not bool(args.allow_missing_video)
    require_audio = not bool(args.allow_missing_audio)

    kept: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    for item in items:
        ok = True
        if require_video and not _path_exists(item.get("video_path")):
            ok = False
        if require_audio and not _path_exists(item.get("audio_path")):
            ok = False
        if ok:
            kept.append(dict(item))
        else:
            dropped.append(dict(item))

    out_manifest = dict(manifest)
    out_manifest["items"] = kept
    out_manifest["summary"] = _summarize_manifest_items(kept)
    out_manifest["filtered_from"] = {
        "source_manifest": str(input_path),
        "dropped_rows": int(len(dropped)),
        "kept_rows": int(len(kept)),
        "require_video": bool(require_video),
        "require_audio": bool(require_audio),
    }
    out_manifest["manifest_sha256"] = ""
    out_manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(out_manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = out_manifest.get("summary", {})
    print(output_path)
    print(
        json.dumps(
            {
                "kept": len(kept),
                "dropped": len(dropped),
                "split_counts": summary.get("split_counts"),
                "manifest_sha256": out_manifest.get("manifest_sha256"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
