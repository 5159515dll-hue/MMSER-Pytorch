"""把散文件主线输入缓存合并成 shard 版目录。

这个脚本不会修改原始 cache 目录，而是在旁边生成一个新的 shard cache。
训练/推理主线兼容读取旧的 per-sample cache 和新的 sharded cache。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from input_cache import (
    INPUT_CACHE_SHARD_FORMAT_VERSION,
    INPUT_CACHE_STORAGE_PER_SAMPLE,
    INPUT_CACHE_STORAGE_SHARDED,
    load_input_cache_entry_payload,
    load_input_cache_index,
    load_input_cache_meta,
    save_input_cache_index,
    save_input_cache_meta,
    shard_relpath_for_index,
)


def parse_args() -> argparse.Namespace:
    """定义并解析 shard 转换参数。"""

    p = argparse.ArgumentParser(description="Merge mainline input-cache sample files into shard files")
    p.add_argument("cache_dirs", nargs="+", type=Path, help="One or more existing input-cache directories")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Only valid when converting a single cache dir; defaults to <cache_dir>_sharded",
    )
    p.add_argument(
        "--output-suffix",
        type=str,
        default="_sharded",
        help="Suffix appended to each input cache dir name when --output-dir is not given",
    )
    p.add_argument("--samples-per-shard", type=int, default=512)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace, input_dir: Path) -> Path:
    """解析某个输入目录对应的输出 shard 目录。"""

    if args.output_dir is not None:
        if len(args.cache_dirs) != 1:
            raise RuntimeError("--output-dir can only be used when exactly one cache dir is provided.")
        return args.output_dir.expanduser()
    return input_dir.parent / f"{input_dir.name}{str(args.output_suffix)}"


def _prepare_output_dir(out_dir: Path, *, overwrite: bool) -> None:
    """检查并准备输出目录。"""

    if out_dir.exists():
        if not bool(overwrite) and any(out_dir.iterdir()):
            raise RuntimeError(f"Output shard cache dir already exists and is not empty: {out_dir}")
        if bool(overwrite):
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _flush_shard(
    *,
    out_dir: Path,
    shard_idx: int,
    shard_payloads: list[dict[str, Any]],
    shard_keys: list[str],
    pending_entries: list[dict[str, Any]],
    out_entries: list[dict[str, Any]],
) -> None:
    """把当前累计的一批样本写成一个 shard。"""

    if not shard_payloads:
        return
    import torch

    relpath = shard_relpath_for_index(shard_idx)
    path = out_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": INPUT_CACHE_SHARD_FORMAT_VERSION,
            "cache_keys": list(shard_keys),
            "payloads": list(shard_payloads),
        },
        path,
    )
    for sample_idx, entry in enumerate(pending_entries):
        row = dict(entry)
        row.pop("relpath", None)
        row.pop("shard_relpath", None)
        row.pop("shard_index", None)
        row["shard_relpath"] = str(relpath)
        row["shard_index"] = int(sample_idx)
        out_entries.append(row)


def _convert_one_cache(input_dir: Path, out_dir: Path, *, samples_per_shard: int) -> dict[str, Any]:
    """把单个 cache 目录转换成 shard 目录。"""

    if int(samples_per_shard) <= 0:
        raise RuntimeError(f"--samples-per-shard must be > 0, got {samples_per_shard}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input cache dir not found: {input_dir}")

    meta = load_input_cache_meta(input_dir)
    entries = load_input_cache_index(input_dir)
    shard_payloads: list[dict[str, Any]] = []
    shard_keys: list[str] = []
    pending_entries: list[dict[str, Any]] = []
    out_entries: list[dict[str, Any]] = []
    shard_count = 0

    from tqdm import tqdm

    for entry in tqdm(entries, desc=f"Shard cache {input_dir.name}", unit="sample"):
        payload = load_input_cache_entry_payload(input_dir, entry)
        shard_payloads.append(payload)
        shard_keys.append(str(entry.get("cache_key", "")))
        pending_entries.append(dict(entry))
        if len(shard_payloads) >= int(samples_per_shard):
            _flush_shard(
                out_dir=out_dir,
                shard_idx=shard_count,
                shard_payloads=shard_payloads,
                shard_keys=shard_keys,
                pending_entries=pending_entries,
                out_entries=out_entries,
            )
            shard_count += 1
            shard_payloads = []
            shard_keys = []
            pending_entries = []

    if shard_payloads:
        _flush_shard(
            out_dir=out_dir,
            shard_idx=shard_count,
            shard_payloads=shard_payloads,
            shard_keys=shard_keys,
            pending_entries=pending_entries,
            out_entries=out_entries,
        )
        shard_count += 1

    out_meta = dict(meta)
    out_meta["storage_format"] = INPUT_CACHE_STORAGE_SHARDED
    out_meta["source_storage_format"] = str(meta.get("storage_format", INPUT_CACHE_STORAGE_PER_SAMPLE))
    out_meta["samples_per_shard"] = int(samples_per_shard)
    out_meta["shard_count"] = int(shard_count)
    save_input_cache_meta(out_dir, out_meta)
    save_input_cache_index(out_dir, out_entries)
    return {
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "sample_count": int(len(out_entries)),
        "shard_count": int(shard_count),
        "samples_per_shard": int(samples_per_shard),
        "source_storage_format": str(meta.get("storage_format", INPUT_CACHE_STORAGE_PER_SAMPLE)),
    }


def main() -> None:
    """主入口。"""

    args = parse_args()
    summaries: list[dict[str, Any]] = []
    for raw_input_dir in args.cache_dirs:
        input_dir = raw_input_dir.expanduser()
        out_dir = _resolve_output_dir(args, input_dir)
        _prepare_output_dir(out_dir, overwrite=bool(args.overwrite))
        summary = _convert_one_cache(
            input_dir,
            out_dir,
            samples_per_shard=int(args.samples_per_shard),
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    print(
        json.dumps(
            {
                "converted": len(summaries),
                "outputs": [summary["output_dir"] for summary in summaries],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
