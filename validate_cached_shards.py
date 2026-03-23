from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

from runtime_adapt import detect_runtime, resolve_worker_count


def parse_args() -> argparse.Namespace:
    """解析缓存校验命令行参数。"""

    p = argparse.ArgumentParser(description="Validate cached .pt shards (raw or feature cache)")
    p.add_argument(
        "--cached-dataset",
        type=Path,
        required=True,
        help="Path to cached shard dir (contains N.pt) or a single shard .pt file",
    )
    p.add_argument(
        "--cache-kind",
        type=str,
        default="auto",
        choices=["auto", "raw", "feature"],
        help="Expected cache kind. auto infers from shard config and fields.",
    )
    p.add_argument(
        "--num-workers",
        type=str,
        default="auto",
        help="Validation worker count. Use auto to adapt to the current server.",
    )
    p.add_argument(
        "--max-show",
        type=int,
        default=20,
        help="Max number of corrupt shards to show",
    )
    return p.parse_args()


def iter_shards(path: Path) -> list[Path]:
    """把目录或单文件输入统一展开成 shard 路径列表。"""

    path = path.expanduser()
    if path.is_file():
        return [path]
    return sorted(path.glob("*.pt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def _detect_cache_kind(obj: Any, sample: dict[str, Any] | None) -> str:
    """从 shard config/sample 字段推断 raw 或 feature cache。"""

    if isinstance(obj, dict):
        cfg = obj.get("config", {})
        if isinstance(cfg, dict) and isinstance(cfg.get("feature_cache"), dict):
            return "feature"
    if isinstance(sample, dict):
        if any(k in sample for k in ("flow_emb", "rgb_emb", "text_emb")):
            return "feature"
    return "raw"


def _validate_sample(sample: dict[str, Any], cache_kind: str) -> None:
    """做轻量 schema 校验。"""

    required = ("prosody", "label", "stem", "mn")
    for key in required:
        if key not in sample:
            raise RuntimeError(f"sample_missing_{key}")
    prosody = sample["prosody"]
    if not isinstance(prosody, torch.Tensor):
        raise RuntimeError("prosody_not_tensor")
    if prosody.numel() <= 0:
        raise RuntimeError("prosody_empty")

    if cache_kind == "feature":
        if not any(k in sample for k in ("flow_emb", "rgb_emb", "text_emb", "audio_emb", "flow", "rgb", "audio")):
            raise RuntimeError("feature_cache_has_no_model_inputs")
        for key in ("flow_emb", "rgb_emb", "text_emb", "audio_emb"):
            if key in sample and isinstance(sample[key], torch.Tensor) and sample[key].ndim != 1:
                raise RuntimeError(f"{key}_expected_rank1")
    else:
        if not any(k in sample for k in ("flow", "rgb", "audio", "audio_emb")):
            raise RuntimeError("raw_cache_has_no_modalities")


def _validate_one_shard(path: Path, expected_kind: str) -> tuple[Path, bool, str, str]:
    """加载并校验单个 shard。"""

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "samples" in obj:
            samples = obj["samples"]
        elif isinstance(obj, list):
            samples = obj
        else:
            raise RuntimeError(f"unexpected_root_type_{type(obj).__name__}")
        if not isinstance(samples, list):
            raise RuntimeError("samples_not_list")
        first = samples[0] if samples else None
        if first is not None and not isinstance(first, dict):
            raise RuntimeError("sample_not_dict")
        detected_kind = _detect_cache_kind(obj, first if isinstance(first, dict) else None)
        if expected_kind != "auto" and detected_kind != expected_kind:
            raise RuntimeError(f"cache_kind_mismatch expected={expected_kind} got={detected_kind}")
        if isinstance(first, dict):
            _validate_sample(first, detected_kind)
        return path, True, detected_kind, ""
    except Exception as e:
        return path, False, "unknown", f"{type(e).__name__}: {e}"


def _format_bytes(num_bytes: int) -> str:
    """Render a byte size into a compact human-readable string."""

    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(0, int(num_bytes)))
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)}{unit}"
    return f"{value:.1f}{unit}"


def main() -> None:
    """并行读取 shard，检查是否损坏或格式异常。"""

    args = parse_args()
    shards = iter_shards(args.cached_dataset)
    if not shards:
        raise FileNotFoundError(f"No .pt shards found under: {args.cached_dataset}")

    profile = detect_runtime("cpu")
    num_workers = resolve_worker_count(args.num_workers, phase="validate", profile=profile, dataset_in_memory=False, total_items=len(shards))
    ok = 0
    bad: list[tuple[Path, str]] = []
    kind_counts: dict[str, int] = {"raw": 0, "feature": 0}

    print(
        f"Validating {len(shards)} shard(s) under {args.cached_dataset} "
        f"(cache_kind={args.cache_kind}, workers={int(num_workers)})",
        flush=True,
    )

    if len(shards) == 1:
        shard = shards[0]
        try:
            size_text = _format_bytes(shard.stat().st_size)
        except Exception:
            size_text = "unknown"
        print(f"Loading shard 1/1: {shard} ({size_text})", flush=True)
        path, success, detected_kind, err = _validate_one_shard(shard, str(args.cache_kind))
        if success:
            ok += 1
            if detected_kind in kind_counts:
                kind_counts[detected_kind] += 1
        else:
            bad.append((path, err))
    else:
        with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
            futures = {
                ex.submit(_validate_one_shard, shard, str(args.cache_kind)): shard
                for shard in shards
            }
            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                path, success, detected_kind, err = future.result()
                completed += 1
                print(f"Validated {completed}/{total}: {path.name}", flush=True)
                if success:
                    ok += 1
                    if detected_kind in kind_counts:
                        kind_counts[detected_kind] += 1
                else:
                    bad.append((path, err))

    print(
        f"Shards: {len(shards)} | OK: {ok} | Corrupt: {len(bad)} | "
        f"Kinds: raw={kind_counts['raw']} feature={kind_counts['feature']} | workers={int(num_workers)}",
        flush=True,
    )
    if bad:
        print("-- Corrupt shards:", flush=True)
        for p, err in bad[: max(1, int(args.max_show))]:
            print(f"  - {p} | {err}", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
