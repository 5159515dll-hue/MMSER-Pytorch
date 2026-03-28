"""构建当前主线可复用的输入缓存。

这一步设计给 CPU 服务器执行：
1. 读取 manifest
2. 预加载音频波形
3. 预采样视频帧
4. 预分词 full / masked 文本
5. 把结果写成可搬运到 GPU 服务器的缓存目录

GPU 服务器训练/推理时只需要读取这些缓存文件，不再承担大部分音频/视频解码开销。
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from hf_compat import ensure_transformers_torch_compat
from hf_loading import resolve_hf_pretrained_source
from input_cache import (
    INPUT_CACHE_PROTOCOL_VERSION,
    build_cached_media_payload,
    manifest_item_cache_key,
    sample_relpath_for_key,
    save_input_cache_index,
    save_input_cache_meta,
)
from manifest_utils import load_split_manifest, manifest_sha256, select_manifest_items


def parse_args() -> argparse.Namespace:
    """定义并解析输入缓存构建参数。"""

    p = argparse.ArgumentParser(description="Build the paper-grade-equivalent mainline input cache")
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--subset", type=str, default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--max-audio-sec", type=float, default=6.0)
    p.add_argument("--text-model", type=str, default="FacebookAI/xlm-roberta-large")
    p.add_argument("--max-text-len", type=int, default=128)
    p.add_argument("--include-video", action="store_true")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--audio-backend", type=str, default="auto", choices=["auto", "torchaudio", "soundfile"])
    p.add_argument("--video-decode-backend", type=str, default="auto", choices=["auto", "decord", "cpu"])
    p.add_argument("--num-workers", type=str, default="auto")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_tokenizer(model_name: str) -> Any:
    """加载缓存构建用 tokenizer。"""

    ensure_transformers_torch_compat()
    from transformers import AutoTokenizer

    source, load_kwargs = resolve_hf_pretrained_source(str(model_name))
    return AutoTokenizer.from_pretrained(source, **load_kwargs)


def _clone_token_row(enc: dict[str, Any], idx: int) -> dict[str, Any]:
    """从 batched tokenizer 输出里取出单个样本的 token。"""

    row = {
        "input_ids": enc["input_ids"][idx].to("cpu", dtype=enc["input_ids"].dtype).clone(),
        "attention_mask": enc["attention_mask"][idx].to("cpu", dtype=enc["attention_mask"].dtype).clone(),
    }
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        row["token_type_ids"] = token_type_ids[idx].to("cpu", dtype=token_type_ids.dtype).clone()
    return row


def _tokenize_manifest_items(
    items: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_text_len: int,
) -> dict[str, dict[str, Any]]:
    """一次性把 manifest 条目的 full / masked 文本都分好词。"""

    full_texts = [str(item.get("mn", item.get("text", ""))) for item in items]
    masked_texts = [str(item.get("masked_mn", item.get("masked_text", ""))) for item in items]
    full_enc = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=int(max_text_len),
        return_tensors="pt",
    )
    masked_enc = tokenizer(
        masked_texts,
        padding=True,
        truncation=True,
        max_length=int(max_text_len),
        return_tensors="pt",
    )

    token_map: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(items):
        key = manifest_item_cache_key(item)
        token_map[key] = {
            "text_full": _clone_token_row(full_enc, idx),
            "text_masked": _clone_token_row(masked_enc, idx),
        }
    return token_map


def _prepare_tasks(
    items: list[dict[str, Any]],
    *,
    token_map: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    dataset_kind: str,
) -> list[dict[str, Any]]:
    """把 manifest 条目展开成 worker 可处理的任务描述。"""

    tasks: list[dict[str, Any]] = []
    for item in items:
        key = manifest_item_cache_key(item)
        audio_path = str(item.get("audio_path") or "")
        if not audio_path:
            raise RuntimeError(f"Manifest item missing audio_path: {key}")
        task = {
            "cache_key": key,
            "split": str(item.get("split", "")),
            "seq": str(item.get("seq", "")),
            "sample_id": str(item.get("sample_id", "")),
            "speaker_id": str(item.get("speaker_id", "")),
            "label_en": str(item.get("label_en", "")),
            "dataset_kind": str(item.get("dataset_kind", dataset_kind) or dataset_kind),
            "need_audio": True,
            "need_video": bool(args.include_video),
            "audio_path": audio_path,
            "sample_rate": int(args.sample_rate),
            "max_audio_sec": float(args.max_audio_sec),
            "audio_backend_mode": str(args.audio_backend),
            "video_decode_backend": str(args.video_decode_backend),
            "num_frames": int(args.num_frames),
            "text_full": token_map[key]["text_full"],
            "text_masked": token_map[key]["text_masked"],
        }
        if bool(args.include_video):
            video_path = str(item.get("video_path") or "")
            if not video_path:
                raise RuntimeError(f"Manifest item missing video_path while --include-video is enabled: {key}")
            task["video_path"] = video_path
        tasks.append(task)
    return tasks


def main() -> None:
    """主缓存构建入口。"""

    args = parse_args()
    split_manifest_path = args.split_manifest.expanduser()
    out_dir = args.output_dir.expanduser()
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")

    if out_dir.exists():
        if not bool(args.overwrite) and any(out_dir.iterdir()):
            raise RuntimeError(f"Output cache dir already exists and is not empty: {out_dir}")
        if bool(args.overwrite):
            shutil.rmtree(out_dir)

    manifest = load_split_manifest(split_manifest_path)
    manifest_hash = manifest_sha256(split_manifest_path)
    dataset_kind = str(manifest.get("dataset_kind", "") or "")
    items = select_manifest_items(manifest, str(args.subset))
    if int(args.limit) > 0:
        items = items[: int(args.limit)]
    if not items:
        raise RuntimeError("No usable manifest items selected for cache build.")

    tokenizer = _load_tokenizer(str(args.text_model))
    token_map = _tokenize_manifest_items(items, tokenizer=tokenizer, max_text_len=int(args.max_text_len))
    tasks = _prepare_tasks(items, token_map=token_map, args=args, dataset_kind=dataset_kind)

    from runtime_adapt import detect_runtime, resolve_worker_count

    profile = detect_runtime("cpu")
    resolved_workers = resolve_worker_count(
        args.num_workers,
        phase="predecode",
        profile=profile,
        dataset_in_memory=False,
        total_items=len(tasks),
    )
    print(
        json.dumps(
            {
                "selected_items": len(tasks),
                "resolved_num_workers": int(resolved_workers),
                "include_video": bool(args.include_video),
                "num_frames": int(args.num_frames) if bool(args.include_video) else 0,
                "sample_rate": int(args.sample_rate),
                "max_audio_sec": float(args.max_audio_sec),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    import torch
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    def _write_result(result: dict[str, Any]) -> None:
        key = str(result["cache_key"])
        relpath = sample_relpath_for_key(key)
        path = out_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result["payload"], path)
        entries.append(
            {
                "cache_key": key,
                "relpath": str(relpath),
                "sample_bytes": int(result.get("sample_bytes", 0) or 0),
                "split": str(result["payload"].get("meta", {}).get("split", "")),
                "seq": str(result["payload"].get("meta", {}).get("seq", "")),
                "sample_id": str(result["payload"].get("meta", {}).get("sample_id", "")),
                "has_audio": "audio" in result["payload"],
                "has_video": "video_frames" in result["payload"],
            }
        )

    if int(resolved_workers) <= 1:
        for task in tqdm(tasks, desc="Build input cache", unit="sample"):
            _write_result(build_cached_media_payload(task))
    else:
        with ProcessPoolExecutor(max_workers=int(resolved_workers)) as executor:
            for result in tqdm(executor.map(build_cached_media_payload, tasks), total=len(tasks), desc="Build input cache", unit="sample"):
                _write_result(result)

    meta = {
        "protocol_version": INPUT_CACHE_PROTOCOL_VERSION,
        "split_manifest": str(split_manifest_path),
        "manifest_sha256": str(manifest_hash),
        "dataset_kind": str(dataset_kind),
        "subset": str(args.subset),
        "sample_count": int(len(entries)),
        "sample_rate": int(args.sample_rate),
        "max_audio_sec": float(args.max_audio_sec),
        "num_frames": int(args.num_frames) if bool(args.include_video) else 0,
        "text_model": str(args.text_model),
        "max_text_len": int(args.max_text_len),
        "has_audio": True,
        "has_video": bool(args.include_video),
        "has_text_full_tokens": True,
        "has_text_masked_tokens": True,
        "audio_backend_mode": str(args.audio_backend),
        "video_decode_backend": str(args.video_decode_backend),
        "runtime_profile": profile.to_jsonable(),
    }
    save_input_cache_meta(out_dir, meta)
    save_input_cache_index(out_dir, entries)
    print(f"Wrote input cache -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
