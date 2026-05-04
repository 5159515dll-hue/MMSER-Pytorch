"""Build exploratory frozen-backbone embeddings for the mainline pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CachedManifestDataset, ManifestIngressConfig, StreamingManifestDataset, cache_manifest_text_tokens, collate_manifest_items
from embedding_cache import (
    EMBEDDING_CACHE_INDEX_VERSION,
    EMBEDDING_CACHE_PROTOCOL_VERSION,
    EMBEDDING_CACHE_STORAGE_PER_SAMPLE,
    save_embedding_cache_index,
    save_embedding_cache_meta,
    save_embedding_payload,
)
from gpu_stream import GpuStreamConfig, GpuStreamPreprocessor
from input_cache import build_input_cache_contract, load_input_cache_meta, manifest_item_cache_key, validate_input_cache_contract
from mainline_utils import set_seed
from manifest_utils import load_split_manifest, manifest_sha256, select_manifest_items
from models import FusionClassifier
from runtime_adapt import detect_runtime, resolve_batch_size, resolve_worker_count, select_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build exploratory frozen-backbone mainline embedding cache")
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--subset", type=str, default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--input-cache", type=Path, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--batch-size", type=str, default="auto")
    p.add_argument("--num-workers", type=str, default="auto")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--max-audio-sec", type=float, default=6.0)
    p.add_argument("--text-model", type=str, default="xlm-roberta-large")
    p.add_argument("--max-text-len", type=int, default=128)
    p.add_argument("--audio-model", type=str, default="microsoft/wavlm-large")
    p.add_argument("--audio-model-revision", type=str, default="")
    p.add_argument("--video-model", type=str, default="MCG-NJU/videomae-large")
    p.add_argument("--num-frames", type=int, default=64)
    p.add_argument("--rgb-size", type=int, default=224)
    p.add_argument("--audio-backend", type=str, default="auto", choices=["auto", "torchaudio", "soundfile"])
    p.add_argument("--video-decode-backend", type=str, default="auto", choices=["auto", "decord", "cpu"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_tokenizer(model_name: str) -> Any:
    from hf_compat import ensure_transformers_torch_compat
    from hf_loading import resolve_hf_pretrained_source

    ensure_transformers_torch_compat()
    from transformers import AutoTokenizer

    source, load_kwargs = resolve_hf_pretrained_source(str(model_name))
    return AutoTokenizer.from_pretrained(source, **load_kwargs)


def main() -> None:
    args = parse_args()
    set_seed(0)
    split_manifest_path = args.split_manifest.expanduser()
    out_dir = args.output_dir.expanduser()
    if out_dir.exists():
        if not bool(args.overwrite) and any(out_dir.iterdir()):
            raise RuntimeError(f"Output embedding cache dir already exists and is not empty: {out_dir}")
        if bool(args.overwrite):
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_split_manifest(split_manifest_path)
    manifest_hash = manifest_sha256(split_manifest_path)
    dataset_kind = str(manifest.get("dataset_kind", "") or "")
    items = select_manifest_items(manifest, str(args.subset))
    if int(args.limit) > 0:
        items = items[: int(args.limit)]
    if not items:
        raise RuntimeError("No usable manifest items selected for embedding cache build.")

    profile = detect_runtime(str(args.device))
    device = select_device(str(args.device))
    input_cache_contract = None
    input_cache_dir = args.input_cache.expanduser() if args.input_cache is not None else None
    ingress = ManifestIngressConfig(
        sample_rate=int(args.sample_rate),
        max_audio_sec=float(args.max_audio_sec),
        audio_backend_mode=str(args.audio_backend),
        video_decode_backend=str(args.video_decode_backend),
        num_frames=int(args.num_frames),
        zero_audio=False,
        zero_video=False,
        video_backbone="videomae",
    )
    if input_cache_dir is not None:
        cache_meta = load_input_cache_meta(input_cache_dir)
        input_cache_contract = build_input_cache_contract(cache_meta)
        mismatches = validate_input_cache_contract(
            input_cache_contract,
            manifest_sha256=manifest_hash,
            dataset_kind=dataset_kind,
            sample_rate=int(args.sample_rate),
            max_audio_sec=float(args.max_audio_sec),
            num_frames=int(args.num_frames),
            rgb_size=int(args.rgb_size),
            text_model=str(args.text_model),
            max_text_len=int(args.max_text_len),
            need_audio=True,
            need_video=True,
            need_text=True,
            text_policy="full",
        )
        if mismatches:
            raise RuntimeError("Input cache contract mismatch: " + ", ".join(mismatches))
        ds = CachedManifestDataset(
            items,
            ingress=ingress,
            cache_dir=input_cache_dir,
            text_policy="full",
            runtime_profile=profile,
            keep_in_memory=False,
        )
    else:
        ds = StreamingManifestDataset(items, ingress=ingress)

    tokenizer = _load_tokenizer(str(args.text_model))
    if input_cache_contract is None:
        cache_manifest_text_tokens(ds.items, tokenizer, max_text_len=int(args.max_text_len), text_policy="full")
    batch_size = resolve_batch_size(
        args.batch_size,
        phase="inference",
        profile=profile,
        feature_cache=False,
        video_backbone="videomae",
        freeze_audio=True,
        freeze_text=True,
        freeze_flow=True,
        freeze_rgb=True,
    )
    num_workers = resolve_worker_count(
        args.num_workers,
        phase="inference",
        profile=profile,
        dataset_in_memory=bool(getattr(ds, "in_memory", False)),
        cache_backed=bool(input_cache_contract is not None),
        total_items=len(items),
    )
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_manifest_items,
    )
    preprocessor = GpuStreamPreprocessor(
        GpuStreamConfig(
            device=device,
            video_backbone="videomae",
            sample_rate=int(args.sample_rate),
            max_audio_sec=float(args.max_audio_sec),
            audio_backend_mode=str(args.audio_backend),
            num_frames=int(args.num_frames),
            rgb_size=int(args.rgb_size),
            zero_video=False,
            zero_audio=False,
            zero_text=False,
            video_decode_backend=str(args.video_decode_backend),
        )
    )
    model = FusionClassifier(
        freeze_audio=True,
        freeze_video=True,
        freeze_flow=True,
        freeze_rgb=True,
        freeze_prosody=True,
        text_model=str(args.text_model),
        freeze_text=True,
        audio_model=str(args.audio_model),
        audio_model_revision=(str(args.audio_model_revision).strip() or None),
        video_backbone="videomae",
        video_model=str(args.video_model),
        fusion_mode="concat",
        intensity_head=False,
    ).to(device)
    model.eval()

    meta = {
        "protocol_version": EMBEDDING_CACHE_PROTOCOL_VERSION,
        "manifest_sha256": manifest_hash,
        "dataset_kind": dataset_kind,
        "text_model": str(args.text_model),
        "audio_model": str(args.audio_model),
        "audio_model_revision": str(args.audio_model_revision).strip(),
        "video_model": str(args.video_model),
        "max_text_len": int(args.max_text_len),
        "sample_rate": int(args.sample_rate),
        "max_audio_sec": float(args.max_audio_sec),
        "num_frames": int(args.num_frames),
        "rgb_size": int(args.rgb_size),
        "text_pooling": "cls",
        "audio_pooling": "mean_std_masked_v1",
        "rgb_pooling": "videomae_pooler_or_cls_v1",
        "pooling_version": "mainline_embedding_pooling_v1",
        "embedding_dtype": "float32",
        "text_dim": int(model.text.out_dim),
        "audio_dim": int(model.audio.out_dim),
        "rgb_dim": int(model.video.out_dim),
        "has_text_emb": True,
        "has_audio_emb": True,
        "has_rgb_emb": True,
        "freeze_text_required": True,
        "freeze_audio_required": True,
        "freeze_rgb_required": True,
        "audio_aug_allowed": False,
        "storage_format": EMBEDDING_CACHE_STORAGE_PER_SAMPLE,
        "index_version": EMBEDDING_CACHE_INDEX_VERSION,
        "source_input_cache_contract": input_cache_contract,
    }
    save_embedding_cache_meta(out_dir, meta)

    entries: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_items in tqdm(loader, desc="Build embedding cache", unit="batch"):
            batch = preprocessor.prepare_batch(
                batch_items,
                tokenizer=tokenizer,
                text_policy="full",
                max_text_len=int(args.max_text_len),
            )
            text_emb = model._encode_text(
                batch.get("text_inputs", None),
                batch_size=int(batch["labels"].shape[0]),
                device=device,
                dtype=torch.float32,
            ).detach().cpu().to(torch.float32)
            audio_emb = model._encode_audio(batch["audio"], batch.get("audio_lens", None)).detach().cpu().to(torch.float32)
            _, rgb_emb = model._encode_video_parts(None, batch.get("rgb", None))
            if rgb_emb is None:
                raise RuntimeError("rgb_embedding_missing")
            rgb_emb = rgb_emb.detach().cpu().to(torch.float32)
            for idx, item in enumerate(batch_items):
                key = manifest_item_cache_key(item)
                entries.append(
                    save_embedding_payload(
                        out_dir,
                        key,
                        {
                            "text_emb": text_emb[idx].contiguous(),
                            "audio_emb": audio_emb[idx].contiguous(),
                            "rgb_emb": rgb_emb[idx].contiguous(),
                        },
                    )
                )
    save_embedding_cache_index(out_dir, entries)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "items": len(entries),
                "contract": meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
