from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _ensure_project_root_on_path() -> None:
    """在脚本直跑时把仓库根目录加入导入路径。"""

    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()

from data import CachedMotionAudioDataset, collate
from manifest_utils import EMOTIONS
from models import FusionClassifier
from runtime_adapt import (
    detect_runtime,
    resolve_amp_mode,
    resolve_batch_size,
    resolve_prefetch_factor,
    resolve_worker_count,
    select_device,
)
from text_policy_utils import resolve_text_policy, select_text_for_policy


def _atomic_torch_save(obj: Any, dst: Path) -> None:
    """原子写 shard，避免半截 `.pt` 文件。"""

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd: int | None = None
    tmp_path: Path | None = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
        tmp_path = Path(tmp_name)
        with os.fdopen(tmp_fd, "wb") as f:
            tmp_fd = None
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(dst))
        tmp_path = None
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except Exception:
                pass
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    """解析 feature cache 构建命令行参数。"""

    p = argparse.ArgumentParser(description="Build GPU feature cache from raw predecoded shards")
    p.add_argument("--cached-dataset", type=Path, required=True, help="Raw cached shard dir/file produced by predecode_dataset.py")
    p.add_argument("--output", type=Path, required=True, help="Output directory for feature cache shards")
    p.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint to load encoder weights/config from")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--amp-mode", type=str, default="auto", choices=["auto", "off", "fp16", "bf16"])
    p.add_argument("--batch-size", type=str, default="auto")
    p.add_argument("--num-workers", type=str, default="auto")
    p.add_argument("--prefetch-factor", type=str, default="auto")
    p.add_argument("--shard-size", type=int, default=100)
    p.add_argument("--retain-raw", action="store_true", help="Keep raw tensors even when an embedding cache is produced")
    p.add_argument("--video-backbone", type=str, default="dual", choices=["flow", "videomae", "dual"])
    p.add_argument("--video-model", type=str, default="MCG-NJU/videomae-large")
    p.add_argument("--audio-model", type=str, default="microsoft/wavlm-large")
    p.add_argument(
        "--audio-model-revision",
        type=str,
        default="",
        help="Optional HuggingFace revision / commit hash for the audio model. Use this when the default branch lacks model.safetensors.",
    )
    p.add_argument("--text-model", type=str, default="xlm-roberta-large")
    p.add_argument("--max-text-len", type=int, default=128)
    p.add_argument(
        "--text-policy",
        type=str,
        default="full",
        choices=["full", "mask_emotion_cues", "drop"],
        help="How text should be represented before caching text embeddings.",
    )
    p.add_argument("--freeze-audio", action="store_true", help="Cache audio encoder output to audio_emb")
    p.add_argument("--freeze-flow", action="store_true", help="Cache flow encoder output to flow_emb")
    p.add_argument("--freeze-rgb", action="store_true", help="Cache RGB VideoMAE output to rgb_emb")
    p.add_argument("--freeze-text", action="store_true", help="Cache text encoder output to text_emb")
    p.add_argument("--print-every", type=int, default=20)
    return p.parse_args()


def _read_checkpoint_args(ckpt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """读取 checkpoint 参数与 state dict。"""

    ckpt = torch.load(ckpt_path.expanduser(), map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}
    return ckpt_args, state


def _resolve_model_args(args: argparse.Namespace) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """合并 CLI 与 checkpoint 中的模型配置。"""

    ckpt_args: dict[str, Any] = {}
    state = None
    if args.checkpoint is not None:
        ckpt_args, state = _read_checkpoint_args(args.checkpoint)

    def _pick(name: str, default: Any) -> Any:
        cli_value = getattr(args, name)
        if isinstance(cli_value, str) and cli_value.strip():
            if args.checkpoint is None or cli_value.strip() != str(default):
                return cli_value
        if cli_value is not None and not isinstance(cli_value, str):
            if args.checkpoint is None or cli_value != default:
                return cli_value
        ckpt_value = ckpt_args.get(name, default)
        return default if ckpt_value is None else ckpt_value

    model_args = {
        "video_backbone": str(_pick("video_backbone", "dual")),
        "video_model": str(_pick("video_model", "MCG-NJU/videomae-large")),
        "audio_model": str(_pick("audio_model", "microsoft/wavlm-large")),
        "audio_model_revision": str(_pick("audio_model_revision", "")),
        "text_model": str(_pick("text_model", "xlm-roberta-large")),
        "max_text_len": int(_pick("max_text_len", 128)),
    }

    if args.checkpoint is not None:
        if not bool(args.freeze_audio):
            args.freeze_audio = bool(ckpt_args.get("freeze_audio", False))
        if not bool(args.freeze_text):
            args.freeze_text = bool(ckpt_args.get("freeze_text", False))
        if not bool(args.freeze_flow):
            args.freeze_flow = bool(ckpt_args.get("freeze_video", False) or ckpt_args.get("freeze_flow", False))
        if not bool(args.freeze_rgb):
            args.freeze_rgb = bool(ckpt_args.get("freeze_video", False) or ckpt_args.get("freeze_rgb", False))

    return model_args, state


def _amp_context(device: torch.device, amp_mode: str) -> contextlib.AbstractContextManager[Any]:
    """构造 AMP 上下文。"""

    if device.type != "cuda" or amp_mode == "off":
        return contextlib.nullcontext()
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        return amp_mod.autocast("cuda", enabled=True, dtype=torch.bfloat16 if amp_mode == "bf16" else torch.float16)  # type: ignore[attr-defined]
    return torch.cuda.amp.autocast(enabled=True)


def _slice_audio(batch: dict[str, Any], idx: int) -> Optional[torch.Tensor]:
    """从 padding 后 batch 中裁回单条音频。"""

    if "audio" not in batch:
        return None
    lens = batch.get("audio_lens", None)
    wav = batch["audio"][idx]
    if lens is None:
        return wav.clone()
    end = int(lens[idx].item())
    return wav[:end].clone()


def _resolve_text_inputs(batch: dict[str, Any], device: torch.device) -> Optional[dict[str, torch.Tensor]]:
    """把 collate 产出的预分词文本搬到 device。"""

    text_inputs = batch.get("text_inputs", None)
    if text_inputs is None:
        return None
    return {k: v.to(device, non_blocking=True) for k, v in text_inputs.items()}


def main() -> None:
    """构建 GPU feature cache。"""

    args = parse_args()
    profile = detect_runtime(args.device)
    device = select_device(args.device)
    amp_mode = resolve_amp_mode(args.amp_mode, profile)
    model_args, state = _resolve_model_args(args)

    ds = CachedMotionAudioDataset(args.cached_dataset.expanduser())
    dataset_in_memory = True
    batch_size = resolve_batch_size(
        args.batch_size,
        phase="feature_cache",
        profile=profile,
        feature_cache=True,
        video_backbone=str(model_args["video_backbone"]),
        freeze_audio=bool(args.freeze_audio),
        freeze_text=bool(args.freeze_text),
        freeze_flow=bool(args.freeze_flow),
        freeze_rgb=bool(args.freeze_rgb),
    )
    num_workers = resolve_worker_count(
        args.num_workers,
        phase="feature_cache",
        profile=profile,
        dataset_in_memory=dataset_in_memory,
        total_items=len(ds),
    )
    prefetch_factor = resolve_prefetch_factor(args.prefetch_factor, num_workers=num_workers)
    pin_memory = device.type == "cuda"
    dl_kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "collate_fn": collate,
        "pin_memory": pin_memory,
    }
    if int(num_workers) > 0:
        dl_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = int(prefetch_factor)
    loader = DataLoader(ds, **dl_kwargs)

    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required for build_feature_cache.py") from e

    tokenizer = None
    text_policy = resolve_text_policy(args.text_policy)
    effective_freeze_text = bool(args.freeze_text) and text_policy != "drop"
    if effective_freeze_text:
        tokenizer = AutoTokenizer.from_pretrained(str(model_args["text_model"]))

    model = FusionClassifier(
        num_classes=7,
        text_model=str(model_args["text_model"]),
        freeze_text=True,
        audio_model=str(model_args["audio_model"]),
        audio_model_revision=(str(model_args["audio_model_revision"]).strip() or None),
        freeze_audio=True,
        video_backbone=str(model_args["video_backbone"]),
        video_model=str(model_args["video_model"]),
        freeze_flow=True,
        freeze_rgb=True,
        freeze_prosody=True,
    ).to(device)
    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"WARNING: missing keys while loading checkpoint into feature cache builder: {missing[:6]}", flush=True)
        if unexpected:
            print(f"WARNING: unexpected keys while loading checkpoint into feature cache builder: {unexpected[:6]}", flush=True)
    model.eval()

    print(
        json.dumps(
            {
                "runtime": profile.to_jsonable(),
                "device": str(device),
                "amp_mode": amp_mode,
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "cached_dataset": str(args.cached_dataset.expanduser()),
                "output": str(args.output.expanduser()),
                "freeze_audio": bool(args.freeze_audio),
                "freeze_flow": bool(args.freeze_flow),
                "freeze_rgb": bool(args.freeze_rgb),
                "freeze_text": bool(effective_freeze_text),
                "text_policy": text_policy,
                "retain_raw": bool(args.retain_raw),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    out_dir = args.output.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cached: list[dict[str, Any]] = []
    shard_id = 1
    iterator = tqdm(loader, total=math.ceil(len(ds) / max(1, int(batch_size))), desc="Feature cache", unit="batch")
    with torch.inference_mode():
        for batch_idx, batch in enumerate(iterator, start=1):
            nb = pin_memory and device.type == "cuda"
            flow = batch.get("flow", None)
            rgb = batch.get("rgb", None)
            wav = batch.get("audio", None)
            audio_lens = batch.get("audio_lens", None)
            raw_audio_emb = batch.get("audio_emb", None)
            text_inputs = _resolve_text_inputs(batch, device) if batch.get("text_inputs", None) is not None else None

            if effective_freeze_text and text_inputs is None and tokenizer is not None:
                batch_texts = []
                global_labels = []
                for i in range(int(batch["labels"].shape[0])):
                    raw_label = str(batch.get("_global_label_en", [""] * int(batch["labels"].shape[0]))[i] if "_global_label_en" in batch else "")
                    if not raw_label:
                        try:
                            label_idx = int(batch["labels"][i].item())
                            raw_label = EMOTIONS[label_idx] if 0 <= label_idx < len(EMOTIONS) else ""
                        except Exception:
                            raw_label = ""
                    global_labels.append(raw_label)
                masked_texts = [str(x) for x in batch.get("masked_mn", [""] * int(batch["labels"].shape[0]))]
                full_texts = [str(x) for x in batch.get("mn", [""] * int(batch["labels"].shape[0]))]
                for i in range(int(batch["labels"].shape[0])):
                    batch_texts.append(
                        select_text_for_policy(
                            full_text=full_texts[i],
                            masked_text=masked_texts[i],
                            label_en=global_labels[i] or None,
                            policy=text_policy,
                        )
                    )
                text_inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=int(model_args["max_text_len"]),
                    return_tensors="pt",
                )
                text_inputs = {k: v.to(device, non_blocking=nb) for k, v in text_inputs.items()}

            if flow is not None:
                flow = flow.to(device, non_blocking=nb)
            if rgb is not None:
                rgb = rgb.to(device, non_blocking=nb)
            if wav is not None:
                wav = wav.to(device, non_blocking=nb)
            if audio_lens is not None:
                audio_lens = audio_lens.to(device, non_blocking=nb)
            if raw_audio_emb is not None:
                raw_audio_emb = raw_audio_emb.to(device, non_blocking=nb)

            batch_audio_emb = raw_audio_emb
            batch_flow_emb = batch.get("flow_emb", None)
            batch_rgb_emb = batch.get("rgb_emb", None)
            batch_text_emb = batch.get("text_emb", None)
            if batch_flow_emb is not None:
                batch_flow_emb = batch_flow_emb.to(device, non_blocking=nb)
            if batch_rgb_emb is not None:
                batch_rgb_emb = batch_rgb_emb.to(device, non_blocking=nb)
            if batch_text_emb is not None:
                batch_text_emb = batch_text_emb.to(device, non_blocking=nb)

            with _amp_context(device, amp_mode):
                if bool(args.freeze_flow) and batch_flow_emb is None:
                    if str(model_args["video_backbone"]) == "dual":
                        if flow is None or model.video_flow is None:
                            raise RuntimeError("Cannot build flow_emb without raw flow tensors")
                        batch_flow_emb = model.video_flow(flow)
                    elif str(model_args["video_backbone"]) == "flow":
                        if flow is None or model.video is None:
                            raise RuntimeError("Cannot build flow_emb without raw flow tensors")
                        batch_flow_emb = model.video(flow)
                if bool(args.freeze_rgb) and batch_rgb_emb is None:
                    if str(model_args["video_backbone"]) == "dual":
                        if rgb is None or model.video_rgb is None:
                            raise RuntimeError("Cannot build rgb_emb without raw rgb tensors")
                        batch_rgb_emb = model.video_rgb(rgb)
                    elif str(model_args["video_backbone"]) == "videomae":
                        if rgb is None or model.video is None:
                            raise RuntimeError("Cannot build rgb_emb without raw rgb tensors")
                        batch_rgb_emb = model.video(rgb)
                if bool(args.freeze_audio) and batch_audio_emb is None:
                    if wav is None:
                        raise RuntimeError("Cannot build audio_emb without raw audio tensors")
                    if audio_lens is not None:
                        try:
                            batch_audio_emb = model.audio(wav, lengths=audio_lens)
                        except TypeError:
                            batch_audio_emb = model.audio(wav)
                    else:
                        batch_audio_emb = model.audio(wav)
                if effective_freeze_text and batch_text_emb is None:
                    if text_inputs is None:
                        raise RuntimeError("Cannot build text_emb without text inputs")
                    batch_text_emb = model.text(text_inputs)

            batch_size_actual = int(batch["labels"].shape[0])
            for i in range(batch_size_actual):
                sample: dict[str, Any] = {
                    "prosody": batch["prosody"][i].detach().cpu().to(torch.float32),
                    "label": batch["labels"][i].detach().cpu().to(torch.long),
                    "stem": str(batch["stems"][i]),
                    "mn": str(batch["mn"][i]),
                    "masked_mn": str(batch.get("masked_mn", [""] * batch_size_actual)[i]),
                    "speaker_id": str(batch["speaker_id"][i]),
                    "text_cue_flag": bool(batch["text_cue_flag"][i].item()),
                    "cue_severity": str(batch.get("cue_severity", ["none"] * batch_size_actual)[i]),
                    "prompt_group_id": str(batch.get("prompt_group_id", [""] * batch_size_actual)[i]),
                    "intensity": batch["intensity"][i].detach().cpu().to(torch.float32),
                }
                if bool(args.freeze_flow):
                    if batch_flow_emb is None:
                        raise RuntimeError("Requested flow feature cache but flow_emb is missing")
                    sample["flow_emb"] = batch_flow_emb[i].detach().cpu().to(torch.float16)
                    if bool(args.retain_raw) and "flow" in batch:
                        sample["flow"] = batch["flow"][i].detach().cpu().to(torch.float16)
                elif "flow" in batch:
                    sample["flow"] = batch["flow"][i].detach().cpu().to(torch.float16)

                if bool(args.freeze_rgb):
                    if batch_rgb_emb is None:
                        raise RuntimeError("Requested rgb feature cache but rgb_emb is missing")
                    sample["rgb_emb"] = batch_rgb_emb[i].detach().cpu().to(torch.float16)
                    if bool(args.retain_raw) and "rgb" in batch:
                        sample["rgb"] = batch["rgb"][i].detach().cpu().to(torch.float16)
                elif "rgb" in batch:
                    sample["rgb"] = batch["rgb"][i].detach().cpu().to(torch.float16)

                if bool(args.freeze_audio):
                    if batch_audio_emb is None:
                        raise RuntimeError("Requested audio feature cache but audio_emb is missing")
                    sample["audio_emb"] = batch_audio_emb[i].detach().cpu().to(torch.float16)
                    if bool(args.retain_raw):
                        wav_i = _slice_audio(batch, i)
                        if wav_i is not None:
                            sample["audio"] = wav_i.detach().cpu().to(torch.float32)
                else:
                    wav_i = _slice_audio(batch, i)
                    if wav_i is not None:
                        sample["audio"] = wav_i.detach().cpu().to(torch.float32)
                    elif "audio_emb" in batch:
                        sample["audio_emb"] = batch["audio_emb"][i].detach().cpu().to(torch.float16)

                if effective_freeze_text:
                    if batch_text_emb is None:
                        raise RuntimeError("Requested text feature cache but text_emb is missing")
                    sample["text_emb"] = batch_text_emb[i].detach().cpu().to(torch.float16)

                cached.append(sample)
                if len(cached) >= int(args.shard_size):
                    shard_path = out_dir / f"{shard_id}.pt"
                    _atomic_torch_save(
                        {
                            "samples": cached,
                            "config": {
                                "source_cached_dataset": str(args.cached_dataset.expanduser()),
                                "source_cache_config": ds.config,
                                "runtime": profile.to_jsonable(),
                                "feature_cache": {
                                    "checkpoint": str(args.checkpoint.expanduser()) if args.checkpoint is not None else None,
                                    "device": str(device),
                                    "amp_mode": amp_mode,
                                    "video_backbone": str(model_args["video_backbone"]),
                                    "video_model": str(model_args["video_model"]),
                                    "audio_model": str(model_args["audio_model"]),
                                    "audio_model_revision": str(model_args["audio_model_revision"]),
                                    "text_model": str(model_args["text_model"]),
                                    "max_text_len": int(model_args["max_text_len"]),
                                    "freeze_audio": bool(args.freeze_audio),
                                    "freeze_flow": bool(args.freeze_flow),
                                    "freeze_rgb": bool(args.freeze_rgb),
                                    "freeze_text": bool(effective_freeze_text),
                                    "text_policy": text_policy,
                                    "retain_raw": bool(args.retain_raw),
                                },
                            },
                        },
                        shard_path,
                    )
                    cached = []
                    shard_id += 1

            if args.print_every > 0 and (batch_idx % int(args.print_every)) == 0:
                iterator.set_postfix_str(f"shard={shard_id} cached={((shard_id - 1) * int(args.shard_size)) + len(cached)}")

    if cached:
        shard_path = out_dir / f"{shard_id}.pt"
        _atomic_torch_save(
            {
                "samples": cached,
                "config": {
                    "source_cached_dataset": str(args.cached_dataset.expanduser()),
                    "source_cache_config": ds.config,
                    "runtime": profile.to_jsonable(),
                    "feature_cache": {
                        "checkpoint": str(args.checkpoint.expanduser()) if args.checkpoint is not None else None,
                        "device": str(device),
                        "amp_mode": amp_mode,
                        "video_backbone": str(model_args["video_backbone"]),
                        "video_model": str(model_args["video_model"]),
                        "audio_model": str(model_args["audio_model"]),
                        "audio_model_revision": str(model_args["audio_model_revision"]),
                        "text_model": str(model_args["text_model"]),
                        "max_text_len": int(model_args["max_text_len"]),
                        "freeze_audio": bool(args.freeze_audio),
                        "freeze_flow": bool(args.freeze_flow),
                        "freeze_rgb": bool(args.freeze_rgb),
                        "freeze_text": bool(effective_freeze_text),
                        "text_policy": text_policy,
                        "retain_raw": bool(args.retain_raw),
                    },
                },
            },
            shard_path,
        )

    print(f"Feature cache ready -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
