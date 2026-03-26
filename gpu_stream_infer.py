from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

torch: Any = None
DataLoader: Any = None
tqdm: Any = None
ManifestIngressConfig: Any = None
StreamingManifestDataset: Any = None
cache_manifest_text_tokens: Any = None
collate_manifest_items: Any = None
GpuStreamConfig: Any = None
GpuStreamPreprocessor: Any = None
autocast_context: Any = None
ccc: Any = None
prepare_manifest_items_for_task: Any = None
resolve_ablation_flags: Any = None
EMOTIONS: Any = None
build_validity_summary: Any = None
filter_manifest_items_for_task: Any = None
load_split_manifest: Any = None
manifest_sha256: Any = None
map_label_to_task_index: Any = None
resolve_task_label_names: Any = None
resolve_task_mode: Any = None
select_manifest_items: Any = None
classification_summary: Any = None
speaker_majority_baseline: Any = None
speaker_only_baseline: Any = None
FusionClassifier: Any = None
detect_runtime: Any = None
resolve_amp_mode: Any = None
resolve_batch_size: Any = None
resolve_prefetch_factor: Any = None
resolve_worker_count: Any = None
select_device: Any = None
resolve_text_policy: Any = None


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()


def _lazy_runtime_imports() -> None:
    global torch, DataLoader, tqdm
    global ManifestIngressConfig, StreamingManifestDataset, cache_manifest_text_tokens, collate_manifest_items
    global GpuStreamConfig, GpuStreamPreprocessor
    global autocast_context, ccc, prepare_manifest_items_for_task, resolve_ablation_flags
    global EMOTIONS, build_validity_summary, filter_manifest_items_for_task, load_split_manifest
    global manifest_sha256, map_label_to_task_index, resolve_task_label_names, resolve_task_mode, select_manifest_items
    global classification_summary, speaker_majority_baseline, speaker_only_baseline
    global FusionClassifier
    global detect_runtime, resolve_amp_mode, resolve_batch_size, resolve_prefetch_factor, resolve_worker_count, select_device
    global resolve_text_policy
    if torch is not None:
        return

    import torch as _torch
    from torch.utils.data import DataLoader as _DataLoader
    from tqdm import tqdm as _tqdm

    from data import (
        ManifestIngressConfig as _ManifestIngressConfig,
        StreamingManifestDataset as _StreamingManifestDataset,
        cache_manifest_text_tokens as _cache_manifest_text_tokens,
        collate_manifest_items as _collate_manifest_items,
    )
    from gpu_stream import GpuStreamConfig as _GpuStreamConfig, GpuStreamPreprocessor as _GpuStreamPreprocessor
    from mainline_utils import (
        autocast_context as _autocast_context,
        ccc as _ccc,
        prepare_manifest_items_for_task as _prepare_manifest_items_for_task,
        resolve_ablation_flags as _resolve_ablation_flags,
    )
    from manifest_utils import (
        EMOTIONS as _EMOTIONS,
        build_validity_summary as _build_validity_summary,
        filter_manifest_items_for_task as _filter_manifest_items_for_task,
        load_split_manifest as _load_split_manifest,
        manifest_sha256 as _manifest_sha256,
        map_label_to_task_index as _map_label_to_task_index,
        resolve_task_label_names as _resolve_task_label_names,
        resolve_task_mode as _resolve_task_mode,
        select_manifest_items as _select_manifest_items,
    )
    from metrics_utils import (
        classification_summary as _classification_summary,
        speaker_majority_baseline as _speaker_majority_baseline,
        speaker_only_baseline as _speaker_only_baseline,
    )
    from models import FusionClassifier as _FusionClassifier
    from runtime_adapt import (
        detect_runtime as _detect_runtime,
        resolve_amp_mode as _resolve_amp_mode,
        resolve_batch_size as _resolve_batch_size,
        resolve_prefetch_factor as _resolve_prefetch_factor,
        resolve_worker_count as _resolve_worker_count,
        select_device as _select_device,
    )
    from text_policy_utils import resolve_text_policy as _resolve_text_policy

    torch = _torch
    DataLoader = _DataLoader
    tqdm = _tqdm
    ManifestIngressConfig = _ManifestIngressConfig
    StreamingManifestDataset = _StreamingManifestDataset
    cache_manifest_text_tokens = _cache_manifest_text_tokens
    collate_manifest_items = _collate_manifest_items
    GpuStreamConfig = _GpuStreamConfig
    GpuStreamPreprocessor = _GpuStreamPreprocessor
    autocast_context = _autocast_context
    ccc = _ccc
    prepare_manifest_items_for_task = _prepare_manifest_items_for_task
    resolve_ablation_flags = _resolve_ablation_flags
    EMOTIONS = _EMOTIONS
    build_validity_summary = _build_validity_summary
    filter_manifest_items_for_task = _filter_manifest_items_for_task
    load_split_manifest = _load_split_manifest
    manifest_sha256 = _manifest_sha256
    map_label_to_task_index = _map_label_to_task_index
    resolve_task_label_names = _resolve_task_label_names
    resolve_task_mode = _resolve_task_mode
    select_manifest_items = _select_manifest_items
    classification_summary = _classification_summary
    speaker_majority_baseline = _speaker_majority_baseline
    speaker_only_baseline = _speaker_only_baseline
    FusionClassifier = _FusionClassifier
    detect_runtime = _detect_runtime
    resolve_amp_mode = _resolve_amp_mode
    resolve_batch_size = _resolve_batch_size
    resolve_prefetch_factor = _resolve_prefetch_factor
    resolve_worker_count = _resolve_worker_count
    select_device = _select_device
    resolve_text_policy = _resolve_text_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch inference for the manifest-driven gpu_stream mainline")
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--subset", type=str, default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--task-mode", type=str, default="")
    p.add_argument("--speaker-id", type=str, default="")
    p.add_argument("--benchmark-tag", type=str, default="default")
    p.add_argument("--text-policy", type=str, default="")
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/motion_prosody/checkpoints/best.pt"))
    p.add_argument("--output", type=Path, default=Path("outputs/motion_prosody/inference_results.jsonl"))
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--batch-size", type=str, default="auto")
    p.add_argument("--num-workers", type=str, default="auto")
    p.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "text-only", "audio-only", "video-only", "no-text", "no-audio", "no-video"],
    )
    p.add_argument("--num-frames", type=int, default=64)
    p.add_argument("--flow-size", type=int, default=112, help=argparse.SUPPRESS)
    p.add_argument("--rgb-size", type=int, default=224, help=argparse.SUPPRESS)
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--max-audio-sec", type=float, default=6.0)
    p.add_argument("--audio-backend", type=str, default="auto", choices=["auto", "torchaudio", "soundfile"], help=argparse.SUPPRESS)
    p.add_argument("--video-decode-backend", type=str, default="auto", choices=["auto", "decord", "cpu"], help=argparse.SUPPRESS)
    p.add_argument("--flow-backend", type=str, default="torch_motion", choices=["torch_motion", "legacy"], help=argparse.SUPPRESS)
    p.add_argument("--prosody-no-pitch", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--amp-mode", type=str, default="auto", choices=["auto", "off", "fp16", "bf16"])
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--print-json", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--print-result", type=str, default="none", choices=["none", "ok", "error", "all"], help=argparse.SUPPRESS)
    p.add_argument("--print-per-sample", action="store_true")
    p.add_argument("--print-text", action="store_true")
    p.add_argument("--text-max-len", type=int, default=0)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--text-model", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--audio-model", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--audio-model-revision", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--video-model", type=str, default="", help=argparse.SUPPRESS)

    # Hidden compatibility args.
    p.add_argument("--zero-video", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--zero-audio", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--zero-text", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--skip-video-encoder", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-amp", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cached-dataset", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--feature-cache", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--pipeline", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--cache-mode", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--ram-cache-size", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def _validate_compat_args(args: argparse.Namespace) -> None:
    if args.cached_dataset is not None or args.feature_cache is not None:
        raise RuntimeError(
            "Mainline inference no longer supports --cached-dataset or --feature-cache. "
            "Use --split-manifest with the gpu_stream path."
        )
    pipeline = str(args.pipeline or "").strip().lower()
    if pipeline and pipeline not in {"auto", "gpu_stream"}:
        raise RuntimeError(f"Unsupported retired --pipeline value: {args.pipeline}")
    cache_mode = str(args.cache_mode or "").strip().lower()
    if cache_mode and cache_mode not in {"none"}:
        raise RuntimeError("--cache-mode is retired; gpu_stream-only inference accepts at most --cache-mode none.")
    if args.ram_cache_size not in {None, 0}:
        raise RuntimeError("--ram-cache-size is retired in gpu_stream-only inference.")
    if bool(args.no_amp):
        args.amp_mode = "off"
    if bool(args.skip_video_encoder):
        args.zero_video = True
    if bool(args.print_per_sample):
        args.print_result = "all"
        args.no_progress = True


def _load_tokenizer(model_name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required for text-enabled inference.") from e
    return AutoTokenizer.from_pretrained(str(model_name))


def _load_model(
    ckpt_path: Path,
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[FusionClassifier, bool, list[str], str, str | None, str, dict[str, Any] | None]:
    _lazy_runtime_imports()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    label_names = [str(x) for x in ckpt.get("label_names", [])] if isinstance(ckpt, dict) and isinstance(ckpt.get("label_names"), list) else list(EMOTIONS)
    validity = ckpt.get("validity", None) if isinstance(ckpt, dict) and isinstance(ckpt.get("validity"), dict) else None

    text_model = str(ckpt_args.get("text_model", "") or "xlm-roberta-large")
    audio_model = str(ckpt_args.get("audio_model", "") or "microsoft/wavlm-large")
    audio_model_revision = str(ckpt_args.get("audio_model_revision", "") or "")
    video_model = str(ckpt_args.get("video_model", "") or "MCG-NJU/videomae-large")
    video_backbone = str(ckpt_args.get("video_backbone", "") or "dual")
    fusion_mode = str(ckpt_args.get("fusion_mode", "") or "gated_text")
    gate_temperature = float(ckpt_args.get("gate_temperature", 1.0) or 1.0)
    gate_scale = float(ckpt_args.get("gate_scale", 1.0) or 1.0)
    delta_scale = float(ckpt_args.get("delta_scale", 1.0) or 1.0)
    modality_dropout = float(ckpt_args.get("modality_dropout", 0.0) or 0.0)
    use_intensity = bool(ckpt_args.get("use_intensity", False))
    task_mode = str(ckpt_args.get("task_mode", "") or "confounded_7way")
    task_speaker_id = str(ckpt_args.get("speaker_id", "") or "").strip().upper() or None
    text_policy = str(ckpt_args.get("text_policy", "") or "full")

    if str(args.text_model).strip():
        text_model = str(args.text_model).strip()
    if str(args.audio_model).strip():
        audio_model = str(args.audio_model).strip()
    if str(args.audio_model_revision).strip():
        audio_model_revision = str(args.audio_model_revision).strip()
    if str(args.video_model).strip():
        video_model = str(args.video_model).strip()
    if str(args.task_mode).strip():
        task_mode = str(args.task_mode).strip()
    if str(args.speaker_id).strip():
        task_speaker_id = str(args.speaker_id).strip().upper()
    if str(args.text_policy).strip():
        text_policy = str(args.text_policy).strip()

    task_mode = resolve_task_mode(task_mode)
    text_policy = resolve_text_policy(text_policy)
    if label_names == list(EMOTIONS):
        label_names = resolve_task_label_names(task_mode, task_speaker_id)
    if validity is None or str(validity.get("task_mode", "")).strip().lower() != task_mode:
        validity = build_validity_summary({}, task_mode, task_speaker_id)

    model = FusionClassifier(
        num_classes=len(label_names),
        freeze_audio=True,
        freeze_video=True,
        freeze_flow=True,
        freeze_rgb=True,
        freeze_prosody=True,
        text_model=text_model,
        freeze_text=True,
        audio_model=audio_model,
        audio_model_revision=(audio_model_revision or None),
        video_backbone=video_backbone,
        video_model=video_model,
        fusion_mode=fusion_mode,
        gate_temperature=gate_temperature,
        gate_scale=gate_scale,
        delta_scale=delta_scale,
        modality_dropout=modality_dropout,
        intensity_head=bool(use_intensity),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING: missing checkpoint keys: {missing[:6]}" + (" ..." if len(missing) > 6 else ""), flush=True)
    if unexpected:
        print(f"WARNING: unexpected checkpoint keys: {unexpected[:6]}" + (" ..." if len(unexpected) > 6 else ""), flush=True)
    model.to(device)
    model.eval()
    setattr(model, "_label_names", list(label_names))
    return model, bool(use_intensity), label_names, task_mode, task_speaker_id, text_policy, validity


def _truncate_text(text: str, max_len: int) -> str:
    if int(max_len) <= 0 or len(text) <= int(max_len):
        return text
    return text[: int(max_len)] + "..."


def _format_error_record(item: dict[str, Any], err: Exception, *, text_max_len: int) -> dict[str, Any]:
    return {
        "stem": str(item.get("stem", item.get("seq", ""))),
        "status": "error",
        "error": f"{type(err).__name__}: {err}",
        "speaker_id": str(item.get("speaker_id", "UNKNOWN")),
        "label": str(item.get("label_en", "")),
        "text": _truncate_text(str(item.get("mn", item.get("text", ""))), text_max_len),
    }


def _infer_batch(
    *,
    items: list[dict[str, Any]],
    preprocessor: GpuStreamPreprocessor,
    tokenizer: Any | None,
    text_policy: str,
    max_text_len: int,
    model: FusionClassifier,
    use_intensity: bool,
    amp_mode: str,
) -> list[dict[str, Any]]:
    _lazy_runtime_imports()
    batch = preprocessor.prepare_batch(items, tokenizer=tokenizer, text_policy=text_policy, max_text_len=max_text_len)
    labels = batch["labels"]
    device = next(model.parameters()).device
    with torch.no_grad(), autocast_context(device, amp_mode):
        if model.video_backbone == "videomae":
            flow = None
        else:
            flow = batch.get("flow", None)
        logits_or_pair = model(
            flow,
            batch["audio"],
            batch["prosody"],
            text_inputs=batch.get("text_inputs", None),
            return_intensity=bool(use_intensity),
            rgb=batch.get("rgb", None),
            audio_lens=batch.get("audio_lens", None),
        )
        if bool(use_intensity):
            logits, pred_intensity = logits_or_pair
        else:
            logits = logits_or_pair
            pred_intensity = None
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
    pred_probs = probs.gather(1, pred.view(-1, 1)).squeeze(1).detach().cpu().tolist()
    labels_cpu = labels.detach().cpu().tolist()
    pred_cpu = pred.detach().cpu().tolist()
    intensity = batch["intensity"].detach().cpu().to(torch.float32)
    intensity_mask = batch["intensity_mask"].detach().cpu()
    pred_intensity_cpu = pred_intensity.detach().cpu().to(torch.float32) if pred_intensity is not None else None
    label_names = list(getattr(model, "_label_names", []))

    records: list[dict[str, Any]] = []
    for idx, stem in enumerate(batch["stems"]):
        rec = {
            "stem": str(stem),
            "status": "ok",
            "label": label_names[int(labels_cpu[idx])],
            "pred": label_names[int(pred_cpu[idx])],
            "match": bool(int(labels_cpu[idx]) == int(pred_cpu[idx])),
            "probability": float(pred_probs[idx]),
            "speaker_id": str(batch["speaker_id"][idx]),
            "text": str(batch["mn"][idx]),
        }
        if bool(use_intensity):
            if bool(intensity_mask[idx].item()):
                rec["intensity_gt"] = float(intensity[idx].item())
                rec["pred_intensity"] = float(pred_intensity_cpu[idx].item()) if pred_intensity_cpu is not None else None
            else:
                rec["intensity_gt"] = None
                rec["pred_intensity"] = None
        records.append(rec)
    return records


def main() -> None:
    args = parse_args()
    _validate_compat_args(args)
    _lazy_runtime_imports()

    zero_video, zero_audio, zero_text = resolve_ablation_flags(
        ablation=str(args.ablation),
        zero_video=bool(args.zero_video),
        zero_audio=bool(args.zero_audio),
        zero_text=bool(args.zero_text),
    )
    args.zero_video = bool(zero_video)
    args.zero_audio = bool(zero_audio)
    args.zero_text = bool(zero_text)

    split_manifest_path = args.split_manifest.expanduser()
    ckpt_path = args.checkpoint.expanduser()
    out_path = args.output.expanduser()
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    profile = detect_runtime(str(args.device))
    device = select_device(str(args.device))
    resolved_amp_mode = resolve_amp_mode(str(args.amp_mode), profile)
    print(f"Using device: {device}", flush=True)

    model, use_intensity, label_names, task_mode, task_speaker_id, text_policy, validity = _load_model(
        ckpt_path,
        device=device,
        args=args,
    )
    if str(text_policy) == "drop":
        args.zero_text = True

    manifest = load_split_manifest(split_manifest_path)
    manifest_hash = manifest_sha256(split_manifest_path)
    manifest_summary = manifest.get("summary", {})
    dataset_kind = str(manifest.get("dataset_kind", "") or "")
    if validity is None:
        validity = build_validity_summary(manifest_summary, task_mode, task_speaker_id)

    items = prepare_manifest_items_for_task(
        filter_manifest_items_for_task(select_manifest_items(manifest, args.subset), task_mode, task_speaker_id),
        task_mode=task_mode,
        speaker_id=task_speaker_id,
        map_label_to_task_index=map_label_to_task_index,
    )
    if int(args.limit) > 0:
        items = items[: int(args.limit)]
    if not items:
        raise RuntimeError("No usable manifest items remain after task/subset filtering.")

    ingress_cfg = ManifestIngressConfig(
        sample_rate=int(args.sample_rate),
        max_audio_sec=float(args.max_audio_sec),
        audio_backend_mode=str(args.audio_backend),
        video_decode_backend=str(args.video_decode_backend),
        num_frames=int(args.num_frames),
        zero_audio=bool(args.zero_audio),
        zero_video=bool(args.zero_video),
        video_backbone=str(model.video_backbone),
    )
    ds = StreamingManifestDataset(items, ingress=ingress_cfg)
    tokenizer = None
    if not bool(args.zero_text):
        tokenizer = _load_tokenizer(str(args.text_model).strip() or str(model.text.model_name))
        cache_manifest_text_tokens(ds.items, tokenizer, max_text_len=128, text_policy=str(text_policy))

    resolved_batch_size = resolve_batch_size(
        args.batch_size,
        phase="inference",
        profile=profile,
        feature_cache=False,
        video_backbone=str(model.video_backbone),
        freeze_audio=True,
        freeze_text=True,
        freeze_flow=True,
        freeze_rgb=True,
    )
    resolved_num_workers = resolve_worker_count(
        args.num_workers,
        phase="inference",
        profile=profile,
        dataset_in_memory=False,
        total_items=len(items),
    )
    prefetch_factor = resolve_prefetch_factor("auto", num_workers=int(resolved_num_workers))
    dl_kwargs: dict[str, Any] = {
        "num_workers": int(resolved_num_workers),
        "collate_fn": collate_manifest_items,
        "pin_memory": bool(device.type == "cuda"),
    }
    if int(resolved_num_workers) > 0:
        dl_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    loader = DataLoader(ds, batch_size=int(resolved_batch_size), shuffle=False, **dl_kwargs)

    preprocessor = GpuStreamPreprocessor(
        GpuStreamConfig(
            device=device,
            video_backbone=str(model.video_backbone),
            sample_rate=int(args.sample_rate),
            max_audio_sec=float(args.max_audio_sec),
            audio_backend_mode=str(args.audio_backend),
            num_frames=int(args.num_frames),
            flow_size=int(args.flow_size),
            rgb_size=int(args.rgb_size),
            zero_video=bool(args.zero_video),
            zero_audio=bool(args.zero_audio),
            zero_text=bool(args.zero_text),
            prosody_use_pitch=(not bool(args.prosody_no_pitch)),
            video_decode_backend=str(args.video_decode_backend),
            flow_backend=str(args.flow_backend),
        )
    )

    attempted = 0
    error_count = 0
    print_every = max(0, int(args.print_every))
    error_types: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    backend_logged = False

    iterator = loader
    if not bool(args.no_progress):
        iterator = tqdm(loader, desc="Batch inference [gpu_stream]", unit="batch", dynamic_ncols=True)

    for batch_items in iterator:
        attempted += len(batch_items)
        try:
            batch_records = _infer_batch(
                items=batch_items,
                preprocessor=preprocessor,
                tokenizer=tokenizer,
                text_policy=str(text_policy),
                max_text_len=128,
                model=model,
                use_intensity=bool(use_intensity),
                amp_mode=resolved_amp_mode,
            )
            if not backend_logged:
                print(
                    json.dumps(
                        {
                            "gpu_stream_backends": preprocessor.backend_summary(),
                            "gpu_stream_prepare_stats": preprocessor.consume_prepare_stats(),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                backend_logged = True
            records.extend(batch_records)
        except Exception as batch_err:
            if bool(args.fail_fast):
                raise
            if len(batch_items) > 1:
                for item in batch_items:
                    try:
                        records.extend(
                            _infer_batch(
                                items=[item],
                                preprocessor=preprocessor,
                                tokenizer=tokenizer,
                                text_policy=str(text_policy),
                                max_text_len=128,
                                model=model,
                                use_intensity=bool(use_intensity),
                                amp_mode=resolved_amp_mode,
                            )
                        )
                    except Exception as item_err:
                        error_count += 1
                        error_types[type(item_err).__name__] += 1
                        records.append(_format_error_record(item, item_err, text_max_len=int(args.text_max_len)))
            else:
                error_count += 1
                error_types[type(batch_err).__name__] += 1
                records.append(_format_error_record(batch_items[0], batch_err, text_max_len=int(args.text_max_len)))

        if print_every > 0 and len(records) > 0 and (len(records) % print_every) == 0:
            ok_records = [rec for rec in records if rec.get("status") == "ok"]
            correct = sum(1 for rec in ok_records if bool(rec.get("match")))
            acc = float(correct / max(1, len(ok_records)))
            print(f"[{len(records)}/{len(items)}] ok={len(ok_records)} err={error_count} acc={acc:.4f}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if str(args.print_result) in {"all", str(rec.get("status", "none"))}:
                line = json.dumps(rec, ensure_ascii=False) if bool(args.print_json) else (
                    f"{str(rec.get('status', 'ok')).upper()} stem={rec.get('stem')} "
                    f"label={rec.get('label')} pred={rec.get('pred')} p={rec.get('probability')}"
                )
                print(line, flush=True)
                if bool(args.print_text) and str(rec.get("text", "")):
                    print(f"    text={_truncate_text(str(rec.get('text', '')), int(args.text_max_len))}", flush=True)

    ok_records = [rec for rec in records if rec.get("status") == "ok"]
    y_true = [label_names.index(str(rec["label"])) for rec in ok_records]
    y_pred = [label_names.index(str(rec["pred"])) for rec in ok_records]
    class_summary = classification_summary(y_true, y_pred, label_names)

    reg_true = [float(rec["intensity_gt"]) for rec in ok_records if rec.get("intensity_gt") is not None and rec.get("pred_intensity") is not None]
    reg_pred = [float(rec["pred_intensity"]) for rec in ok_records if rec.get("intensity_gt") is not None and rec.get("pred_intensity") is not None]
    mse = None
    mae = None
    ccc_value = None
    if reg_true and reg_pred:
        rt = torch.tensor(reg_true, dtype=torch.float32)
        rp = torch.tensor(reg_pred, dtype=torch.float32)
        diff = (rp - rt).to(torch.float64)
        mse = float((diff * diff).mean().item())
        mae = float(diff.abs().mean().item())
        ccc_value = ccc(rp, rt)

    speaker_baseline = None
    speaker_only = None
    if args.subset in {"train", "val"} and task_mode == "confounded_7way":
        train_items = filter_manifest_items_for_task(select_manifest_items(manifest, "train"), task_mode, task_speaker_id)
        eval_items = filter_manifest_items_for_task(select_manifest_items(manifest, args.subset), task_mode, task_speaker_id)
        speaker_baseline = speaker_majority_baseline(train_items, eval_items, label_names)
        speaker_only = speaker_only_baseline(train_items, eval_items, label_names)

    metrics_summary = {
        "split_manifest": str(split_manifest_path),
        "subset": str(args.subset),
        "benchmark_tag": str(args.benchmark_tag),
        "dataset_kind": dataset_kind,
        "ablation": str(args.ablation),
        "task_mode": task_mode,
        "speaker_id": task_speaker_id,
        "text_policy": str(text_policy),
        "pipeline": "gpu_stream",
        "label_names": list(label_names),
        "zero_video": bool(args.zero_video),
        "zero_audio": bool(args.zero_audio),
        "zero_text": bool(args.zero_text),
        "checkpoint": str(ckpt_path),
        "output": str(out_path),
        "attempted": int(attempted),
        "ok": int(len(ok_records)),
        "errors": int(error_count),
        "skipped": 0,
        "accuracy_on_ok": float(class_summary.get("accuracy", 0.0)),
        "macro_f1_on_ok": float(class_summary.get("macro_f1", 0.0)),
        "per_class_recall": class_summary.get("per_class_recall", {}),
        "confusion_matrix": class_summary.get("confusion_matrix", []),
        "support": class_summary.get("support", {}),
        "pred_counts": class_summary.get("pred_counts", {}),
        "intensity_enabled": bool(use_intensity),
        "intensity_n": int(len(reg_true)),
        "intensity_mse": mse,
        "intensity_mae": mae,
        "intensity_ccc": ccc_value,
        "device": str(device),
        "runtime_profile": profile.to_jsonable(),
        "amp_mode": resolved_amp_mode,
        "resolved_batch_size": int(resolved_batch_size),
        "resolved_num_workers": int(resolved_num_workers),
        "resolved_prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
        "manifest_sha256": manifest_hash,
        "manifest_summary": manifest_summary,
        "validity": validity,
        "error_types": dict(sorted(error_types.items())),
    }
    if speaker_baseline is not None:
        metrics_summary["speaker_majority_baseline"] = speaker_baseline
    if speaker_only is not None:
        metrics_summary["speaker_only_baseline"] = speaker_only

    metrics_path = out_path.with_suffix(out_path.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
