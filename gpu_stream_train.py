from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# HuggingFace tokenizers may warn when DataLoader workers fork after tokenizers were used.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TASK_MODE_CHOICES = ("confounded_7way", "within_speaker")

torch: Any = None
DataLoader: Any = None
tqdm: Any = None
AudioAugConfig: Any = None
augment_wav: Any = None
normalize_wav: Any = None
ManifestIngressConfig: Any = None
StreamingManifestDataset: Any = None
cache_manifest_text_tokens: Any = None
collate_manifest_items: Any = None
GpuStreamConfig: Any = None
GpuStreamPreprocessor: Any = None
autocast_context: Any = None
ccc: Any = None
prepare_manifest_items_for_task: Any = None
print_manifest_label_hist: Any = None
resolve_ablation_flags: Any = None
save_metrics_and_plots: Any = None
set_seed: Any = None
to_jsonable: Any = None
write_best_val_inference_outputs: Any = None
write_results_summary: Any = None
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
ProsodyConfig: Any = None
extract_prosody_features_gpu: Any = None
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
    global AudioAugConfig, augment_wav, normalize_wav
    global ManifestIngressConfig, StreamingManifestDataset, cache_manifest_text_tokens, collate_manifest_items
    global GpuStreamConfig, GpuStreamPreprocessor
    global autocast_context, ccc, prepare_manifest_items_for_task, print_manifest_label_hist
    global resolve_ablation_flags, save_metrics_and_plots, set_seed, to_jsonable
    global write_best_val_inference_outputs, write_results_summary
    global build_validity_summary, filter_manifest_items_for_task, load_split_manifest
    global manifest_sha256, map_label_to_task_index, resolve_task_label_names, resolve_task_mode, select_manifest_items
    global classification_summary, speaker_majority_baseline, speaker_only_baseline
    global FusionClassifier, ProsodyConfig, extract_prosody_features_gpu
    global detect_runtime, resolve_amp_mode, resolve_batch_size, resolve_prefetch_factor, resolve_worker_count, select_device
    global resolve_text_policy
    if torch is not None:
        return

    import torch as _torch
    from torch.utils.data import DataLoader as _DataLoader
    from tqdm import tqdm as _tqdm

    from audio_aug import AudioAugConfig as _AudioAugConfig, augment_wav as _augment_wav, normalize_wav as _normalize_wav
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
        print_manifest_label_hist as _print_manifest_label_hist,
        resolve_ablation_flags as _resolve_ablation_flags,
        save_metrics_and_plots as _save_metrics_and_plots,
        set_seed as _set_seed,
        to_jsonable as _to_jsonable,
        write_best_val_inference_outputs as _write_best_val_inference_outputs,
        write_results_summary as _write_results_summary,
    )
    from manifest_utils import (
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
    from prosody import ProsodyConfig as _ProsodyConfig, extract_prosody_features_gpu as _extract_prosody_features_gpu
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
    AudioAugConfig = _AudioAugConfig
    augment_wav = _augment_wav
    normalize_wav = _normalize_wav
    ManifestIngressConfig = _ManifestIngressConfig
    StreamingManifestDataset = _StreamingManifestDataset
    cache_manifest_text_tokens = _cache_manifest_text_tokens
    collate_manifest_items = _collate_manifest_items
    GpuStreamConfig = _GpuStreamConfig
    GpuStreamPreprocessor = _GpuStreamPreprocessor
    autocast_context = _autocast_context
    ccc = _ccc
    prepare_manifest_items_for_task = _prepare_manifest_items_for_task
    print_manifest_label_hist = _print_manifest_label_hist
    resolve_ablation_flags = _resolve_ablation_flags
    save_metrics_and_plots = _save_metrics_and_plots
    set_seed = _set_seed
    to_jsonable = _to_jsonable
    write_best_val_inference_outputs = _write_best_val_inference_outputs
    write_results_summary = _write_results_summary
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
    ProsodyConfig = _ProsodyConfig
    extract_prosody_features_gpu = _extract_prosody_features_gpu
    detect_runtime = _detect_runtime
    resolve_amp_mode = _resolve_amp_mode
    resolve_batch_size = _resolve_batch_size
    resolve_prefetch_factor = _resolve_prefetch_factor
    resolve_worker_count = _resolve_worker_count
    select_device = _select_device
    resolve_text_policy = _resolve_text_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the manifest-driven gpu_stream mainline")
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--task-mode", type=str, default="confounded_7way", choices=list(TASK_MODE_CHOICES))
    p.add_argument("--speaker-id", type=str, default="")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=str, default="auto")
    p.add_argument("--num-workers", type=str, default="auto")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--monitor", type=str, default="val_f1", choices=["val_acc", "val_f1"])
    p.add_argument("--early-stop-patience", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/motion_prosody"))
    p.add_argument("--benchmark-tag", type=str, default="default")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--freeze-audio", action="store_true")
    p.add_argument("--freeze-video", action="store_true")
    p.add_argument("--freeze-flow", action="store_true")
    p.add_argument("--freeze-rgb", action="store_true")
    p.add_argument("--freeze-prosody", action="store_true")
    p.add_argument("--video-backbone", type=str, default="dual", choices=["flow", "videomae", "dual"])
    p.add_argument("--video-model", type=str, default="MCG-NJU/videomae-large")
    p.add_argument("--audio-model", type=str, default="microsoft/wavlm-large")
    p.add_argument("--audio-model-revision", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--fusion-mode", type=str, default="gated_text", choices=["gated_text", "concat"])
    p.add_argument("--gate-temperature", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--gate-scale", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--delta-scale", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--modality-dropout", type=float, default=0.1)
    p.add_argument("--text-model", type=str, default="xlm-roberta-large")
    p.add_argument("--text-policy", type=str, default="full", choices=["full", "mask_emotion_cues", "drop"])
    p.add_argument("--freeze-text", action="store_true")
    p.add_argument("--max-text-len", type=int, default=128)
    p.add_argument("--audio-aug", action="store_true")
    p.set_defaults(recompute_prosody_on_aug=True)
    p.add_argument(
        "--no-recompute-prosody-on-aug",
        dest="recompute_prosody_on_aug",
        action="store_false",
        help="When using --audio-aug, keep the precomputed prosody vectors.",
    )
    p.add_argument("--prosody-no-pitch", action="store_true")
    p.add_argument("--seed", type=int, default=42)
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
    p.add_argument("--amp-mode", type=str, default="auto", choices=["auto", "off", "fp16", "bf16"])
    p.add_argument("--use-intensity", action="store_true")
    p.add_argument("--intensity-loss", type=str, default="mse", choices=["mse", "mae"])
    p.add_argument("--intensity-weight", type=float, default=1.0)

    # Hidden compatibility args. They either translate to current behavior or fail fast.
    p.add_argument("--zero-video", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--zero-audio", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--zero-text", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cached-dataset", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--feature-cache", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--pipeline", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--cache-mode", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--ram-cache-size", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--train-split", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--split-mode", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--prefetch-factor", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--persistent-workers", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--pin-memory", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--amp", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def _validate_compat_args(args: argparse.Namespace) -> None:
    if args.cached_dataset is not None or args.feature_cache is not None:
        raise RuntimeError(
            "Mainline training no longer supports --cached-dataset or --feature-cache. "
            "Use --split-manifest with the gpu_stream path."
        )
    pipeline = str(args.pipeline or "").strip().lower()
    if pipeline and pipeline not in {"auto", "gpu_stream"}:
        raise RuntimeError(f"Unsupported retired --pipeline value: {args.pipeline}")
    cache_mode = str(args.cache_mode or "").strip().lower()
    if cache_mode and cache_mode not in {"none"}:
        raise RuntimeError(
            "Mainline training no longer supports cache modes other than --cache-mode none. "
            "The manifest/gpu_stream path is now the only active runtime."
        )
    if args.ram_cache_size not in {None, 0}:
        raise RuntimeError("--ram-cache-size is retired in gpu_stream-only training.")
    for name in ("train_split", "split_mode", "prefetch_factor"):
        if getattr(args, name) is not None:
            raise RuntimeError(f"--{name.replace('_', '-')} is retired in gpu_stream-only training.")
    if bool(args.persistent_workers) or bool(args.pin_memory) or bool(args.amp):
        raise RuntimeError("Legacy performance toggles are retired; mainline training now resolves them automatically.")


def _load_tokenizer(model_name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required for text-enabled training.") from e
    return AutoTokenizer.from_pretrained(str(model_name))


def _run_phase(
    *,
    phase: str,
    model: FusionClassifier,
    loader: DataLoader,
    preprocessor: GpuStreamPreprocessor,
    tokenizer: Any | None,
    args: argparse.Namespace,
    device: torch.device,
    amp_mode: str,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    prosody_cfg: ProsodyConfig,
    aug_cfg: AudioAugConfig,
) -> dict[str, Any]:
    _lazy_runtime_imports()
    is_train = optimizer is not None
    model.train(is_train)
    label_names = list(getattr(model, "_label_names", []))

    total = 0
    correct = 0
    loss_sum = 0.0
    prepare_sec = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    reg_true: list[float] = []
    reg_pred: list[float] = []
    records: list[dict[str, Any]] = []
    backend_logged = False

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=phase, unit="batch", dynamic_ncols=True)

    for batch_items in iterator:
        prep_started = time.perf_counter()
        batch = preprocessor.prepare_batch(
            batch_items,
            tokenizer=tokenizer,
            text_policy=str(args.text_policy),
            max_text_len=int(args.max_text_len),
        )
        prepare_sec += float(time.perf_counter() - prep_started)

        if not backend_logged:
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "gpu_stream_backends": preprocessor.backend_summary(),
                        "gpu_stream_prepare_stats": preprocessor.consume_prepare_stats(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            backend_logged = True

        wav = batch["audio"]
        audio_lens = batch["audio_lens"]
        prosody = batch["prosody"]
        flow = batch.get("flow", None)
        rgb = batch.get("rgb", None)
        text_inputs = batch.get("text_inputs", None)
        labels = batch["labels"]

        if is_train and bool(args.audio_aug) and (not bool(args.zero_audio)):
            wav = normalize_wav(wav, target_rms=0.1)
            wav = augment_wav(wav, aug_cfg)
            wav = normalize_wav(wav, target_rms=0.1)
            if bool(args.recompute_prosody_on_aug):
                prosody = extract_prosody_features_gpu(wav, prosody_cfg, lengths=audio_lens).to(torch.float32)

        grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
        with grad_ctx, autocast_context(device, amp_mode):
            if bool(args.use_intensity):
                logits, pred_intensity = model(
                    flow,
                    wav,
                    prosody,
                    text_inputs=text_inputs,
                    return_intensity=True,
                    rgb=rgb,
                    audio_lens=audio_lens,
                )
            else:
                pred_intensity = None
                logits = model(
                    flow,
                    wav,
                    prosody,
                    text_inputs=text_inputs,
                    rgb=rgb,
                    audio_lens=audio_lens,
                )

            loss = loss_fn(logits, labels)
            if bool(args.use_intensity):
                intensity = batch["intensity"].to(torch.float32)
                intensity_mask = batch["intensity_mask"]
                if bool(intensity_mask.any().item()):
                    pred_f = pred_intensity.to(torch.float32)[intensity_mask]
                    gt_f = intensity[intensity_mask]
                    if str(args.intensity_loss).strip().lower() == "mae":
                        reg_loss = torch.abs(pred_f - gt_f).mean()
                    else:
                        reg_loss = torch.square(pred_f - gt_f).mean()
                    loss = loss + float(args.intensity_weight) * reg_loss

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and bool(scaler.is_enabled()):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        pred = logits.argmax(dim=1)
        batch_size = int(labels.shape[0])
        total += batch_size
        correct += int((pred == labels).sum().item())
        loss_sum += float(loss.item()) * batch_size

        labels_cpu = labels.detach().cpu().tolist()
        pred_cpu = pred.detach().cpu().tolist()
        y_true.extend(int(x) for x in labels_cpu)
        y_pred.extend(int(x) for x in pred_cpu)

        if bool(args.use_intensity) and pred_intensity is not None:
            intensity_cpu = batch["intensity"].detach().cpu().to(torch.float32)
            mask_cpu = batch["intensity_mask"].detach().cpu()
            pred_intensity_cpu = pred_intensity.detach().cpu().to(torch.float32)
            for idx in range(batch_size):
                if bool(mask_cpu[idx].item()):
                    reg_true.append(float(intensity_cpu[idx].item()))
                    reg_pred.append(float(pred_intensity_cpu[idx].item()))

        if not is_train:
            probs = torch.softmax(logits, dim=1)
            pred_probs = probs.gather(1, pred.view(-1, 1)).squeeze(1).detach().cpu().tolist()
            gt_intensity = batch["intensity"].detach().cpu().to(torch.float32)
            mask_cpu = batch["intensity_mask"].detach().cpu()
            pred_intensity_cpu = pred_intensity.detach().cpu().to(torch.float32) if pred_intensity is not None else None
            for idx, stem in enumerate(batch["stems"]):
                rec = {
                    "stem": str(stem),
                    "status": "ok",
                    "label": label_names[int(labels_cpu[idx])],
                    "pred": label_names[int(pred_cpu[idx])],
                    "match": bool(int(labels_cpu[idx]) == int(pred_cpu[idx])),
                    "probability": float(pred_probs[idx]),
                    "speaker_id": str(batch["speaker_id"][idx]),
                }
                if bool(args.use_intensity):
                    if bool(mask_cpu[idx].item()):
                        rec["intensity_gt"] = float(gt_intensity[idx].item())
                        rec["pred_intensity"] = float(pred_intensity_cpu[idx].item()) if pred_intensity_cpu is not None else None
                    else:
                        rec["intensity_gt"] = None
                        rec["pred_intensity"] = None
                records.append(rec)

    summary = classification_summary(y_true, y_pred, label_names)

    mse = 0.0
    mae = 0.0
    ccc_value = 0.0
    intensity_n = int(len(reg_true))
    if reg_true and reg_pred:
        rt = torch.tensor(reg_true, dtype=torch.float32)
        rp = torch.tensor(reg_pred, dtype=torch.float32)
        diff = (rp - rt).to(torch.float64)
        mse = float((diff * diff).mean().item())
        mae = float(diff.abs().mean().item())
        ccc_value = ccc(rp, rt)

    return {
        "loss": float(loss_sum / max(1, total)),
        "accuracy": float(correct / max(1, total)),
        "macro_f1": float(summary.get("macro_f1", 0.0)),
        "class_summary": summary,
        "prepare_sec": float(prepare_sec),
        "records": records,
        "intensity_n": intensity_n,
        "mse": mse,
        "mae": mae,
        "ccc": ccc_value,
    }


def main() -> None:
    args = parse_args()
    _validate_compat_args(args)
    _lazy_runtime_imports()

    args.text_policy = resolve_text_policy(args.text_policy)
    zero_video, zero_audio, zero_text = resolve_ablation_flags(
        ablation=str(args.ablation),
        zero_video=bool(args.zero_video),
        zero_audio=bool(args.zero_audio),
        zero_text=bool(args.zero_text),
    )
    if str(args.text_policy) == "drop":
        zero_text = True
    args.zero_video = bool(zero_video)
    args.zero_audio = bool(zero_audio)
    args.zero_text = bool(zero_text)

    set_seed(int(args.seed))
    profile = detect_runtime(args.device)
    device = select_device(args.device)
    resolved_amp_mode = resolve_amp_mode(str(args.amp_mode), profile)
    print(f"Using device: {device}", flush=True)
    print(f"Runtime profile: {json.dumps(profile.to_jsonable(), ensure_ascii=False)}", flush=True)

    split_manifest_path = args.split_manifest.expanduser()
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
    manifest = load_split_manifest(split_manifest_path)
    manifest_hash = manifest_sha256(split_manifest_path)
    manifest_summary = manifest.get("summary", {})
    dataset_kind = str(manifest.get("dataset_kind", "") or "")

    task_mode = resolve_task_mode(args.task_mode)
    task_speaker_id = str(args.speaker_id).strip().upper() or None
    label_names = resolve_task_label_names(task_mode, task_speaker_id)
    validity_summary = build_validity_summary(manifest_summary, task_mode, task_speaker_id)

    train_items = prepare_manifest_items_for_task(
        filter_manifest_items_for_task(select_manifest_items(manifest, "train"), task_mode, task_speaker_id),
        task_mode=task_mode,
        speaker_id=task_speaker_id,
        map_label_to_task_index=map_label_to_task_index,
    )
    val_items = prepare_manifest_items_for_task(
        filter_manifest_items_for_task(select_manifest_items(manifest, "val"), task_mode, task_speaker_id),
        task_mode=task_mode,
        speaker_id=task_speaker_id,
        map_label_to_task_index=map_label_to_task_index,
    )
    if not train_items or not val_items:
        raise RuntimeError("Split manifest produced an empty train or val subset for the selected task.")

    print(f"Using split manifest: {split_manifest_path}", flush=True)
    print(f"Manifest sha256: {manifest_hash}", flush=True)
    print_manifest_label_hist("train", train_items, label_names)
    print_manifest_label_hist("val", val_items, label_names)

    speaker_baseline = None
    speaker_only = None
    if task_mode == "confounded_7way":
        speaker_baseline = speaker_majority_baseline(train_items, val_items, label_names)
        speaker_only = speaker_only_baseline(train_items, val_items, label_names)

    ingress_cfg = ManifestIngressConfig(
        sample_rate=int(args.sample_rate),
        max_audio_sec=float(args.max_audio_sec),
        audio_backend_mode=str(args.audio_backend),
        video_decode_backend=str(args.video_decode_backend),
        num_frames=int(args.num_frames),
        zero_audio=bool(args.zero_audio),
        zero_video=bool(args.zero_video),
        video_backbone=str(args.video_backbone),
    )
    train_ds = StreamingManifestDataset(train_items, ingress=ingress_cfg)
    val_ds = StreamingManifestDataset(val_items, ingress=ingress_cfg)

    tokenizer = None
    if not bool(args.zero_text):
        tokenizer = _load_tokenizer(str(args.text_model))
        cache_manifest_text_tokens(train_ds.items, tokenizer, max_text_len=int(args.max_text_len), text_policy=str(args.text_policy))
        cache_manifest_text_tokens(val_ds.items, tokenizer, max_text_len=int(args.max_text_len), text_policy=str(args.text_policy))

    resolved_batch_size = resolve_batch_size(
        args.batch_size,
        phase="train",
        profile=profile,
        feature_cache=False,
        video_backbone=str(args.video_backbone),
        freeze_audio=bool(args.freeze_audio),
        freeze_text=bool(args.freeze_text),
        freeze_flow=bool(args.freeze_video or args.freeze_flow),
        freeze_rgb=bool(args.freeze_video or args.freeze_rgb),
    )
    resolved_num_workers = resolve_worker_count(
        args.num_workers,
        phase="train",
        profile=profile,
        dataset_in_memory=False,
        total_items=len(train_items) + len(val_items),
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

    train_loader = DataLoader(train_ds, batch_size=int(resolved_batch_size), shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=int(resolved_batch_size), shuffle=False, **dl_kwargs)

    model = FusionClassifier(
        num_classes=len(label_names),
        freeze_audio=bool(args.freeze_audio),
        freeze_video=bool(args.freeze_video),
        freeze_flow=bool(args.freeze_video or args.freeze_flow),
        freeze_rgb=bool(args.freeze_video or args.freeze_rgb),
        freeze_prosody=bool(args.freeze_prosody),
        text_model=str(args.text_model),
        freeze_text=bool(args.freeze_text),
        audio_model=str(args.audio_model),
        audio_model_revision=(str(args.audio_model_revision).strip() or None),
        video_backbone=str(args.video_backbone),
        video_model=str(args.video_model),
        fusion_mode=str(args.fusion_mode),
        modality_dropout=float(args.modality_dropout),
        gate_temperature=float(args.gate_temperature),
        gate_scale=float(args.gate_scale),
        delta_scale=float(args.delta_scale),
        intensity_head=bool(args.use_intensity),
    ).to(device)
    setattr(model, "_label_names", list(label_names))

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = torch.nn.CrossEntropyLoss()

    stream_preprocessor = GpuStreamPreprocessor(
        GpuStreamConfig(
            device=device,
            video_backbone=str(args.video_backbone),
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
    aug_cfg = AudioAugConfig(sample_rate=int(args.sample_rate))
    prosody_cfg = ProsodyConfig(sample_rate=int(args.sample_rate), use_pitch=(not bool(args.prosody_no_pitch)))
    scaler = None
    if device.type == "cuda":
        amp_mod = getattr(torch, "amp", None)
        if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
            scaler = amp_mod.GradScaler("cuda", enabled=(resolved_amp_mode == "fp16"))  # type: ignore[attr-defined]
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=(resolved_amp_mode == "fp16"))

    print(
        json.dumps(
            {
                "resolved_batch_size": int(resolved_batch_size),
                "resolved_num_workers": int(resolved_num_workers),
                "resolved_prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
                "pin_memory": bool(device.type == "cuda"),
                "pipeline": "gpu_stream",
                "amp_mode": resolved_amp_mode,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    out_dir = args.output_dir.expanduser()
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    history: dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "train_per_class_recall": [],
        "val_per_class_recall": [],
    }
    if bool(args.use_intensity):
        history.update(
            {
                "train_mse": [],
                "train_mae": [],
                "train_ccc": [],
                "train_intensity_n": [],
                "val_mse": [],
                "val_mae": [],
                "val_ccc": [],
                "val_intensity_n": [],
            }
        )

    best_epoch = 0
    best_acc = 0.0
    best_f1 = 0.0
    best_monitor_value = float("-inf")
    best_val_summary: dict[str, Any] = {}
    best_train_summary: dict[str, Any] = {}
    best_val_records: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    final_epoch = 0

    for epoch in range(1, int(args.epochs) + 1):
        final_epoch = int(epoch)
        train_stats = _run_phase(
            phase=f"Epoch {epoch}/{args.epochs} [train]",
            model=model,
            loader=train_loader,
            preprocessor=stream_preprocessor,
            tokenizer=tokenizer,
            args=args,
            device=device,
            amp_mode=resolved_amp_mode,
            loss_fn=loss_fn,
            optimizer=opt,
            scaler=scaler,
            prosody_cfg=prosody_cfg,
            aug_cfg=aug_cfg,
        )
        val_stats = _run_phase(
            phase=f"Epoch {epoch}/{args.epochs} [val]",
            model=model,
            loader=val_loader,
            preprocessor=stream_preprocessor,
            tokenizer=tokenizer,
            args=args,
            device=device,
            amp_mode=resolved_amp_mode,
            loss_fn=loss_fn,
            optimizer=None,
            scaler=None,
            prosody_cfg=prosody_cfg,
            aug_cfg=aug_cfg,
        )

        print(
            "Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} "
            "val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            "train_prepare_s={train_prepare:.2f} val_prepare_s={val_prepare:.2f}".format(
                epoch=epoch,
                train_loss=train_stats["loss"],
                train_acc=train_stats["accuracy"],
                train_f1=train_stats["macro_f1"],
                val_loss=val_stats["loss"],
                val_acc=val_stats["accuracy"],
                val_f1=val_stats["macro_f1"],
                train_prepare=train_stats["prepare_sec"],
                val_prepare=val_stats["prepare_sec"],
            ),
            flush=True,
        )
        if bool(args.use_intensity):
            print(
                "Intensity: train_mse={train_mse:.4f} train_mae={train_mae:.4f} train_ccc={train_ccc:.4f} (n={train_n}) "
                "val_mse={val_mse:.4f} val_mae={val_mae:.4f} val_ccc={val_ccc:.4f} (n={val_n})".format(
                    train_mse=train_stats["mse"],
                    train_mae=train_stats["mae"],
                    train_ccc=train_stats["ccc"],
                    train_n=train_stats["intensity_n"],
                    val_mse=val_stats["mse"],
                    val_mae=val_stats["mae"],
                    val_ccc=val_stats["ccc"],
                    val_n=val_stats["intensity_n"],
                ),
                flush=True,
            )

        history["train_loss"].append(float(train_stats["loss"]))
        history["train_acc"].append(float(train_stats["accuracy"]))
        history["train_f1"].append(float(train_stats["macro_f1"]))
        history["val_loss"].append(float(val_stats["loss"]))
        history["val_acc"].append(float(val_stats["accuracy"]))
        history["val_f1"].append(float(val_stats["macro_f1"]))
        history["train_per_class_recall"].append(dict(train_stats["class_summary"].get("per_class_recall", {})))
        history["val_per_class_recall"].append(dict(val_stats["class_summary"].get("per_class_recall", {})))
        if bool(args.use_intensity):
            history["train_mse"].append(float(train_stats["mse"]))
            history["train_mae"].append(float(train_stats["mae"]))
            history["train_ccc"].append(float(train_stats["ccc"]))
            history["train_intensity_n"].append(float(train_stats["intensity_n"]))
            history["val_mse"].append(float(val_stats["mse"]))
            history["val_mae"].append(float(val_stats["mae"]))
            history["val_ccc"].append(float(val_stats["ccc"]))
            history["val_intensity_n"].append(float(val_stats["intensity_n"]))

        monitor_value = float(val_stats["macro_f1"] if str(args.monitor) == "val_f1" else val_stats["accuracy"])
        improved = bool(best_epoch == 0 or monitor_value > best_monitor_value)
        if improved:
            best_epoch = int(epoch)
            best_acc = float(val_stats["accuracy"])
            best_f1 = float(val_stats["macro_f1"])
            best_monitor_value = float(monitor_value)
            best_train_summary = dict(train_stats["class_summary"])
            best_val_summary = dict(val_stats["class_summary"])
            best_val_records = list(val_stats["records"])
            epochs_without_improvement = 0
            ckpt = {
                "model": model.state_dict(),
                "epoch": int(epoch),
                "best_acc": float(best_acc),
                "best_f1": float(best_f1),
                "best_monitor_name": str(args.monitor),
                "best_monitor_value": float(best_monitor_value),
                "args": to_jsonable(vars(args)),
                "label_names": list(label_names),
                "task_mode": task_mode,
                "speaker_id": task_speaker_id,
                "text_policy": str(args.text_policy),
                "validity": validity_summary,
                "resolved_runtime": {
                    "device": str(device),
                    "profile": profile.to_jsonable(),
                    "batch_size": int(resolved_batch_size),
                    "num_workers": int(resolved_num_workers),
                    "prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
                    "amp_mode": resolved_amp_mode,
                    "pipeline": "gpu_stream",
                },
                "benchmark_tag": str(args.benchmark_tag),
                "dataset_kind": dataset_kind,
                "split_manifest": str(split_manifest_path),
                "manifest_sha256": manifest_hash,
                "ablation": str(args.ablation),
                "zero_video": bool(args.zero_video),
                "zero_audio": bool(args.zero_audio),
                "zero_text": bool(args.zero_text),
                "best_val_summary": best_val_summary,
            }
            torch.save(ckpt, out_dir / "checkpoints" / "best.pt")

            val_metrics_summary = {
                "split_manifest": str(split_manifest_path),
                "subset": "val",
                "benchmark_tag": str(args.benchmark_tag),
                "checkpoint": str(out_dir / "checkpoints" / "best.pt"),
                "task_mode": task_mode,
                "speaker_id": task_speaker_id,
                "text_policy": str(args.text_policy),
                "label_names": list(label_names),
                "attempted": len(best_val_records),
                "ok": len(best_val_records),
                "errors": 0,
                "skipped": 0,
                "accuracy_on_ok": float(best_acc),
                "macro_f1_on_ok": float(best_f1),
                "per_class_recall": best_val_summary.get("per_class_recall", {}),
                "confusion_matrix": best_val_summary.get("confusion_matrix", []),
                "support": best_val_summary.get("support", {}),
                "pred_counts": best_val_summary.get("pred_counts", {}),
                "manifest_sha256": manifest_hash,
                "manifest_summary": manifest_summary,
                "validity": validity_summary,
                "speaker_majority_baseline": speaker_baseline,
                "speaker_only_baseline": speaker_only,
                "pipeline": "gpu_stream",
            }
            write_best_val_inference_outputs(out_dir, records=best_val_records, metrics_summary=val_metrics_summary)
            print(f"Saved best checkpoint -> {out_dir / 'checkpoints' / 'best.pt'}", flush=True)
        else:
            epochs_without_improvement += 1

        if int(args.early_stop_patience) > 0 and epochs_without_improvement >= int(args.early_stop_patience):
            print(
                f"Early stopping at epoch {epoch}: no improvement on {args.monitor} for {epochs_without_improvement} epoch(s).",
                flush=True,
            )
            break

    last_ckpt = {
        "model": model.state_dict(),
        "epoch": int(final_epoch),
        "best_acc": float(best_acc),
        "best_f1": float(best_f1),
        "best_monitor_name": str(args.monitor),
        "best_monitor_value": float(best_monitor_value if best_monitor_value != float("-inf") else 0.0),
        "args": to_jsonable(vars(args)),
        "label_names": list(label_names),
        "task_mode": task_mode,
        "speaker_id": task_speaker_id,
        "text_policy": str(args.text_policy),
        "validity": validity_summary,
        "resolved_runtime": {
            "device": str(device),
            "profile": profile.to_jsonable(),
            "batch_size": int(resolved_batch_size),
            "num_workers": int(resolved_num_workers),
            "prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
            "amp_mode": resolved_amp_mode,
            "pipeline": "gpu_stream",
        },
    }
    torch.save(last_ckpt, out_dir / "checkpoints" / "last.pt")

    metrics_payload: dict[str, Any] = dict(history)
    metrics_payload["meta"] = {
        "benchmark_tag": str(args.benchmark_tag),
        "split_manifest": str(split_manifest_path),
        "manifest_sha256": manifest_hash,
        "task_mode": task_mode,
        "speaker_id": task_speaker_id,
        "text_policy": str(args.text_policy),
        "label_names": list(label_names),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "ablation": str(args.ablation),
        "zero_video": bool(args.zero_video),
        "zero_audio": bool(args.zero_audio),
        "zero_text": bool(args.zero_text),
        "args": to_jsonable(vars(args)),
        "device": str(device),
        "runtime_profile": profile.to_jsonable(),
        "resolved_batch_size": int(resolved_batch_size),
        "resolved_num_workers": int(resolved_num_workers),
        "resolved_prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
        "amp_mode": resolved_amp_mode,
        "pipeline": "gpu_stream",
        "manifest_summary": manifest_summary,
    }
    metrics_payload["best"] = {
        "epoch": int(best_epoch),
        "best_acc": float(best_acc),
        "best_f1": float(best_f1),
        "best_monitor_name": str(args.monitor),
        "best_monitor_value": float(best_monitor_value if best_monitor_value != float("-inf") else 0.0),
        "best_train_summary": best_train_summary,
        "best_val_summary": best_val_summary,
    }
    metrics_payload["validity"] = validity_summary
    if speaker_baseline is not None:
        metrics_payload["speaker_majority_baseline"] = speaker_baseline
    if speaker_only is not None:
        metrics_payload["speaker_only_baseline"] = speaker_only

    save_metrics_and_plots(out_dir, metrics_payload)
    write_results_summary(out_dir, metrics_payload)
    print(f"Best val macro_f1={best_f1:.4f}, acc={best_acc:.4f}, epoch={best_epoch}", flush=True)


if __name__ == "__main__":
    main()
