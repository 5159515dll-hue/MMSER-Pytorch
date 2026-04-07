"""主线训练/推理共享工具。

这个模块承载随机种子、paper-grade 元数据、结果落盘、自动混合精度上下文等
跨文件复用逻辑。它本身不执行训练，但训练与推理的很多关键步骤都依赖这里。
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

PAPER_PROTOCOL_VERSION = "paper_grade_v1"
PAPER_MULTI_SEED = [13, 17, 23, 42, 3407]
LEGACY_FLOW_VIDEO_ENCODER_VARIANT = "flow3d_maxpool_v1"
FLOW_VIDEO_ENCODER_VARIANT = "flow3d_strideconv_mean_v3"


def _safe_module_version(name: str) -> str | None:
    """安全读取一个已安装 Python 包的版本号。"""

    try:
        return str(importlib.metadata.version(name))
    except Exception:
        return None


def _safe_git_output(args: list[str], *, cwd: Path | None = None) -> str | None:
    """安全执行只读 git 命令，并返回裁剪后的 stdout。"""

    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    text = str(proc.stdout or "").strip()
    return text or None


def set_seed(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Fix Python / NumPy / PyTorch RNG state and return the active policy."""

    import random
    import numpy as np

    seed_int = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed_int))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        # 对 CUDA 来说，仅仅设置 seed 还不够；
        # 还要告诉 cuBLAS / cuDNN 尽量走确定性路径。
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    cudnn_enabled = bool(getattr(cudnn_backend, "enabled", False)) if cudnn_backend is not None else False
    deterministic_enabled = False
    warn_only = None
    if deterministic:
        use_det = getattr(torch, "use_deterministic_algorithms", None)
        if callable(use_det):
            try:
                use_det(True)
                deterministic_enabled = True
            except TypeError:
                use_det(True, warn_only=False)
                deterministic_enabled = True
        if cudnn_backend is not None:
            try:
                cudnn_backend.deterministic = True
                cudnn_backend.benchmark = False
            except Exception:
                pass
    else:
        is_det_enabled = getattr(torch, "are_deterministic_algorithms_enabled", None)
        if callable(is_det_enabled):
            deterministic_enabled = bool(is_det_enabled())
    is_det_enabled = getattr(torch, "are_deterministic_algorithms_enabled", None)
    if callable(is_det_enabled):
        deterministic_enabled = bool(is_det_enabled())
    is_warn_only_enabled = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
    if callable(is_warn_only_enabled):
        warn_only = bool(is_warn_only_enabled())
    return {
        "seed": seed_int,
        "deterministic_requested": bool(deterministic),
        "deterministic_algorithms_enabled": bool(deterministic_enabled),
        "deterministic_warn_only": warn_only,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cudnn_enabled": bool(cudnn_enabled),
        "cudnn_deterministic": bool(getattr(cudnn_backend, "deterministic", False)) if cudnn_backend is not None else None,
        "cudnn_benchmark": bool(getattr(cudnn_backend, "benchmark", False)) if cudnn_backend is not None else None,
    }


def make_torch_generator(seed: int) -> torch.Generator:
    """Build a deterministic torch Generator for DataLoader shuffles."""

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def make_dataloader_worker_init_fn(base_seed: int) -> Any:
    """Build a worker init fn that deterministically seeds each worker."""

    seed_int = int(base_seed)

    def _seed_worker(worker_id: int) -> None:
        import random
        import numpy as np

        worker_seed = seed_int + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

    return _seed_worker


def build_run_contract(
    *,
    split_manifest: Path,
    manifest_sha256: str,
    dataset_kind: str,
    task_mode: str,
    speaker_id: str | None,
    text_policy: str,
    label_names: list[str],
    validity_summary: dict[str, Any],
    ablation: str,
    zero_video: bool,
    zero_audio: bool,
    zero_text: bool,
    use_intensity: bool,
    video_backbone: str,
    flow_encoder_variant: str,
    text_model: str,
    max_text_len: int,
    rgb_size: int,
    sample_rate: int,
    max_audio_sec: float,
    num_frames: int,
) -> dict[str, Any]:
    """Build the immutable experiment contract used by training and inference."""

    return {
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "split_manifest": str(split_manifest),
        "manifest_sha256": str(manifest_sha256),
        "dataset_kind": str(dataset_kind),
        "task_mode": str(task_mode),
        "speaker_id": str(speaker_id).strip().upper() if speaker_id is not None and str(speaker_id).strip() else None,
        "text_policy": str(text_policy),
        "label_names": list(label_names),
        "claim_scope": str(validity_summary.get("claim_scope", "")),
        "scientific_validity": bool(validity_summary.get("scientific_validity", False)),
        "ablation": str(ablation),
        "zero_video": bool(zero_video),
        "zero_audio": bool(zero_audio),
        "zero_text": bool(zero_text),
        "use_intensity": bool(use_intensity),
        "video_backbone": str(video_backbone),
        "flow_encoder_variant": str(flow_encoder_variant),
        "text_model": str(text_model),
        "max_text_len": int(max_text_len),
        "rgb_size": int(rgb_size),
        "sample_rate": int(sample_rate),
        "max_audio_sec": float(max_audio_sec),
        "num_frames": int(num_frames),
    }


def build_run_provenance(
    *,
    runtime_profile: dict[str, Any],
    deterministic_policy: dict[str, Any],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Collect a compact provenance record for later reproduction."""

    root = repo_root.expanduser() if repo_root is not None else Path(__file__).resolve().parent
    git_commit = _safe_git_output(["git", "rev-parse", "HEAD"], cwd=root)
    git_short_commit = _safe_git_output(["git", "rev-parse", "--short", "HEAD"], cwd=root)
    git_dirty = _safe_git_output(["git", "status", "--porcelain"], cwd=root)
    cudnn_version = None
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        version_fn = getattr(cudnn_backend, "version", None)
        if callable(version_fn):
            try:
                cudnn_version = version_fn()
            except Exception:
                cudnn_version = None
    return {
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn_version": cudnn_version,
        "torchaudio_version": _safe_module_version("torchaudio"),
        "transformers_version": _safe_module_version("transformers"),
        "decord_version": _safe_module_version("decord"),
        "git_commit": git_commit,
        "git_short_commit": git_short_commit,
        "git_is_dirty": bool(git_dirty),
        "runtime_profile": dict(runtime_profile),
        "deterministic_policy": dict(deterministic_policy),
    }


def build_paper_grade(
    *,
    validity_summary: dict[str, Any],
    ineligibility_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize whether a run satisfies the paper-grade protocol."""

    reasons = [str(reason) for reason in (ineligibility_reasons or []) if str(reason).strip()]
    return {
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "eligible": len(reasons) == 0,
        "ineligibility_reasons": reasons,
        "scientific_validity": bool(validity_summary.get("scientific_validity", False)),
        "claim_scope": str(validity_summary.get("claim_scope", "")),
    }


def autocast_context(device: torch.device, amp_mode: str) -> contextlib.AbstractContextManager[Any]:
    """Build a torch autocast context from the resolved AMP mode."""

    if device.type != "cuda" or str(amp_mode) == "off":
        return contextlib.nullcontext()
    amp_mod = getattr(torch, "amp", None)
    amp_dtype = torch.bfloat16 if str(amp_mode) == "bf16" else torch.float16
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        return amp_mod.autocast("cuda", enabled=True, dtype=amp_dtype)  # type: ignore[attr-defined]
    return torch.cuda.amp.autocast(enabled=True)


def resolve_ablation_flags(
    *,
    ablation: str,
    zero_video: bool = False,
    zero_audio: bool = False,
    zero_text: bool = False,
) -> tuple[bool, bool, bool]:
    """Merge an ablation preset with explicit zero-* overrides."""

    presets = {
        "full": (False, False, False),
        "text-only": (True, True, False),
        "audio-only": (True, False, True),
        "video-only": (False, True, True),
        "no-text": (False, False, True),
        "no-audio": (False, True, False),
        "no-video": (True, False, False),
    }
    preset = presets.get(str(ablation or "full").strip().lower())
    if preset is not None:
        zero_video = bool(zero_video or preset[0])
        zero_audio = bool(zero_audio or preset[1])
        zero_text = bool(zero_text or preset[2])
    return bool(zero_video), bool(zero_audio), bool(zero_text)


def prepare_manifest_items_for_task(
    items: list[dict[str, Any]],
    *,
    task_mode: str,
    speaker_id: str | None,
    map_label_to_task_index: Any,
) -> list[dict[str, Any]]:
    """Attach task-local label_idx and drop items outside the selected task."""

    prepared: list[dict[str, Any]] = []
    for item in items:
        label_en = str(item.get("label_en", "") or "")
        task_label_idx = map_label_to_task_index(label_en, task_mode, speaker_id)
        if task_label_idx is None:
            continue
        enriched = dict(item)
        enriched["label_idx"] = int(task_label_idx)
        prepared.append(enriched)
    return prepared


def print_manifest_label_hist(name: str, items: list[dict[str, Any]], label_names: list[str]) -> None:
    """Print a label histogram for a manifest subset."""

    counts = torch.zeros(len(label_names), dtype=torch.long)
    for item in items:
        idx = item.get("label_idx", None)
        if idx is None:
            continue
        idx_int = int(idx)
        if 0 <= idx_int < len(label_names):
            counts[idx_int] += 1
    total = int(counts.sum().item())
    parts = [f"{label_names[k]}:{int(v)}" for k, v in enumerate(counts.tolist())]
    print(f"{name} label histogram (n={total}): " + " ".join(parts), flush=True)


def ccc(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute concordance correlation coefficient."""

    x = x.flatten().to(torch.float64)
    y = y.flatten().to(torch.float64)
    if int(x.numel()) < 2:
        return 0.0
    mx = float(x.mean().item())
    my = float(y.mean().item())
    vx = float(x.var(unbiased=False).item())
    vy = float(y.var(unbiased=False).item())
    cov = float(((x - mx) * (y - my)).mean().item())
    denom = vx + vy + (mx - my) ** 2
    if denom <= 0:
        return 0.0
    return float((2.0 * cov) / denom)


def to_jsonable(value: Any) -> Any:
    """Recursively convert Path / tuple containers into JSON-serializable data."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_best_val_inference_outputs(
    out_dir: Path,
    *,
    records: list[dict[str, Any]],
    metrics_summary: dict[str, Any],
) -> None:
    """Write best validation JSONL and summary JSON."""

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "inference_val.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "inference_val.metrics.json").write_text(
        json.dumps(metrics_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_metrics_and_plots(out_dir: Path, metrics: dict[str, Any]) -> None:
    """Persist metrics.json and best-effort training plots."""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plot skipped (matplotlib not available): {type(e).__name__}: {e}", flush=True)
        return

    epochs = list(range(1, len(metrics.get("train_loss", [])) + 1))
    if not epochs:
        return

    label_names = list(metrics.get("meta", {}).get("label_names", []))
    try:
        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, metrics.get("train_acc", []), label="train")
        plt.plot(epochs, metrics.get("val_acc", []), label="val")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Accuracy vs Epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "accuracy_curve.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, metrics.get("train_loss", []), label="train")
        plt.plot(epochs, metrics.get("val_loss", []), label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss vs Epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, metrics.get("train_f1", []), label="train")
        plt.plot(epochs, metrics.get("val_f1", []), label="val")
        plt.xlabel("epoch")
        plt.ylabel("macro_f1")
        plt.title("Macro-F1 vs Epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "macro_f1_curve.png", dpi=160)
        plt.close()

        lr_history = metrics.get("lr", [])
        if lr_history:
            plt.figure(figsize=(8, 4.5))
            plt.plot(epochs[: len(lr_history)], lr_history, label="lr")
            plt.xlabel("epoch")
            plt.ylabel("learning_rate")
            plt.title("Learning Rate vs Epoch")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "lr_curve.png", dpi=160)
            plt.close()

        train_recall_history = metrics.get("train_per_class_recall", [])
        val_recall_history = metrics.get("val_per_class_recall", [])
        if train_recall_history or val_recall_history:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for ax, recall_history, title in (
                (axes[0], train_recall_history, "Train Per-Class Recall"),
                (axes[1], val_recall_history, "Val Per-Class Recall"),
            ):
                if not recall_history:
                    ax.set_title(f"{title} (empty)")
                    ax.grid(True, alpha=0.3)
                    continue
                for label_name in label_names:
                    values = [float(step.get(label_name, 0.0)) for step in recall_history]
                    ax.plot(epochs[: len(values)], values, label=label_name)
                ax.set_ylabel("recall")
                ax.set_title(title)
                ax.set_ylim(0.0, 1.05)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel("epoch")
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=min(4, max(1, len(labels))), frameon=False)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(out_dir / "per_class_recall_curve.png", dpi=160)
            plt.close(fig)

        best_val_summary = metrics.get("best", {}).get("best_val_summary", {})
        matrix = best_val_summary.get("confusion_matrix", [])
        if not isinstance(matrix, list) or not matrix:
            return
        plt.figure(figsize=(8.5, 7))
        ax = plt.gca()
        im = ax.imshow(matrix, cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Best Val Confusion Matrix")
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        ax.set_xticks(range(len(label_names)))
        ax.set_yticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha="right")
        ax.set_yticklabels(label_names)
        vmax = max((max(row) for row in matrix if row), default=0)
        threshold = vmax / 2.0 if vmax > 0 else 0.0
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(
                    j,
                    i,
                    str(int(value)),
                    ha="center",
                    va="center",
                    color="white" if float(value) > threshold else "black",
                    fontsize=8,
                )
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix_best_val.png", dpi=160)
        plt.close()
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        print(f"Plot skipped (plot generation failed): {type(e).__name__}: {e}", flush=True)


def write_results_summary(out_dir: Path, metrics: dict[str, Any]) -> None:
    """Generate a compact human-readable summary markdown."""

    out_dir.mkdir(parents=True, exist_ok=True)
    best = metrics.get("best", {})
    validity = metrics.get("validity", {})
    stop = metrics.get("stop", {})
    paper_grade = metrics.get("paper_grade", {})
    provenance = metrics.get("provenance", {})
    speaker_majority = metrics.get("speaker_majority_baseline", {})
    speaker_only = metrics.get("speaker_only_baseline", {})
    meta = metrics.get("meta", {})
    best_val = best.get("best_val_summary", {})
    input_cache_contract = meta.get("input_cache_contract", {}) if isinstance(meta.get("input_cache_contract"), dict) else {}
    best_bundle_relpath = str(meta.get("best_bundle_relpath", "") or "")
    lines = [
        "# Run Summary",
        "",
        f"- protocol_version: `{paper_grade.get('protocol_version', meta.get('paper_protocol_version'))}`",
        f"- paper_grade_eligible: `{paper_grade.get('eligible')}`",
        f"- ineligibility_reasons: `{paper_grade.get('ineligibility_reasons', [])}`",
        f"- task_mode: `{validity.get('task_mode', 'confounded_7way')}`",
        f"- speaker_id: `{validity.get('speaker_id')}`",
        f"- text_policy: `{meta.get('text_policy', 'full')}`",
        f"- input_cache: `{meta.get('input_cache')}`",
        f"- input_cache_in_memory: `{meta.get('input_cache_in_memory')}`",
        f"- input_cache_protocol: `{input_cache_contract.get('protocol_version') if input_cache_contract else None}`",
        f"- input_cache_manifest_sha256: `{input_cache_contract.get('manifest_sha256') if input_cache_contract else None}`",
        f"- scientific_validity: `{validity.get('scientific_validity')}`",
        f"- claim_scope: `{validity.get('claim_scope')}`",
        f"- recommended_interpretation: {validity.get('recommended_interpretation', '')}",
        f"- git_commit: `{provenance.get('git_short_commit')}`",
        "",
        "## Best Validation",
        "",
        f"- epoch: `{best.get('epoch')}`",
        f"- accuracy: `{best_val.get('accuracy')}`",
        f"- macro_f1: `{best_val.get('macro_f1')}`",
        f"- loss: `{best.get('best_val_loss')}`",
        f"- checkpoint_reason: `{best.get('checkpoint_reason')}`",
        f"- best_monitor_value: `{best.get('best_monitor_value')}`",
        f"- significant_best_monitor_value: `{best.get('significant_best_monitor_value')}`",
        "",
        "## Stop",
        "",
        f"- stop_reason: `{stop.get('reason')}`",
        f"- stop_epoch: `{stop.get('epoch')}`",
        f"- epochs_without_improvement: `{stop.get('epochs_without_improvement')}`",
        f"- lr_drop_epochs: `{stop.get('lr_drop_epochs')}`",
        "",
        "## Baselines",
        "",
        f"- speaker_majority_accuracy: `{speaker_majority.get('accuracy')}`",
        f"- speaker_majority_macro_f1: `{speaker_majority.get('macro_f1')}`",
        f"- speaker_only_accuracy: `{speaker_only.get('accuracy')}`",
        f"- speaker_only_macro_f1: `{speaker_only.get('macro_f1')}`",
        "",
        "## Artifacts",
        "",
        f"- published_metrics: `{meta.get('attempt_dir', '')}/published/metrics.json`",
        f"- published_last_checkpoint: `{meta.get('attempt_dir', '')}/published/last.pt`",
        f"- best_bundle: `{best_bundle_relpath}`",
        "- published plots: `published/*.png`",
    ]
    (out_dir / "results_summary.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
