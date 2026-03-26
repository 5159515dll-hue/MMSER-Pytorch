from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import torch


def set_seed(seed: int) -> None:
    """Fix Python / NumPy / PyTorch RNG state."""

    import random
    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def write_results_summary(out_dir: Path, metrics: dict[str, Any]) -> None:
    """Generate a compact human-readable summary markdown."""

    out_dir.mkdir(parents=True, exist_ok=True)
    best = metrics.get("best", {})
    validity = metrics.get("validity", {})
    speaker_majority = metrics.get("speaker_majority_baseline", {})
    speaker_only = metrics.get("speaker_only_baseline", {})
    meta = metrics.get("meta", {})
    best_val = best.get("best_val_summary", {})
    lines = [
        "# Run Summary",
        "",
        f"- task_mode: `{validity.get('task_mode', 'confounded_7way')}`",
        f"- speaker_id: `{validity.get('speaker_id')}`",
        f"- text_policy: `{meta.get('text_policy', 'full')}`",
        f"- scientific_validity: `{validity.get('scientific_validity')}`",
        f"- claim_scope: `{validity.get('claim_scope')}`",
        f"- recommended_interpretation: {validity.get('recommended_interpretation', '')}",
        "",
        "## Best Validation",
        "",
        f"- epoch: `{best.get('epoch')}`",
        f"- accuracy: `{best_val.get('accuracy')}`",
        f"- macro_f1: `{best_val.get('macro_f1')}`",
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
        "- `metrics.json`",
        "- `checkpoints/best.pt`",
        "- `checkpoints/last.pt`",
        "- `inference_val.jsonl`",
        "- `inference_val.metrics.json`",
        "- training curves and confusion matrix PNGs",
    ]
    (out_dir / "results_summary.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
