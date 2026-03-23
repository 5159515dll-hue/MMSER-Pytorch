from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the MMSA-AM comparison audit."""

    p = argparse.ArgumentParser(description="Audit MMSA-AM as a comparison track")
    p.add_argument("--mmsa-am-dir", type=Path, default=Path("MMSA -AM"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/motion_prosody/comparison_mmsa_am"))
    p.add_argument("--pkl", type=Path, default=None, help="Optional explicit MMSA-AM dataset PKL path")
    return p.parse_args()


def _read_text(path: Path) -> str:
    """Read a text file if it exists; return an empty string otherwise."""

    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _norm_text(text: str) -> str:
    """Normalize text for exact duplicate checks."""

    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def _infer_speaker(label_cn: str) -> str:
    """Infer speaker id from the known emotion-to-speaker mapping."""

    label = str(label_cn or "").strip()
    if label in {"愤怒", "恐惧", "惊讶"}:
        return "A"
    if label in {"厌恶", "中性"}:
        return "B"
    if label in {"快乐", "高兴", "开心", "悲伤"}:
        return "C"
    return "UNKNOWN"


def _audit_pkl(path: Path) -> dict[str, Any]:
    """Audit MMSA-AM's serialized dataset split."""

    with path.open("rb") as f:
        data = pickle.load(f)

    split_stats: dict[str, Any] = {}
    ids_by_split: dict[str, set[str]] = {}
    text_by_split: dict[str, set[str]] = {}
    for split in ("train", "valid", "test"):
        block = data.get(split, {})
        labels = list(block.get("label", []))
        ids = [str(x) for x in block.get("id", [])]
        texts = [_norm_text(str(x)) for x in block.get("raw_text", [])]
        speaker_counts = Counter(_infer_speaker(x) for x in labels)
        split_stats[split] = {
            "n": int(len(labels)),
            "label_counts": dict(sorted(Counter(labels).items())),
            "speaker_counts": dict(sorted(speaker_counts.items())),
        }
        ids_by_split[split] = set(ids)
        text_by_split[split] = set(texts)

    overlaps = {
        "id_overlap_train_valid": int(len(ids_by_split["train"] & ids_by_split["valid"])),
        "id_overlap_train_test": int(len(ids_by_split["train"] & ids_by_split["test"])),
        "id_overlap_valid_test": int(len(ids_by_split["valid"] & ids_by_split["test"])),
        "text_overlap_train_valid": int(len(text_by_split["train"] & text_by_split["valid"])),
        "text_overlap_train_test": int(len(text_by_split["train"] & text_by_split["test"])),
        "text_overlap_valid_test": int(len(text_by_split["valid"] & text_by_split["test"])),
    }
    return {"splits": split_stats, "overlaps": overlaps}


def main() -> None:
    """Build a comparison audit for the local MMSA-AM project."""

    args = parse_args()
    root = args.mmsa_am_dir.expanduser()
    out_dir = args.output_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_py = _read_text(root / "train.py")
    dataset_py = _read_text(root / "core" / "dataset.py")
    to_pkl_py = _read_text(root / "core" / "to_pkl.py")
    metric_py = _read_text(root / "core" / "metric.py")
    opts_py = _read_text(root / "opts.py")

    pkl_path = args.pkl.expanduser() if args.pkl is not None else (root / "dataset" / "mongolian" / "mongolian_data_1.pkl")
    pkl_stats = _audit_pkl(pkl_path) if pkl_path.exists() else None

    report = {
        "mmsa_am_dir": str(root),
        "exists": bool(root.exists()),
        "objective": {
            "regression_head": "MSELoss" in train_py or "SmoothL1Loss" in train_py,
            "digitize_to_7way": "np.digitize" in metric_py,
            "label_space_minus3_to_3": "label_map" in dataset_py and "-3" in dataset_py and "3" in dataset_py,
        },
        "training_behavior": {
            "resume_by_default": "Loaded checkpoint for full fine-tuning" in train_py and "test_checkpoint" in opts_py,
            "ema_default_override": '"use_ema": True' in opts_py or '"use_ema": True' in opts_py.replace("'", '"'),
            "tail_guard_default": "enforce_tail_acc" in opts_py and "tail_threshold" in opts_py,
            "huber_or_smoothl1": "SmoothL1Loss" in train_py or '"loss_name": "huber"' in opts_py or "'loss_name': 'huber'" in opts_py,
        },
        "data_protocol": {
            "per_emotion_random_split": "for emotion in ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprise']" in to_pkl_py
            and "random.shuffle(samples)" in to_pkl_py,
            "split_ratio_6_1_3": "train_size = int(total_samples * 0.6)" in to_pkl_py and "valid_size = int(total_samples * 0.1)" in to_pkl_py,
            "speaker_aware_split": False,
            "prompt_group_control": False,
        },
        "modality_protocol": {
            "text_source": "mongolian_only" if "row['蒙文']" in to_pkl_py or 'row["蒙文"]' in to_pkl_py else "unknown",
            "text_masking": False,
            "vision_features": "hog" if "HOGDescriptor" in to_pkl_py else "unknown",
            "audio_features": "mfcc" if "librosa.feature.mfcc" in to_pkl_py else "unknown",
            "text_features": "static_bert_embeddings" if "outputs.last_hidden_state" in to_pkl_py else "unknown",
        },
        "scientific_assessment": {
            "speaker_confound_resolved": False,
            "text_leakage_controlled": False,
            "why_curves_look_normal": [
                "weaker_static_features",
                "regression_objective_then_digitize",
                "ema_and_tail_guard",
                "resume_from_checkpoint_by_default",
            ],
            "recommended_interpretation": (
                "MMSA-AM should be treated as an optimization/style comparison, not as evidence that the confounded 7-way task became scientifically valid."
            ),
        },
        "pkl_audit": pkl_stats,
    }

    json_path = out_dir / "comparison_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# MMSA-AM Comparison Report",
        "",
        f"- exists: `{report['exists']}`",
        f"- speaker_confound_resolved: `{report['scientific_assessment']['speaker_confound_resolved']}`",
        f"- text_leakage_controlled: `{report['scientific_assessment']['text_leakage_controlled']}`",
        "",
        "## Why Its Curves Look More Normal",
        "",
    ]
    for item in report["scientific_assessment"]["why_curves_look_normal"]:
        md_lines.append(f"- {item}")
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- {report['scientific_assessment']['recommended_interpretation']}",
        ]
    )
    if pkl_stats is not None:
        md_lines.extend(
            [
                "",
                "## PKL Audit",
                "",
                f"- train speaker counts: `{pkl_stats['splits']['train']['speaker_counts']}`",
                f"- valid speaker counts: `{pkl_stats['splits']['valid']['speaker_counts']}`",
                f"- test speaker counts: `{pkl_stats['splits']['test']['speaker_counts']}`",
                f"- text overlap train/test: `{pkl_stats['overlaps']['text_overlap_train_test']}`",
            ]
        )
    md_path = out_dir / "comparison_report.md"
    md_path.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    print(f"Wrote comparison report -> {json_path}")
    print(f"Wrote comparison report -> {md_path}")


if __name__ == "__main__":
    main()
