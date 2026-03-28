from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Any

from metrics_utils import holm_bonferroni_adjust, mean_confidence_interval_t, paired_t_test

PAPER_MULTI_SEED = [13, 17, 23, 42, 3407]

GROUP_INVARIANT_KEYS = (
    "protocol_version",
    "manifest_sha256",
    "dataset_kind",
    "task_mode",
    "speaker_id",
    "text_policy",
    "claim_scope",
    "scientific_validity",
    "deterministic_algorithms_enabled",
    "ablation",
    "zero_video",
    "zero_audio",
    "zero_text",
    "use_intensity",
    "video_backbone",
)
COMPARISON_INVARIANT_KEYS = (
    "protocol_version",
    "manifest_sha256",
    "dataset_kind",
    "task_mode",
    "speaker_id",
    "text_policy",
    "claim_scope",
    "scientific_validity",
    "deterministic_algorithms_enabled",
)


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def parse_group_spec(spec: str) -> tuple[str, str]:
    name, sep, pattern = str(spec).partition("=")
    if not sep or not str(name).strip() or not str(pattern).strip():
        raise ValueError(f"invalid group spec: {spec!r}; expected NAME=GLOB")
    return str(name).strip(), str(pattern).strip()


def expand_run_dirs(pattern: str) -> list[Path]:
    matches = sorted(Path(p).expanduser() for p in glob(str(pattern)))
    run_dirs = [path for path in matches if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"no run directories matched pattern: {pattern}")
    return run_dirs


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_inference_metrics_path(run_dir: Path, stem: str) -> Path:
    candidates = [
        run_dir / f"{stem}.metrics.json",
        run_dir / f"{stem}.jsonl.metrics.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing metrics summary for {stem} under {run_dir}")


def _merge_reasons(*reason_groups: Any) -> list[str]:
    merged: list[str] = []
    for group in reason_groups:
        if isinstance(group, list):
            for reason in group:
                text = str(reason).strip()
                if text and text not in merged:
                    merged.append(text)
    return merged


def _contract_or_default(train_metrics: dict[str, Any], test_metrics: dict[str, Any], validity: dict[str, Any]) -> dict[str, Any]:
    contract = train_metrics.get("meta", {}).get("paper_contract", {})
    if not isinstance(contract, dict) or not contract:
        contract = test_metrics.get("paper_contract", {})
    if not isinstance(contract, dict):
        contract = {}
    return {
        "protocol_version": str(
            contract.get("protocol_version")
            or train_metrics.get("paper_grade", {}).get("protocol_version")
            or test_metrics.get("paper_grade", {}).get("protocol_version")
            or ""
        ),
        "manifest_sha256": str(contract.get("manifest_sha256") or test_metrics.get("manifest_sha256") or ""),
        "dataset_kind": str(contract.get("dataset_kind") or test_metrics.get("dataset_kind") or ""),
        "task_mode": str(contract.get("task_mode") or test_metrics.get("task_mode") or ""),
        "speaker_id": contract.get("speaker_id", test_metrics.get("speaker_id")),
        "text_policy": str(contract.get("text_policy") or test_metrics.get("text_policy") or ""),
        "claim_scope": str(contract.get("claim_scope") or validity.get("claim_scope") or ""),
        "scientific_validity": bool(
            contract.get("scientific_validity", validity.get("scientific_validity", False))
        ),
        "deterministic_algorithms_enabled": train_metrics.get("meta", {})
        .get("deterministic_policy", {})
        .get("deterministic_algorithms_enabled"),
        "ablation": str(contract.get("ablation") or test_metrics.get("ablation") or ""),
        "zero_video": bool(contract.get("zero_video", test_metrics.get("zero_video", False))),
        "zero_audio": bool(contract.get("zero_audio", test_metrics.get("zero_audio", False))),
        "zero_text": bool(contract.get("zero_text", test_metrics.get("zero_text", False))),
        "use_intensity": bool(contract.get("use_intensity", test_metrics.get("intensity_enabled", False))),
        "video_backbone": str(contract.get("video_backbone") or ""),
        "label_names": list(contract.get("label_names", test_metrics.get("label_names", []))),
    }


def load_run_bundle(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser()
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing metrics.json under {run_dir}")
    train_metrics = _load_json(metrics_path)
    val_metrics = _load_json(_resolve_inference_metrics_path(run_dir, "inference_val"))
    test_metrics = _load_json(_resolve_inference_metrics_path(run_dir, "inference_test"))
    seed = int(train_metrics.get("meta", {}).get("args", {}).get("seed", 0))
    best = train_metrics.get("best", {})
    stop = train_metrics.get("stop", {})
    validity = train_metrics.get("validity", {})
    train_paper = train_metrics.get("paper_grade", {})
    val_paper = val_metrics.get("paper_grade", {})
    test_paper = test_metrics.get("paper_grade", {})
    contract = _contract_or_default(train_metrics, test_metrics, validity if isinstance(validity, dict) else {})
    reasons = _merge_reasons(
        train_paper.get("ineligibility_reasons", []) if isinstance(train_paper, dict) else [],
        val_paper.get("ineligibility_reasons", []) if isinstance(val_paper, dict) else [],
        test_paper.get("ineligibility_reasons", []) if isinstance(test_paper, dict) else [],
    )
    return {
        "run_dir": str(run_dir),
        "seed": int(seed),
        "selected_epoch": int(best.get("epoch", 0) or 0),
        "selected_val_f1": float(best.get("best_f1", 0.0) or 0.0),
        "selected_val_acc": float(best.get("best_acc", 0.0) or 0.0),
        "selected_val_loss": float(best.get("best_val_loss", 0.0) or 0.0),
        "significant_best_monitor_value": float(
            best.get("significant_best_monitor_value", best.get("best_monitor_value", 0.0)) or 0.0
        ),
        "checkpoint_reason": str(best.get("checkpoint_reason", "") or ""),
        "stop_reason": str(stop.get("reason", "") or ""),
        "stop_epoch": int(stop.get("epoch", 0) or 0),
        "lr_drop_epochs": [int(x) for x in stop.get("lr_drop_epochs", [])],
        "test_macro_f1": float(test_metrics.get("macro_f1_on_ok", 0.0) or 0.0),
        "test_accuracy": float(test_metrics.get("accuracy_on_ok", 0.0) or 0.0),
        "val_macro_f1_reported": float(val_metrics.get("macro_f1_on_ok", 0.0) or 0.0),
        "val_accuracy_reported": float(val_metrics.get("accuracy_on_ok", 0.0) or 0.0),
        "train_metrics_path": str(metrics_path),
        "val_metrics_path": str(_resolve_inference_metrics_path(run_dir, "inference_val")),
        "test_metrics_path": str(_resolve_inference_metrics_path(run_dir, "inference_test")),
        "val_checkpoint": str(val_metrics.get("checkpoint", "") or ""),
        "test_checkpoint": str(test_metrics.get("checkpoint", "") or ""),
        "paper_grade_eligible": bool(train_paper.get("eligible", False))
        and bool(val_paper.get("eligible", False))
        and bool(test_paper.get("eligible", False)),
        "paper_grade_reasons": reasons,
        "run_status": str(train_metrics.get("run_status", "") or ""),
        "contract": contract,
    }


def _validate_run_bundle(run: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not bool(run.get("paper_grade_eligible", False)):
        reasons.append("paper_grade_ineligible")
    if Path(str(run.get("val_checkpoint", ""))).name != "best.pt":
        reasons.append("val_checkpoint_not_best")
    if Path(str(run.get("test_checkpoint", ""))).name != "best.pt":
        reasons.append("test_checkpoint_not_best")
    for key in ("selected_val_f1", "selected_val_acc", "selected_val_loss", "test_macro_f1", "test_accuracy"):
        value = float(run.get(key, 0.0) or 0.0)
        if not math.isfinite(value):
            reasons.append(f"non_finite_{key}")
    contract = run.get("contract", {})
    for key in ("protocol_version", "manifest_sha256", "dataset_kind", "task_mode", "text_policy", "claim_scope"):
        if not str(contract.get(key, "")).strip():
            reasons.append(f"missing_contract_{key}")
    return _merge_reasons(reasons, run.get("paper_grade_reasons", []))


def _build_compatibility_report(runs: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    by_key: dict[str, list[Any]] = defaultdict(list)
    mismatch_keys: list[str] = []
    reference: dict[str, Any] = {}
    for key in keys:
        values = [run.get("contract", {}).get(key) for run in runs]
        by_key[key] = values
        distinct = []
        for value in values:
            if value not in distinct:
                distinct.append(value)
        if distinct:
            reference[key] = distinct[0]
        if len(distinct) > 1:
            mismatch_keys.append(str(key))
    return {
        "reference": reference,
        "values_by_key": dict(by_key),
        "mismatch_keys": mismatch_keys,
    }


def summarize_experiment_group(
    name: str,
    run_dirs: list[Path],
    confidence: float = 0.95,
    *,
    expected_seeds: list[int] | None = None,
) -> dict[str, Any]:
    expected = [int(seed) for seed in (expected_seeds or PAPER_MULTI_SEED)]
    loaded = [load_run_bundle(run_dir) for run_dir in run_dirs]
    loaded.sort(key=lambda item: (int(item["seed"]), str(item["run_dir"])))
    valid_runs: list[dict[str, Any]] = []
    rejected_runs: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    duplicate_seeds: list[int] = []
    for run in loaded:
        seed = int(run["seed"])
        reasons = _validate_run_bundle(run)
        if seed in seen_seeds:
            duplicate_seeds.append(seed)
            reasons = _merge_reasons(reasons, ["duplicate_seed"])
        seen_seeds.add(seed)
        if reasons:
            rejected_runs.append({"run_dir": run["run_dir"], "seed": seed, "reasons": reasons})
        else:
            valid_runs.append(run)
    seeds = [int(run["seed"]) for run in valid_runs]
    missing_seeds = sorted(set(expected) - set(seeds))
    extra_seeds = sorted(set(seeds) - set(expected))
    compatibility_report = _build_compatibility_report(valid_runs, GROUP_INVARIANT_KEYS)
    macro_values = [float(run["test_macro_f1"]) for run in valid_runs]
    acc_values = [float(run["test_accuracy"]) for run in valid_runs]
    epoch_values = [float(run["selected_epoch"]) for run in valid_runs]
    stop_reason_counts = Counter(str(run["stop_reason"] or "unknown") for run in valid_runs)
    checkpoint_reason_counts = Counter(str(run["checkpoint_reason"] or "unknown") for run in valid_runs)
    paper_grade_ready = (
        len(rejected_runs) == 0
        and len(missing_seeds) == 0
        and len(extra_seeds) == 0
        and len(duplicate_seeds) == 0
        and len(compatibility_report["mismatch_keys"]) == 0
    )
    return {
        "name": str(name),
        "expected_seeds": expected,
        "seeds": seeds,
        "missing_seeds": missing_seeds,
        "extra_seeds": extra_seeds,
        "duplicate_seeds": sorted(set(int(seed) for seed in duplicate_seeds)),
        "run_count": int(len(valid_runs)),
        "runs": valid_runs,
        "rejected_runs": rejected_runs,
        "paper_grade_ready": bool(paper_grade_ready),
        "compatibility_report": compatibility_report,
        "test_macro_f1": mean_confidence_interval_t(macro_values, confidence=confidence),
        "test_accuracy": mean_confidence_interval_t(acc_values, confidence=confidence),
        "selected_epoch": mean_confidence_interval_t(epoch_values, confidence=confidence),
        "stop_reason_counts": dict(sorted(stop_reason_counts.items())),
        "checkpoint_reason_counts": dict(sorted(checkpoint_reason_counts.items())),
    }


def ensure_group_is_paper_grade_ready(group: dict[str, Any]) -> None:
    if bool(group.get("paper_grade_ready", False)):
        return
    issues: list[str] = []
    if group.get("missing_seeds"):
        issues.append(f"missing_seeds={group['missing_seeds']}")
    if group.get("extra_seeds"):
        issues.append(f"extra_seeds={group['extra_seeds']}")
    if group.get("duplicate_seeds"):
        issues.append(f"duplicate_seeds={group['duplicate_seeds']}")
    if group.get("rejected_runs"):
        issues.append(f"rejected_runs={len(group['rejected_runs'])}")
    mismatch_keys = group.get("compatibility_report", {}).get("mismatch_keys", [])
    if mismatch_keys:
        issues.append(f"group_mismatch_keys={mismatch_keys}")
    raise RuntimeError(f"group {group.get('name')} is not paper-grade ready: " + ", ".join(issues))


def build_pairwise_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    confidence: float = 0.95,
    comparison_scope: str = "confirmatory",
) -> dict[str, Any]:
    ensure_group_is_paper_grade_ready(baseline)
    ensure_group_is_paper_grade_ready(candidate)
    baseline_by_seed = {int(run["seed"]): run for run in baseline.get("runs", [])}
    candidate_by_seed = {int(run["seed"]): run for run in candidate.get("runs", [])}
    baseline_seeds = sorted(baseline_by_seed)
    candidate_seeds = sorted(candidate_by_seed)
    if baseline_seeds != candidate_seeds:
        raise RuntimeError(
            f"paired comparison requires identical seeds, got baseline={baseline_seeds}, candidate={candidate_seeds}"
        )
    invariant_mismatches: list[str] = []
    baseline_ref = baseline.get("compatibility_report", {}).get("reference", {})
    candidate_ref = candidate.get("compatibility_report", {}).get("reference", {})
    for key in COMPARISON_INVARIANT_KEYS:
        if baseline_ref.get(key) != candidate_ref.get(key):
            invariant_mismatches.append(str(key))
    if invariant_mismatches:
        raise RuntimeError(
            f"comparison invariants mismatch for {candidate.get('name')} vs {baseline.get('name')}: {invariant_mismatches}"
        )
    shared_seeds = baseline_seeds
    baseline_macro = [float(baseline_by_seed[seed]["test_macro_f1"]) for seed in shared_seeds]
    candidate_macro = [float(candidate_by_seed[seed]["test_macro_f1"]) for seed in shared_seeds]
    baseline_acc = [float(baseline_by_seed[seed]["test_accuracy"]) for seed in shared_seeds]
    candidate_acc = [float(candidate_by_seed[seed]["test_accuracy"]) for seed in shared_seeds]
    macro_stats = paired_t_test(baseline_macro, candidate_macro, confidence=confidence)
    acc_stats = paired_t_test(baseline_acc, candidate_acc, confidence=confidence)
    raw_p_value = _float_or_default(macro_stats.get("p_value"), 1.0)
    macro_stats["adjusted_p_value"] = raw_p_value
    acc_stats["adjusted_p_value"] = _float_or_default(acc_stats.get("p_value"), 1.0)
    paired_n = int(macro_stats.get("n", 0) or 0)
    required_positive = int(math.ceil(0.8 * paired_n)) if paired_n > 0 else 0
    significant = bool(
        paired_n > 0
        and _float_or_default(macro_stats.get("mean_diff"), 0.0) > 0.0
        and _float_or_default(macro_stats.get("adjusted_p_value"), 1.0) < 0.05
        and int(macro_stats.get("positive_gain_count", 0) or 0) >= required_positive
    )
    return {
        "baseline": str(baseline.get("name", "")),
        "candidate": str(candidate.get("name", "")),
        "paired_seeds": shared_seeds,
        "paired_run_count": paired_n,
        "comparison_scope": str(comparison_scope),
        "multiple_comparison_method": "none",
        "macro_f1": macro_stats,
        "accuracy": acc_stats,
        "required_positive_gain_count": required_positive,
        "claim_supported": significant,
    }


def apply_multiple_comparison_correction(
    comparisons: list[dict[str, Any]],
    *,
    method: str = "holm_bonferroni",
) -> list[dict[str, Any]]:
    if not comparisons:
        return comparisons
    if method != "holm_bonferroni":
        raise ValueError(f"unsupported multiple comparison correction: {method}")
    macro_p = [comparison["macro_f1"].get("p_value") for comparison in comparisons]
    acc_p = [comparison["accuracy"].get("p_value") for comparison in comparisons]
    macro_adj = holm_bonferroni_adjust(macro_p)
    acc_adj = holm_bonferroni_adjust(acc_p)
    for idx, comparison in enumerate(comparisons):
        comparison["comparison_scope"] = "exploratory"
        comparison["multiple_comparison_method"] = method
        comparison["macro_f1"]["adjusted_p_value"] = macro_adj[idx]
        comparison["accuracy"]["adjusted_p_value"] = acc_adj[idx]
        macro = comparison["macro_f1"]
        comparison["claim_supported"] = bool(
            int(macro.get("n", 0) or 0) > 0
            and _float_or_default(macro.get("mean_diff"), 0.0) > 0.0
            and _float_or_default(macro.get("adjusted_p_value"), 1.0) < 0.05
            and int(macro.get("positive_gain_count", 0) or 0) >= int(comparison.get("required_positive_gain_count", 0) or 0)
        )
    return comparisons


def render_markdown_report(
    groups: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    *,
    confidence: float = 0.95,
) -> str:
    lines = [
        "# Multi-Seed Experiment Summary",
        "",
        f"- confidence: `{confidence:.2f}`",
        f"- expected_seeds: `{PAPER_MULTI_SEED}`",
        "",
    ]
    for group in groups:
        macro = group["test_macro_f1"]
        acc = group["test_accuracy"]
        contract = group.get("compatibility_report", {}).get("reference", {})
        lines.extend(
            [
                f"## {group['name']}",
                "",
                f"- paper_grade_ready: `{group['paper_grade_ready']}`",
                f"- protocol_version: `{contract.get('protocol_version')}`",
                f"- seeds: `{group['seeds']}`",
                f"- missing_seeds: `{group['missing_seeds']}`",
                f"- rejected_runs: `{len(group['rejected_runs'])}`",
                f"- claim_scope: `{contract.get('claim_scope')}`",
                f"- scientific_validity: `{contract.get('scientific_validity')}`",
                f"- test macro_f1: `{macro['mean']:.4f} ± {macro['std']:.4f}`",
                f"- test macro_f1 95% CI: `[{macro['low']:.4f}, {macro['high']:.4f}]`" if macro["low"] is not None and macro["high"] is not None else "- test macro_f1 95% CI: `n/a`",
                f"- test accuracy: `{acc['mean']:.4f} ± {acc['std']:.4f}`",
                f"- test accuracy 95% CI: `[{acc['low']:.4f}, {acc['high']:.4f}]`" if acc["low"] is not None and acc["high"] is not None else "- test accuracy 95% CI: `n/a`",
                f"- checkpoint reasons: `{group['checkpoint_reason_counts']}`",
                f"- stop reasons: `{group['stop_reason_counts']}`",
                "",
            ]
        )
    if pairwise:
        lines.extend(["# Pairwise Comparisons", ""])
    for comparison in pairwise:
        macro = comparison["macro_f1"]
        lines.extend(
            [
                f"## {comparison['candidate']} vs {comparison['baseline']}",
                "",
                f"- comparison_scope: `{comparison['comparison_scope']}`",
                f"- multiple_comparison_method: `{comparison['multiple_comparison_method']}`",
                f"- paired seeds: `{comparison['paired_seeds']}`",
                f"- delta macro_f1: `{macro['mean_diff']:.4f}`",
                f"- delta macro_f1 95% CI: `[{macro['ci_low']:.4f}, {macro['ci_high']:.4f}]`" if macro["ci_low"] is not None and macro["ci_high"] is not None else "- delta macro_f1 95% CI: `n/a`",
                f"- raw p_value: `{macro['p_value']}`",
                f"- adjusted p_value: `{macro['adjusted_p_value']}`",
                f"- positive gain count: `{macro['positive_gain_count']}/{comparison['paired_run_count']}`",
                f"- claim_supported: `{comparison['claim_supported']}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"
