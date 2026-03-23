from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from manifest_utils import (
    build_manifest_from_split_items,
    filter_manifest_items_for_task,
    load_split_manifest,
    resolve_grouped_cv_splits,
    resolve_task_label_names,
)


def parse_args() -> argparse.Namespace:
    """Parse scientific suite CLI arguments."""

    p = argparse.ArgumentParser(description="Run the fixed-dataset scientific suite")
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--cached-dataset", type=Path, default=None)
    p.add_argument("--feature-cache", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--speakers", type=str, default="A,B,C")
    p.add_argument("--text-policies", type=str, default="drop,mask_emotion_cues")
    p.add_argument("--ablations", type=str, default="full,no-audio,no-video")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=3)
    p.add_argument("--group-key", type=str, default="prompt_group_id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-auxiliary-7way", action="store_true")
    p.add_argument("--mmsa-am-dir", type=Path, default=Path("MMSA -AM"))
    p.add_argument("--skip-comparison-report", action="store_true")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=str, default="16")
    p.add_argument("--num-workers", type=str, default="0")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp-mode", type=str, default="auto")
    p.add_argument("--video-backbone", type=str, default="dual")
    p.add_argument("--video-model", type=str, default="MCG-NJU/videomae-large")
    p.add_argument("--audio-model", type=str, default="microsoft/wavlm-large")
    p.add_argument("--audio-model-revision", type=str, default="")
    p.add_argument("--text-model", type=str, default="xlm-roberta-large")
    p.add_argument("--fusion-mode", type=str, default="gated_text")
    p.add_argument("--freeze-audio", action="store_true")
    p.add_argument("--freeze-rgb", action="store_true")
    p.add_argument("--freeze-flow", action="store_true")
    p.add_argument("--freeze-text", action="store_true")
    p.add_argument("--monitor", type=str, default="val_f1", choices=["val_acc", "val_f1"])
    p.add_argument("--early-stop-patience", type=int, default=4)
    return p.parse_args()


def _write_json(path: Path, obj: Any) -> None:
    """Write JSON with UTF-8 and indentation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return mean/std with stable 0-std handling."""

    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def _base_train_cmd(args: argparse.Namespace) -> list[str]:
    """Build the common train.py CLI prefix."""

    cmd = [sys.executable, "train.py"]
    if args.feature_cache is not None:
        cmd.extend(["--feature-cache", str(args.feature_cache.expanduser())])
    elif args.cached_dataset is not None:
        cmd.extend(["--cached-dataset", str(args.cached_dataset.expanduser())])
    else:
        raise RuntimeError("Provide at least one of --feature-cache or --cached-dataset")
    cmd.extend(
        [
            "--epochs",
            str(args.epochs),
            "--device",
            str(args.device),
            "--amp-mode",
            str(args.amp_mode),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--video-backbone",
            str(args.video_backbone),
            "--video-model",
            str(args.video_model),
            "--audio-model",
            str(args.audio_model),
            "--text-model",
            str(args.text_model),
            "--fusion-mode",
            str(args.fusion_mode),
            "--monitor",
            str(args.monitor),
            "--early-stop-patience",
            str(args.early_stop_patience),
        ]
    )
    if str(args.audio_model_revision).strip():
        cmd.extend(["--audio-model-revision", str(args.audio_model_revision).strip()])
    if bool(args.freeze_audio):
        cmd.append("--freeze-audio")
    if bool(args.freeze_rgb):
        cmd.append("--freeze-rgb")
    if bool(args.freeze_flow):
        cmd.append("--freeze-flow")
    if bool(args.freeze_text):
        cmd.append("--freeze-text")
    return cmd


def _run_one_train(
    *,
    base_cmd: list[str],
    manifest_path: Path,
    run_dir: Path,
    task_mode: str,
    speaker_id: str | None,
    text_policy: str,
    ablation: str,
    benchmark_tag: str,
    cwd: Path,
) -> dict[str, Any]:
    """Run one train.py invocation and return parsed metrics."""

    cmd = list(base_cmd)
    cmd.extend(
        [
            "--split-manifest",
            str(manifest_path),
            "--output-dir",
            str(run_dir),
            "--task-mode",
            str(task_mode),
            "--text-policy",
            str(text_policy),
            "--ablation",
            str(ablation),
            "--benchmark-tag",
            str(benchmark_tag),
        ]
    )
    if speaker_id:
        cmd.extend(["--speaker-id", str(speaker_id)])
    subprocess.run(cmd, cwd=str(cwd), check=True)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Expected metrics.json not found after run: {run_dir}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _aggregate_runs(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate fold results by speaker/text policy/ablation."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = (str(row["speaker_id"]), str(row["text_policy"]), str(row["ablation"]))
        grouped.setdefault(key, []).append(row)

    per_setting: list[dict[str, Any]] = []
    for (speaker_id, text_policy, ablation), rows in sorted(grouped.items()):
        f1s = [float(r["best_f1"]) for r in rows]
        accs = [float(r["best_acc"]) for r in rows]
        mean_f1, std_f1 = _mean_std(f1s)
        mean_acc, std_acc = _mean_std(accs)
        per_setting.append(
            {
                "speaker_id": speaker_id,
                "text_policy": text_policy,
                "ablation": ablation,
                "num_runs": len(rows),
                "mean_macro_f1": mean_f1,
                "std_macro_f1": std_f1,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "scientific_validity": bool(rows[0].get("scientific_validity", False)),
            }
        )

    cross_speaker: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_setting:
        key = (str(row["text_policy"]), str(row["ablation"]))
        cross_speaker.setdefault(key, []).append(row)

    cross_speaker_summary: list[dict[str, Any]] = []
    for (text_policy, ablation), rows in sorted(cross_speaker.items()):
        mean_macro_f1, std_macro_f1 = _mean_std([float(r["mean_macro_f1"]) for r in rows])
        mean_accuracy, std_accuracy = _mean_std([float(r["mean_accuracy"]) for r in rows])
        cross_speaker_summary.append(
            {
                "text_policy": text_policy,
                "ablation": ablation,
                "num_speakers": len(rows),
                "speaker_mean_macro_f1": mean_macro_f1,
                "speaker_std_macro_f1": std_macro_f1,
                "speaker_mean_accuracy": mean_accuracy,
                "speaker_std_accuracy": std_accuracy,
            }
        )

    primary = next((row for row in cross_speaker_summary if row["text_policy"] == "drop" and row["ablation"] == "full"), None)
    secondary = next(
        (row for row in cross_speaker_summary if row["text_policy"] == "mask_emotion_cues" and row["ablation"] == "full"),
        None,
    )
    return {
        "per_setting": per_setting,
        "cross_speaker_summary": cross_speaker_summary,
        "primary": primary,
        "secondary": secondary,
    }


def main() -> None:
    """Run grouped-CV within-speaker scientific experiments and aggregate outputs."""

    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    out_dir = args.output_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = load_split_manifest(args.split_manifest.expanduser())
    speakers = [x.strip().upper() for x in str(args.speakers).split(",") if x.strip()]
    text_policies = [x.strip() for x in str(args.text_policies).split(",") if x.strip()]
    ablations = [x.strip() for x in str(args.ablations).split(",") if x.strip()]
    base_cmd = _base_train_cmd(args)

    run_rows: list[dict[str, Any]] = []
    manifest_dir = out_dir / "generated_manifests"

    for speaker_id in speakers:
        speaker_items = filter_manifest_items_for_task(base_manifest.get("items", []), "within_speaker", speaker_id)
        label_names = resolve_task_label_names("within_speaker", speaker_id)
        for repeat_idx in range(int(args.cv_repeats)):
            folds, actual_splits = resolve_grouped_cv_splits(
                speaker_items,
                label_names=label_names,
                group_key=str(args.group_key),
                requested_splits=int(args.cv_folds),
                seed=int(args.seed) + (repeat_idx * 1009),
            )
            for fold_idx, (train_items, val_items) in enumerate(folds, start=1):
                fold_manifest = build_manifest_from_split_items(
                    base_manifest,
                    train_items=train_items,
                    val_items=val_items,
                    split_strategy=f"grouped_cv:{args.group_key}",
                    extra_meta={
                        "task_mode": "within_speaker",
                        "speaker_id": speaker_id,
                        "repeat": repeat_idx + 1,
                        "fold": fold_idx,
                        "requested_splits": int(args.cv_folds),
                        "actual_splits": int(actual_splits),
                        "group_key": str(args.group_key),
                    },
                )
                manifest_path = manifest_dir / f"within_{speaker_id}" / f"repeat_{repeat_idx + 1}_fold_{fold_idx}.json"
                _write_json(manifest_path, fold_manifest)

                for text_policy in text_policies:
                    for ablation in ablations:
                        run_dir = (
                            out_dir
                            / "scientific"
                            / f"speaker_{speaker_id}"
                            / f"text_{text_policy}"
                            / f"ablation_{ablation}"
                            / f"repeat_{repeat_idx + 1}_fold_{fold_idx}"
                        )
                        benchmark_tag = f"scientific_within_{speaker_id}_r{repeat_idx + 1}_f{fold_idx}_{text_policy}_{ablation}"
                        metrics = _run_one_train(
                            base_cmd=base_cmd,
                            manifest_path=manifest_path,
                            run_dir=run_dir,
                            task_mode="within_speaker",
                            speaker_id=speaker_id,
                            text_policy=text_policy,
                            ablation=ablation,
                            benchmark_tag=benchmark_tag,
                            cwd=repo_root,
                        )
                        best = metrics.get("best", {})
                        validity = metrics.get("validity", {})
                        run_rows.append(
                            {
                                "speaker_id": speaker_id,
                                "text_policy": text_policy,
                                "ablation": ablation,
                                "repeat": repeat_idx + 1,
                                "fold": fold_idx,
                                "manifest_path": str(manifest_path),
                                "run_dir": str(run_dir),
                                "best_f1": float(best.get("best_f1", 0.0)),
                                "best_acc": float(best.get("best_acc", 0.0)),
                                "scientific_validity": bool(validity.get("scientific_validity", False)),
                            }
                        )

    auxiliary_report = None
    if bool(args.include_auxiliary_7way):
        aux_manifest_path = args.split_manifest.expanduser()
        aux_run_dir = out_dir / "auxiliary" / "confounded_7way_full_text"
        aux_metrics = _run_one_train(
            base_cmd=base_cmd,
            manifest_path=aux_manifest_path,
            run_dir=aux_run_dir,
            task_mode="confounded_7way",
            speaker_id=None,
            text_policy="full",
            ablation="full",
            benchmark_tag="aux_confounded_7way_full_text",
            cwd=repo_root,
        )
        auxiliary_report = {
            "run_dir": str(aux_run_dir),
            "best_f1": float(aux_metrics.get("best", {}).get("best_f1", 0.0)),
            "best_acc": float(aux_metrics.get("best", {}).get("best_acc", 0.0)),
            "scientific_validity": bool(aux_metrics.get("validity", {}).get("scientific_validity", False)),
        }

    comparison_report = None
    if not bool(args.skip_comparison_report) and args.mmsa_am_dir.expanduser().exists():
        subprocess.run(
            [
                sys.executable,
                "compare_mmsa_am.py",
                "--mmsa-am-dir",
                str(args.mmsa_am_dir.expanduser()),
                "--output-dir",
                str(out_dir / "comparison_mmsa_am"),
            ],
            cwd=str(repo_root),
            check=True,
        )
        comparison_report = str(out_dir / "comparison_mmsa_am" / "comparison_report.json")

    aggregated = _aggregate_runs(run_rows)
    suite_summary = {
        "suite": {
            "split_manifest": str(args.split_manifest.expanduser()),
            "feature_cache": str(args.feature_cache.expanduser()) if args.feature_cache is not None else None,
            "cached_dataset": str(args.cached_dataset.expanduser()) if args.cached_dataset is not None else None,
            "speakers": speakers,
            "text_policies": text_policies,
            "ablations": ablations,
            "cv_folds": int(args.cv_folds),
            "cv_repeats": int(args.cv_repeats),
            "group_key": str(args.group_key),
            "seed": int(args.seed),
        },
        "runs": run_rows,
        "aggregated": aggregated,
        "auxiliary_confounded_7way": auxiliary_report,
        "comparison_report": comparison_report,
    }
    _write_json(out_dir / "scientific_summary.json", suite_summary)

    md_lines = [
        "# Scientific Suite Summary",
        "",
        f"- split_manifest: `{args.split_manifest.expanduser()}`",
        f"- group_key: `{args.group_key}`",
        f"- cv: `{args.cv_repeats} repeats x up to {args.cv_folds} folds`",
        "",
        "## Primary",
        "",
        f"- {aggregated.get('primary')}",
        "",
        "## Secondary",
        "",
        f"- {aggregated.get('secondary')}",
        "",
        "## Cross-Speaker Aggregates",
        "",
    ]
    for row in aggregated.get("cross_speaker_summary", []):
        md_lines.append(f"- {row}")
    if auxiliary_report is not None:
        md_lines.extend(["", "## Auxiliary 7-Way", "", f"- {auxiliary_report}"])
    if comparison_report is not None:
        md_lines.extend(["", "## Comparison", "", f"- comparison_report: `{comparison_report}`"])
    (out_dir / "scientific_summary.md").write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    print(f"Wrote suite summary -> {out_dir / 'scientific_summary.json'}")
    print(f"Wrote suite summary -> {out_dir / 'scientific_summary.md'}")


if __name__ == "__main__":
    main()
