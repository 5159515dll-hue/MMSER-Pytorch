from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from run_store import RUN_STORE_SCHEMA_VERSION, atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate a legacy flat training run directory into the manifest-driven run_store layout.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempt-id", type=str, default="legacy_import")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.exists() and not bool(args.force):
        raise RuntimeError(f"run_manifest.json already exists under {run_dir}; use --force to overwrite migration metadata.")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Legacy metrics.json not found under {run_dir}")
    train_metrics = _load_json(metrics_path)
    benchmark_tag = str(train_metrics.get("meta", {}).get("benchmark_tag", run_dir.name) or run_dir.name)
    attempt_id = str(args.attempt_id).strip() or "legacy_import"
    attempt_dir = run_dir / "attempts" / attempt_id
    bundle_dir = attempt_dir / "bundles" / "best_epoch_legacy"
    published_dir = attempt_dir / "published"
    (attempt_dir / "state" / "epochs").mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (published_dir / "plots").mkdir(parents=True, exist_ok=True)

    copied = {
        "best_checkpoint": _copy_if_exists(run_dir / "checkpoints" / "best.pt", bundle_dir / "checkpoint.pt"),
        "last_checkpoint": _copy_if_exists(run_dir / "checkpoints" / "last.pt", published_dir / "last.pt"),
        "metrics": _copy_if_exists(metrics_path, published_dir / "metrics.json"),
        "summary": _copy_if_exists(run_dir / "results_summary.md", published_dir / "results_summary.md"),
        "val_jsonl": _copy_if_exists(run_dir / "inference_val.jsonl", bundle_dir / "inference_val.jsonl"),
        "val_metrics": _copy_if_exists(run_dir / "inference_val.metrics.json", bundle_dir / "inference_val.metrics.json"),
        "test_jsonl": _copy_if_exists(run_dir / "inference_test.jsonl", published_dir / "inference_test.jsonl"),
        "test_metrics": _copy_if_exists(run_dir / "inference_test.metrics.json", published_dir / "inference_test.metrics.json"),
    }
    for name in (
        "accuracy_curve.png",
        "loss_curve.png",
        "macro_f1_curve.png",
        "lr_curve.png",
        "per_class_recall_curve.png",
        "confusion_matrix_best_val.png",
    ):
        _copy_if_exists(run_dir / name, published_dir / "plots" / name)

    bundle_manifest = {
        "schema_version": RUN_STORE_SCHEMA_VERSION,
        "bundle_id": "best_epoch_legacy",
        "attempt_id": attempt_id,
        "epoch": int(train_metrics.get("best", {}).get("epoch", 0) or 0),
        "selection_meta": {
            "checkpoint_reason": str(train_metrics.get("best", {}).get("checkpoint_reason", "") or ""),
            "migrated_from_legacy": True,
        },
    }
    atomic_write_json(bundle_manifest, bundle_dir / "bundle_manifest.json")

    attempt_manifest = {
        "schema_version": RUN_STORE_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "attempt_relpath": str(attempt_dir.relative_to(run_dir)),
        "run_dir": str(run_dir),
        "status": "completed",
        "benchmark_tag": benchmark_tag,
        "seed": train_metrics.get("meta", {}).get("args", {}).get("seed"),
        "created_at": None,
        "updated_at": None,
        "started_at": None,
        "finished_at": None,
        "args": train_metrics.get("meta", {}).get("args", {}),
        "run_contract": train_metrics.get("meta", {}).get("paper_contract", {}),
        "provenance": train_metrics.get("provenance", {}),
        "validity": train_metrics.get("validity", {}),
        "input_cache_contract": train_metrics.get("meta", {}).get("input_cache_contract", {}) or {},
        "deterministic_policy": train_metrics.get("meta", {}).get("deterministic_policy", {}) or {},
        "current_epoch": int(train_metrics.get("stop", {}).get("epoch", 0) or 0),
        "epochs_completed": [],
        "best_epoch": int(train_metrics.get("best", {}).get("epoch", 0) or 0),
        "best_bundle_relpath": str(bundle_dir.relative_to(attempt_dir)) if copied["best_checkpoint"] else None,
        "published_last_checkpoint_relpath": str((published_dir / "last.pt").relative_to(attempt_dir)) if copied["last_checkpoint"] else None,
        "published_metrics_relpath": str((published_dir / "metrics.json").relative_to(attempt_dir)) if copied["metrics"] else None,
        "published_results_summary_relpath": str((published_dir / "results_summary.md").relative_to(attempt_dir)) if copied["summary"] else None,
        "published_inference_outputs": {
            **({"val": str((bundle_dir / "inference_val.jsonl").relative_to(attempt_dir))} if copied["val_jsonl"] else {}),
            **({"test": str((published_dir / "inference_test.jsonl").relative_to(attempt_dir))} if copied["test_jsonl"] else {}),
        },
        "published_inference_metrics": {
            **({"val": str((bundle_dir / "inference_val.metrics.json").relative_to(attempt_dir))} if copied["val_metrics"] else {}),
            **({"test": str((published_dir / "inference_test.metrics.json").relative_to(attempt_dir))} if copied["test_metrics"] else {}),
        },
        "run_status": str(train_metrics.get("run_status", "completed") or "completed"),
        "stop_reason": str(train_metrics.get("stop", {}).get("reason", "") or ""),
        "failure": None,
    }
    atomic_write_json(attempt_manifest, attempt_dir / "attempt_manifest.json")

    run_manifest = {
        "schema_version": RUN_STORE_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "created_at": None,
        "updated_at": None,
        "benchmark_tag": benchmark_tag,
        "published_attempt_id": attempt_id,
        "active_attempt_id": None,
        "attempts": [
            {
                "attempt_id": attempt_id,
                "attempt_relpath": str(attempt_dir.relative_to(run_dir)),
                "seed": train_metrics.get("meta", {}).get("args", {}).get("seed"),
                "status": "completed",
                "started_at": None,
                "finished_at": None,
                "best_epoch": int(train_metrics.get("best", {}).get("epoch", 0) or 0),
                "run_status": str(train_metrics.get("run_status", "completed") or "completed"),
            }
        ],
    }
    atomic_write_json(run_manifest, run_manifest_path)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "attempt_id": attempt_id,
                "copied": copied,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
