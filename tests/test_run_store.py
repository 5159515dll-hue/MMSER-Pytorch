from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_store import RUN_STORE_SCHEMA_VERSION, register_published_inference_output, validate_run_dir


class RunStoreTests(unittest.TestCase):
    def test_validate_run_dir_reports_clean_published_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            attempt_dir = run_dir / "attempts" / "attempt_seed13"
            bundle_dir = attempt_dir / "bundles" / "best_epoch_0001"
            published_dir = attempt_dir / "published"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            published_dir.mkdir(parents=True, exist_ok=True)
            (bundle_dir / "bundle_manifest.json").write_text("{}", encoding="utf-8")
            (bundle_dir / "checkpoint.pt").write_bytes(b"ckpt")
            (bundle_dir / "inference_val.jsonl").write_text("", encoding="utf-8")
            (bundle_dir / "inference_val.metrics.json").write_text("{}", encoding="utf-8")
            (published_dir / "metrics.json").write_text("{}", encoding="utf-8")
            attempt_manifest = {
                "schema_version": RUN_STORE_SCHEMA_VERSION,
                "attempt_id": "attempt_seed13",
                "status": "completed",
                "best_bundle_relpath": str(bundle_dir.relative_to(attempt_dir)),
                "published_metrics_relpath": str((published_dir / "metrics.json").relative_to(attempt_dir)),
                "published_inference_metrics": {
                    "val": str((bundle_dir / "inference_val.metrics.json").relative_to(attempt_dir)),
                },
            }
            run_manifest = {
                "schema_version": RUN_STORE_SCHEMA_VERSION,
                "published_attempt_id": "attempt_seed13",
                "active_attempt_id": None,
                "attempts": [
                    {
                        "attempt_id": "attempt_seed13",
                        "attempt_relpath": str(attempt_dir.relative_to(run_dir)),
                    }
                ],
            }
            (attempt_dir / "attempt_manifest.json").write_text(json.dumps(attempt_manifest), encoding="utf-8")
            (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")

            summary = validate_run_dir(run_dir)
            self.assertEqual(summary["issues"], [])

    def test_register_published_inference_output_updates_attempt_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            attempt_dir = Path(tmp) / "attempt"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            output_path = attempt_dir / "published" / "inference_test.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")
            manifest = {
                "schema_version": RUN_STORE_SCHEMA_VERSION,
                "attempt_id": "attempt_seed13",
                "published_inference_outputs": {},
                "published_inference_metrics": {},
            }
            (attempt_dir / "attempt_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            register_published_inference_output(attempt_dir, subset="test", output_path=output_path)

            updated = json.loads((attempt_dir / "attempt_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["published_inference_outputs"]["test"], "published/inference_test.jsonl")
            self.assertEqual(updated["published_inference_metrics"]["test"], "published/inference_test.metrics.json")


if __name__ == "__main__":
    unittest.main()
