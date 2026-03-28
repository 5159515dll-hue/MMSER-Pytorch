from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_report_utils import (
    apply_multiple_comparison_correction,
    build_pairwise_comparison,
    ensure_group_is_paper_grade_ready,
    summarize_experiment_group,
)


class BenchmarkReportUtilsTests(unittest.TestCase):
    def _write_run(
        self,
        root: Path,
        name: str,
        *,
        seed: int,
        test_macro_f1: float,
        test_accuracy: float,
        manifest_sha256: str = "manifest-1",
        input_cache_used: bool = False,
    ) -> Path:
        run_dir = root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        contract = {
            "protocol_version": "paper_grade_v1",
            "manifest_sha256": manifest_sha256,
            "dataset_kind": "meld",
            "task_mode": "confounded_7way",
            "speaker_id": None,
            "text_policy": "full",
            "claim_scope": "multimodal_7way_benchmark",
            "scientific_validity": True,
            "ablation": "full",
            "zero_video": False,
            "zero_audio": False,
            "zero_text": False,
            "use_intensity": False,
            "video_backbone": "dual",
            "flow_encoder_variant": "flow3d_strideconv_mean_v3",
            "label_names": ["a", "b"],
        }
        train_metrics = {
            "meta": {
                "args": {"seed": seed},
                "deterministic_policy": {"deterministic_algorithms_enabled": True},
                "paper_contract": contract,
                "input_cache": "outputs/input_cache/demo" if input_cache_used else None,
                "input_cache_contract": (
                    {"protocol_version": "mainline_input_cache_v1", "manifest_sha256": manifest_sha256}
                    if input_cache_used
                    else None
                ),
            },
            "best": {
                "epoch": 12,
                "best_f1": test_macro_f1 - 0.05,
                "best_acc": test_accuracy - 0.05,
                "best_val_loss": 1.23,
                "best_monitor_value": test_macro_f1 - 0.05,
                "significant_best_monitor_value": test_macro_f1 - 0.04,
                "checkpoint_reason": "monitor_improved",
            },
            "stop": {
                "epoch": 18,
                "reason": "max_epochs_reached",
                "lr_drop_epochs": [8, 14],
            },
            "validity": {
                "claim_scope": "multimodal_7way_benchmark",
                "scientific_validity": True,
            },
            "paper_grade": {
                "protocol_version": "paper_grade_v1",
                "eligible": True,
                "ineligibility_reasons": [],
            },
            "run_status": "completed",
        }
        val_metrics = {
            "checkpoint": str(run_dir / "checkpoints" / "best.pt"),
            "macro_f1_on_ok": test_macro_f1 - 0.05,
            "accuracy_on_ok": test_accuracy - 0.05,
            "manifest_sha256": manifest_sha256,
            "dataset_kind": "meld",
            "task_mode": "confounded_7way",
            "speaker_id": None,
            "text_policy": "full",
            "input_cache": "outputs/input_cache/demo" if input_cache_used else None,
            "input_cache_contract": (
                {"protocol_version": "mainline_input_cache_v1", "manifest_sha256": manifest_sha256}
                if input_cache_used
                else None
            ),
            "paper_contract": contract,
            "paper_grade": {
                "protocol_version": "paper_grade_v1",
                "eligible": True,
                "ineligibility_reasons": [],
            },
        }
        test_metrics = {
            "checkpoint": str(run_dir / "checkpoints" / "best.pt"),
            "macro_f1_on_ok": test_macro_f1,
            "accuracy_on_ok": test_accuracy,
            "manifest_sha256": manifest_sha256,
            "dataset_kind": "meld",
            "task_mode": "confounded_7way",
            "speaker_id": None,
            "text_policy": "full",
            "input_cache": "outputs/input_cache/demo" if input_cache_used else None,
            "input_cache_contract": (
                {"protocol_version": "mainline_input_cache_v1", "manifest_sha256": manifest_sha256}
                if input_cache_used
                else None
            ),
            "paper_contract": contract,
            "paper_grade": {
                "protocol_version": "paper_grade_v1",
                "eligible": True,
                "ineligibility_reasons": [],
            },
        }
        (run_dir / "metrics.json").write_text(json.dumps(train_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "inference_val.metrics.json").write_text(
            json.dumps(val_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "inference_test.metrics.json").write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return run_dir

    def test_group_summary_and_pairwise_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_runs = []
            candidate_runs = []
            seeds = [13, 17, 23, 42, 3407]
            for idx, seed in enumerate(seeds):
                baseline_runs.append(
                    self._write_run(
                        root,
                        f"baseline_seed{seed}",
                        seed=seed,
                        test_macro_f1=0.50 + idx * 0.01,
                        test_accuracy=0.60 + idx * 0.01,
                    )
                )
                candidate_runs.append(
                    self._write_run(
                        root,
                        f"candidate_seed{seed}",
                        seed=seed,
                        test_macro_f1=0.60 + idx * 0.01,
                        test_accuracy=0.70 + idx * 0.01,
                    )
                )

            baseline = summarize_experiment_group("baseline", baseline_runs)
            candidate = summarize_experiment_group("candidate", candidate_runs)
            ensure_group_is_paper_grade_ready(baseline)
            ensure_group_is_paper_grade_ready(candidate)
            comparison = build_pairwise_comparison(baseline, candidate)

            self.assertEqual(baseline["run_count"], 5)
            self.assertTrue(baseline["paper_grade_ready"])
            self.assertAlmostEqual(candidate["test_macro_f1"]["mean"], 0.62, places=6)
            self.assertEqual(comparison["paired_seeds"], seeds)
            self.assertAlmostEqual(comparison["macro_f1"]["mean_diff"], 0.10, places=6)
            self.assertEqual(comparison["macro_f1"]["positive_gain_count"], 5)
            self.assertTrue(comparison["claim_supported"])

    def test_group_requires_complete_expected_seed_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                self._write_run(root, "seed13", seed=13, test_macro_f1=0.50, test_accuracy=0.60),
                self._write_run(root, "seed17", seed=17, test_macro_f1=0.51, test_accuracy=0.61),
            ]
            group = summarize_experiment_group("partial", runs)
            self.assertFalse(group["paper_grade_ready"])
            self.assertIn(23, group["missing_seeds"])
            with self.assertRaises(RuntimeError):
                ensure_group_is_paper_grade_ready(group)

    def test_all_mode_correction_updates_adjusted_p_values(self) -> None:
        comparisons = [
            {
                "baseline": "a",
                "candidate": "b",
                "paired_run_count": 5,
                "required_positive_gain_count": 4,
                "comparison_scope": "exploratory",
                "multiple_comparison_method": "none",
                "macro_f1": {"n": 5, "mean_diff": 0.10, "p_value": 0.01, "positive_gain_count": 5},
                "accuracy": {"n": 5, "mean_diff": 0.08, "p_value": 0.02, "positive_gain_count": 5},
                "claim_supported": True,
            },
            {
                "baseline": "b",
                "candidate": "c",
                "paired_run_count": 5,
                "required_positive_gain_count": 4,
                "comparison_scope": "exploratory",
                "multiple_comparison_method": "none",
                "macro_f1": {"n": 5, "mean_diff": 0.04, "p_value": 0.03, "positive_gain_count": 5},
                "accuracy": {"n": 5, "mean_diff": 0.03, "p_value": 0.04, "positive_gain_count": 5},
                "claim_supported": True,
            },
        ]
        corrected = apply_multiple_comparison_correction(comparisons)
        self.assertEqual(corrected[0]["multiple_comparison_method"], "holm_bonferroni")
        self.assertGreaterEqual(corrected[0]["macro_f1"]["adjusted_p_value"], corrected[0]["macro_f1"]["p_value"])
        self.assertGreaterEqual(corrected[1]["macro_f1"]["adjusted_p_value"], corrected[1]["macro_f1"]["p_value"])

    def test_group_rejects_mixed_input_cache_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                self._write_run(root, "seed13", seed=13, test_macro_f1=0.50, test_accuracy=0.60, input_cache_used=True),
                self._write_run(root, "seed17", seed=17, test_macro_f1=0.51, test_accuracy=0.61, input_cache_used=False),
                self._write_run(root, "seed23", seed=23, test_macro_f1=0.52, test_accuracy=0.62, input_cache_used=True),
                self._write_run(root, "seed42", seed=42, test_macro_f1=0.53, test_accuracy=0.63, input_cache_used=False),
                self._write_run(root, "seed3407", seed=3407, test_macro_f1=0.54, test_accuracy=0.64, input_cache_used=True),
            ]
            group = summarize_experiment_group("mixed-cache", runs)
            self.assertFalse(group["paper_grade_ready"])
            self.assertIn("input_cache_used", group["compatibility_report"]["mismatch_keys"])


if __name__ == "__main__":
    unittest.main()
