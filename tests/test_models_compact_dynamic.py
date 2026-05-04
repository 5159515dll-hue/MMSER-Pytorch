from __future__ import annotations

import unittest

import torch

from models import CompactDynamicFusion
from run_store import missing_paper_contract_fields, normalize_paper_contract_subset


class CompactDynamicFusionTests(unittest.TestCase):
    def test_forward_accepts_missing_video_branch(self) -> None:
        fusion = CompactDynamicFusion(
            {
                "text": 8,
                "audio": 6,
                "rgb": 10,
                "flow": 4,
                "prosody": 3,
            },
            compact_dim=16,
            hidden=32,
            dropout=0.0,
        )
        output = fusion(
            {
                "text": torch.randn(2, 8),
                "audio": torch.randn(2, 6),
                "rgb": torch.randn(2, 10),
                "flow": None,
                "prosody": torch.randn(2, 3),
            }
        )
        self.assertEqual(tuple(output.shape), (2, 16))
        self.assertTrue(torch.isfinite(output).all().item())


class FusionModeContractTests(unittest.TestCase):
    def test_missing_fusion_mode_defaults_to_historical_gated_text(self) -> None:
        contract = {
            "protocol_version": "paper_grade_v1",
            "manifest_sha256": "manifest-1",
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
            "text_model": "xlm-roberta-large",
            "max_text_len": 128,
            "sample_rate": 24000,
            "max_audio_sec": 6.0,
            "num_frames": 64,
            "rgb_size": 224,
            "label_names": ["neutral", "joy"],
        }
        normalized = normalize_paper_contract_subset(contract)
        self.assertEqual(normalized["fusion_mode"], "gated_text")
        self.assertNotIn("fusion_mode", missing_paper_contract_fields(normalized))


if __name__ == "__main__":
    unittest.main()
