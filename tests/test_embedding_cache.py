from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from embedding_cache import (
    EMBEDDING_CACHE_INDEX_VERSION,
    EMBEDDING_CACHE_PROTOCOL_VERSION,
    EMBEDDING_CACHE_STORAGE_PER_SAMPLE,
    EmbeddingCacheReader,
    build_embedding_cache_contract,
    save_embedding_cache_index,
    save_embedding_cache_meta,
    save_embedding_payload,
    validate_embedding_cache_contract,
    validate_embedding_cache_runtime_allowed,
)


def _meta() -> dict[str, object]:
    return {
        "protocol_version": EMBEDDING_CACHE_PROTOCOL_VERSION,
        "manifest_sha256": "manifest-1",
        "dataset_kind": "meld",
        "text_model": "FacebookAI/xlm-roberta-large",
        "audio_model": "microsoft/wavlm-large",
        "audio_model_revision": "rev",
        "video_model": "MCG-NJU/videomae-large",
        "max_text_len": 128,
        "sample_rate": 16000,
        "max_audio_sec": 6.0,
        "num_frames": 32,
        "rgb_size": 224,
        "text_pooling": "cls",
        "audio_pooling": "mean_std_masked_v1",
        "rgb_pooling": "videomae_pooler_or_cls_v1",
        "pooling_version": "mainline_embedding_pooling_v1",
        "embedding_dtype": "float32",
        "text_dim": 4,
        "audio_dim": 6,
        "rgb_dim": 8,
        "has_text_emb": True,
        "has_audio_emb": True,
        "has_rgb_emb": True,
        "freeze_text_required": True,
        "freeze_audio_required": True,
        "freeze_rgb_required": True,
        "audio_aug_allowed": False,
        "storage_format": EMBEDDING_CACHE_STORAGE_PER_SAMPLE,
        "index_version": EMBEDDING_CACHE_INDEX_VERSION,
    }


class EmbeddingCacheContractTests(unittest.TestCase):
    def test_runtime_requires_frozen_backbones_and_no_audio_aug(self) -> None:
        self.assertEqual(
            validate_embedding_cache_runtime_allowed(
                freeze_text=True,
                freeze_audio=True,
                freeze_rgb=True,
                audio_aug=False,
            ),
            [],
        )
        reasons = validate_embedding_cache_runtime_allowed(
            freeze_text=False,
            freeze_audio=True,
            freeze_rgb=False,
            audio_aug=True,
        )
        self.assertIn("freeze_text_required", reasons)
        self.assertIn("freeze_rgb_required", reasons)
        self.assertIn("audio_aug_not_allowed", reasons)

    def test_contract_validation_hard_mismatch_reasons(self) -> None:
        contract = build_embedding_cache_contract(_meta())
        reasons = validate_embedding_cache_contract(
            contract,
            manifest_sha256="other",
            dataset_kind="meld",
            text_model="FacebookAI/xlm-roberta-large",
            audio_model="microsoft/wavlm-large",
            audio_model_revision="rev",
            video_model="MCG-NJU/videomae-large",
            max_text_len=128,
            sample_rate=16000,
            max_audio_sec=6.0,
            num_frames=32,
            rgb_size=224,
            text_dim=4,
            audio_dim=6,
            rgb_dim=8,
            need_text=True,
            need_audio=True,
            need_rgb=True,
        )
        self.assertIn("manifest_sha256_mismatch", reasons)


class EmbeddingCacheReaderTests(unittest.TestCase):
    def test_reader_stacks_batch_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            save_embedding_cache_meta(cache_dir, _meta())
            entries = [
                save_embedding_payload(
                    cache_dir,
                    "train:a",
                    {
                        "text_emb": torch.ones(4),
                        "audio_emb": torch.ones(6) * 2,
                        "rgb_emb": torch.ones(8) * 3,
                    },
                )
            ]
            save_embedding_cache_index(cache_dir, entries)
            reader = EmbeddingCacheReader(cache_dir)
            batch = reader.load_batch([{"split": "train", "seq": "a"}], device=torch.device("cpu"))
            self.assertEqual(tuple(batch["text_emb"].shape), (1, 4))
            self.assertEqual(float(batch["audio_emb"][0, 0].item()), 2.0)
            self.assertEqual(float(batch["rgb_emb"][0, 0].item()), 3.0)


if __name__ == "__main__":
    unittest.main()
