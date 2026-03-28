from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from input_cache import (
    INPUT_CACHE_PROTOCOL_VERSION,
    build_input_cache_contract,
    index_entries_by_key,
    load_input_cache_index,
    load_input_cache_meta,
    manifest_item_cache_key,
    sample_relpath_for_key,
    save_input_cache_index,
    save_input_cache_meta,
    validate_input_cache_contract,
)


class InputCacheTests(unittest.TestCase):
    def test_manifest_item_cache_key_prefers_split_and_seq(self) -> None:
        key = manifest_item_cache_key({"split": "train", "seq": "meld_train_dia1_utt2"})
        self.assertEqual(key, "train:meld_train_dia1_utt2")

    def test_sample_relpath_is_stable(self) -> None:
        path1 = sample_relpath_for_key("train:abc")
        path2 = sample_relpath_for_key("train:abc")
        self.assertEqual(path1, path2)
        self.assertEqual(path1.parent.parent.name, "samples")
        self.assertEqual(path1.suffix, ".pt")

    def test_meta_and_index_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            meta = {
                "protocol_version": INPUT_CACHE_PROTOCOL_VERSION,
                "manifest_sha256": "abc123",
                "dataset_kind": "meld",
            }
            entries = [
                {"cache_key": "train:a", "relpath": "samples/aa/a.pt", "sample_bytes": 123},
                {"cache_key": "val:b", "relpath": "samples/bb/b.pt", "sample_bytes": 456},
            ]
            save_input_cache_meta(cache_dir, meta)
            save_input_cache_index(cache_dir, entries)
            self.assertEqual(load_input_cache_meta(cache_dir), meta)
            loaded_entries = load_input_cache_index(cache_dir)
            self.assertEqual(loaded_entries, entries)
            indexed = index_entries_by_key(loaded_entries)
            self.assertEqual(indexed["train:a"]["sample_bytes"], 123)

    def test_validate_contract_requires_modalities_and_matching_manifest(self) -> None:
        contract = build_input_cache_contract(
            {
                "protocol_version": INPUT_CACHE_PROTOCOL_VERSION,
                "manifest_sha256": "manifest-ok",
                "dataset_kind": "meld",
                "sample_rate": 16000,
                "max_audio_sec": 6.0,
                "num_frames": 16,
                "rgb_size": 224,
                "text_model": "FacebookAI/xlm-roberta-large",
                "max_text_len": 128,
                "has_audio": True,
                "has_video": True,
                "video_representation": "prepared_rgb_fp16",
                "has_text_full_tokens": True,
                "has_text_masked_tokens": True,
            }
        )
        reasons = validate_input_cache_contract(
            contract,
            manifest_sha256="manifest-ok",
            dataset_kind="meld",
            sample_rate=16000,
            max_audio_sec=6.0,
            num_frames=16,
            rgb_size=224,
            text_model="xlm-roberta-large",
            max_text_len=128,
            need_audio=True,
            need_video=True,
            need_text=True,
            text_policy="full",
        )
        self.assertEqual(reasons, [])

        reasons = validate_input_cache_contract(
            contract,
            manifest_sha256="manifest-bad",
            dataset_kind="meld",
            sample_rate=16000,
            max_audio_sec=6.0,
            num_frames=16,
            rgb_size=224,
            text_model="xlm-roberta-large",
            max_text_len=128,
            need_audio=True,
            need_video=True,
            need_text=True,
            text_policy="mask_emotion_cues",
        )
        self.assertIn("manifest_sha256_mismatch", reasons)

        no_video_contract = dict(contract)
        no_video_contract["has_video"] = False
        reasons = validate_input_cache_contract(
            no_video_contract,
            manifest_sha256="manifest-ok",
            dataset_kind="meld",
            sample_rate=16000,
            max_audio_sec=6.0,
            num_frames=16,
            rgb_size=224,
            text_model="xlm-roberta-large",
            max_text_len=128,
            need_audio=False,
            need_video=True,
            need_text=False,
            text_policy="full",
        )
        self.assertIn("missing_cached_video", reasons)


if __name__ == "__main__":
    unittest.main()
