from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hf_loading import resolve_hf_pretrained_source


class HfLoadingTests(unittest.TestCase):
    def test_existing_local_path_is_used_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            source, kwargs = resolve_hf_pretrained_source(str(model_dir))
            self.assertEqual(source, str(model_dir))
            self.assertTrue(kwargs.get("local_files_only"))

    def test_cached_snapshot_is_preferred_over_remote_repo_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp) / "snapshot"
            snapshot_dir.mkdir()
            with patch("hf_loading.resolve_local_hf_snapshot", return_value=snapshot_dir):
                source, kwargs = resolve_hf_pretrained_source("FacebookAI/xlm-roberta-large")
            self.assertEqual(source, str(snapshot_dir))
            self.assertTrue(kwargs.get("local_files_only"))

    def test_offline_mode_forces_local_files_only_on_remote_fallback(self) -> None:
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=False):
            with patch("hf_loading.resolve_local_hf_snapshot", return_value=None):
                source, kwargs = resolve_hf_pretrained_source("FacebookAI/xlm-roberta-large")
        self.assertEqual(source, "FacebookAI/xlm-roberta-large")
        self.assertTrue(kwargs.get("local_files_only"))
        self.assertIn("cache_dir", kwargs)


if __name__ == "__main__":
    unittest.main()
