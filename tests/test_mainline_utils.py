from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class _FakePyplot(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("matplotlib.pyplot")

    def figure(self, *args, **kwargs) -> None:
        return None

    def plot(self, *args, **kwargs) -> None:
        return None

    def xlabel(self, *args, **kwargs) -> None:
        return None

    def ylabel(self, *args, **kwargs) -> None:
        return None

    def title(self, *args, **kwargs) -> None:
        return None

    def grid(self, *args, **kwargs) -> None:
        return None

    def legend(self, *args, **kwargs) -> None:
        return None

    def tight_layout(self, *args, **kwargs) -> None:
        raise RuntimeError("tight_layout failed")

    def savefig(self, *args, **kwargs) -> None:
        return None

    def close(self, *args, **kwargs) -> None:
        return None


class MainlineUtilsTests(unittest.TestCase):
    def test_save_metrics_and_plots_tolerates_plot_failures(self) -> None:
        fake_matplotlib = types.ModuleType("matplotlib")
        fake_matplotlib.use = lambda *args, **kwargs: None
        fake_pyplot = _FakePyplot()
        fake_torch = types.ModuleType("torch")
        metrics = {
            "train_loss": [1.0],
            "val_loss": [1.1],
            "train_acc": [0.5],
            "val_acc": [0.4],
            "train_f1": [0.2],
            "val_f1": [0.1],
            "meta": {"label_names": ["angry", "happy"]},
            "best": {"best_val_summary": {"confusion_matrix": [[1, 0], [0, 1]]}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with mock.patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "matplotlib": fake_matplotlib,
                    "matplotlib.pyplot": fake_pyplot,
                },
            ):
                from mainline_utils import save_metrics_and_plots

                save_metrics_and_plots(out_dir, metrics)

            saved = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["val_f1"], [0.1])


if __name__ == "__main__":
    unittest.main()
