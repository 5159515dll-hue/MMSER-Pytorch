from __future__ import annotations

import unittest

from training_control import EarlyStopConfig, EarlyStopController


class EarlyStopControllerTests(unittest.TestCase):
    def test_tie_break_checkpoint_does_not_reset_patience(self) -> None:
        controller = EarlyStopController(
            EarlyStopConfig(
                monitor_name="val_f1",
                patience=3,
                min_epochs=0,
                min_delta=0.01,
                after_lr_drops=0,
            )
        )

        first = controller.observe(epoch=1, monitor_value=0.50, tie_break_value=1.0)
        self.assertTrue(first.should_save_checkpoint)
        self.assertTrue(first.significant_improvement)
        self.assertEqual(controller.epochs_without_improvement, 0)

        second = controller.observe(epoch=2, monitor_value=0.505, tie_break_value=0.9)
        self.assertTrue(second.should_save_checkpoint)
        self.assertFalse(second.significant_improvement)
        self.assertEqual(second.checkpoint_reason, "tie_break")
        self.assertEqual(controller.epochs_without_improvement, 1)
        self.assertEqual(controller.selected_epoch, 2)

    def test_stop_requires_min_epochs_and_lr_drops(self) -> None:
        controller = EarlyStopController(
            EarlyStopConfig(
                monitor_name="val_f1",
                patience=2,
                min_epochs=3,
                min_delta=0.01,
                after_lr_drops=2,
            )
        )

        controller.observe(epoch=1, monitor_value=0.60, tie_break_value=1.0)
        controller.observe(epoch=2, monitor_value=0.605, tie_break_value=0.9)
        should_stop, _ = controller.evaluate_stop(epoch=2)
        self.assertFalse(should_stop)

        controller.register_lr_drop(2)
        controller.observe(epoch=3, monitor_value=0.604, tie_break_value=0.8)
        should_stop, _ = controller.evaluate_stop(epoch=3)
        self.assertFalse(should_stop)

        controller.register_lr_drop(3)
        should_stop, stop_reason = controller.evaluate_stop(epoch=3)
        self.assertTrue(should_stop)
        self.assertIn("after 2 lr drop(s)", str(stop_reason))

    def test_non_finite_monitor_is_rejected(self) -> None:
        controller = EarlyStopController(
            EarlyStopConfig(
                monitor_name="val_f1",
                patience=2,
            )
        )
        with self.assertRaises(ValueError):
            controller.observe(epoch=1, monitor_value=float("nan"), tie_break_value=1.0)


if __name__ == "__main__":
    unittest.main()
