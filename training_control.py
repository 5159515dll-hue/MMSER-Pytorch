"""训练停止控制器。

这个模块把“什么时候算提升”“什么时候只算 tie-break”“什么时候允许早停”
从训练主循环里抽离出来，让训练逻辑更容易理解和复用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


def _is_better(value: float, best: float | None, *, mode: str) -> bool:
    """按 `max/min` 规则比较当前值是否优于历史值。"""

    if best is None:
        return True
    if mode == "max":
        return float(value) > float(best)
    if mode == "min":
        return float(value) < float(best)
    raise ValueError(f"unsupported mode: {mode}")


@dataclass(frozen=True)
class EarlyStopConfig:
    """早停控制器的静态配置。"""

    monitor_name: str
    monitor_mode: str = "max"
    patience: int = 0
    min_epochs: int = 0
    min_delta: float = 0.0
    tie_break_name: str = "val_loss"
    tie_break_mode: str = "min"
    after_lr_drops: int = 0


@dataclass(frozen=True)
class EpochDecision:
    """单个 epoch 观察后的决策结果。"""

    should_save_checkpoint: bool
    checkpoint_reason: str | None
    significant_improvement: bool
    should_stop: bool
    stop_reason: str | None


@dataclass
class EarlyStopController:
    """维护训练过程中的“最佳 epoch”与“是否早停”状态。"""

    config: EarlyStopConfig
    best_monitor_value: float | None = None
    selected_epoch: int = 0
    selected_monitor_value: float | None = None
    selected_tie_break_value: float | None = None
    selected_reason: str | None = None
    last_significant_epoch: int = 0
    epochs_without_improvement: int = 0
    lr_drop_epochs: list[int] = field(default_factory=list)
    stop_reason: str | None = None

    def _is_significant_improvement(self, value: float) -> bool:
        """判断当前 monitor 是否真正超过 `min_delta` 门槛。"""

        if self.best_monitor_value is None:
            return True
        if self.config.monitor_mode == "max":
            return float(value) > float(self.best_monitor_value) + float(self.config.min_delta)
        if self.config.monitor_mode == "min":
            return float(value) < float(self.best_monitor_value) - float(self.config.min_delta)
        raise ValueError(f"unsupported monitor mode: {self.config.monitor_mode}")

    def _within_delta_of_best(self, value: float) -> bool:
        """判断当前值是否仍落在“接近最佳值”的 tie-break 区间内。"""

        if self.best_monitor_value is None:
            return True
        return abs(float(value) - float(self.best_monitor_value)) <= float(self.config.min_delta)

    def register_lr_drop(self, epoch: int) -> None:
        """记录学习率发生降档的 epoch。"""

        epoch_int = int(epoch)
        if epoch_int not in self.lr_drop_epochs:
            self.lr_drop_epochs.append(epoch_int)

    def evaluate_stop(self, epoch: int) -> tuple[bool, str | None]:
        """在当前 epoch 结束后判断是否允许触发早停。"""

        epoch_int = int(epoch)
        if int(self.config.patience) <= 0:
            return False, None
        if self.epochs_without_improvement < int(self.config.patience):
            return False, None
        if epoch_int < int(self.config.min_epochs):
            return False, None
        if len(self.lr_drop_epochs) < int(self.config.after_lr_drops):
            return False, None
        stop_reason = (
            f"no significant improvement on {self.config.monitor_name} for "
            f"{self.epochs_without_improvement} epoch(s) after {len(self.lr_drop_epochs)} lr drop(s)"
        )
        self.stop_reason = stop_reason
        return True, stop_reason

    def observe(self, *, epoch: int, monitor_value: float, tie_break_value: float) -> EpochDecision:
        """喂入一个 epoch 的观测值，并更新“是否保存/是否停止”状态。

        这里把两层逻辑拆开处理：
        - 先看 monitor 是否有显著提升；如果有，就刷新“真正最佳”
        - 如果没有显著提升，但仍处在 `min_delta` 邻域内，就允许用 tie-break
          指标（通常是更低的 val_loss）决定是否更新 checkpoint
        """

        epoch_int = int(epoch)
        monitor_f = float(monitor_value)
        tie_break_f = float(tie_break_value)
        if not math.isfinite(monitor_f):
            raise ValueError(f"non-finite monitor_value: {monitor_f}")
        if not math.isfinite(tie_break_f):
            raise ValueError(f"non-finite tie_break_value: {tie_break_f}")

        significant_improvement = self._is_significant_improvement(monitor_f)
        checkpoint_reason: str | None = None
        should_save_checkpoint = False

        if significant_improvement:
            self.best_monitor_value = monitor_f
            self.last_significant_epoch = epoch_int
            self.epochs_without_improvement = 0
            should_save_checkpoint = True
            checkpoint_reason = "monitor_improved"
        else:
            self.epochs_without_improvement += 1
            if self._within_delta_of_best(monitor_f) and _is_better(
                tie_break_f,
                self.selected_tie_break_value,
                mode=self.config.tie_break_mode,
            ):
                should_save_checkpoint = True
                checkpoint_reason = "tie_break"

        if should_save_checkpoint:
            self.selected_epoch = epoch_int
            self.selected_monitor_value = monitor_f
            self.selected_tie_break_value = tie_break_f
            self.selected_reason = checkpoint_reason

        should_stop, stop_reason = self.evaluate_stop(epoch_int)

        return EpochDecision(
            should_save_checkpoint=bool(should_save_checkpoint),
            checkpoint_reason=checkpoint_reason,
            significant_improvement=bool(significant_improvement),
            should_stop=bool(should_stop),
            stop_reason=stop_reason,
        )
