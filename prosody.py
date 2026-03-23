"""显式韵律特征提取。

这个模块故意不走 Mel-spectrogram 一类高维频谱路线，而是直接提取
低维、可解释的统计量，例如：
- 基频相关统计；
- 能量相关统计；
- 相邻帧变化速度；
- 音频总时长。

这样做的好处是维度低、解释性强，也便于和大模型声学表征形成互补。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class ProsodyConfig:
    """韵律提取参数。

    - `frame_ms` / `hop_ms` 控制短时分析窗口；
    - `fmin` / `fmax` 限制基频搜索范围；
    - `use_pitch` 可以在速度优先时关闭 F0 提取，只保留能量统计。
    """

    sample_rate: int = 24000
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    fmin: float = 50.0
    fmax: float = 400.0
    use_pitch: bool = True


def _frame_audio(wav: torch.Tensor, frame_len: int, hop_len: int) -> torch.Tensor:
    """把一维波形切成重叠短时帧。

    这是后续 RMS 和基频统计的基础。若波形太短，先补零到至少一帧，
    保证下游不会因为空帧而崩溃。
    """
    if wav.numel() < frame_len:
        pad = frame_len - wav.numel()
        wav = torch.nn.functional.pad(wav, (0, pad))
    n_frames = 1 + (wav.numel() - frame_len) // hop_len
    frames = wav.unfold(0, frame_len, hop_len)  # (n_frames, frame_len)
    return frames


def _rms(frames: torch.Tensor) -> torch.Tensor:
    """计算每一帧的 RMS 能量。"""

    return torch.sqrt(torch.mean(frames * frames, dim=-1) + 1e-9)


def extract_prosody_features(wav: torch.Tensor, cfg: ProsodyConfig) -> torch.Tensor:
    """Return a small prosody feature vector (float32).

    Uses torchaudio if available for pitch; otherwise falls back to energy-only stats.

    Output dims (currently 10):
      [voiced_ratio, f0_mean, f0_std, f0_range, df0_mean,
       rms_mean, rms_std, rms_range, drms_mean, duration_sec]
    """
    # 韵律统计不需要梯度，也不需要驻留 GPU；统一拉平到 CPU 更稳定。
    wav = wav.detach().float().flatten().cpu()

    frame_len = int(cfg.sample_rate * cfg.frame_ms / 1000.0)
    hop_len = int(cfg.sample_rate * cfg.hop_ms / 1000.0)

    frames = _frame_audio(wav, frame_len, hop_len)
    rms = _rms(frames)

    # Pitch estimate
    f0 = None
    try:
        if cfg.use_pitch:
            import torchaudio

            # (n_frames,) in Hz
            f0 = torchaudio.functional.detect_pitch_frequency(
                wav.unsqueeze(0),
                sample_rate=cfg.sample_rate,
                frame_time=cfg.frame_ms / 1000.0,
                win_length=int(cfg.sample_rate * cfg.frame_ms / 1000.0),
                freq_low=int(cfg.fmin),
                freq_high=int(cfg.fmax),
            ).squeeze(0)
            f0 = f0.to(torch.float32).cpu()
    except Exception:
        f0 = None

    if f0 is not None and f0.numel() == rms.numel():
        voiced = (f0 > 1.0) & torch.isfinite(f0)
        voiced_ratio = voiced.float().mean().item()
        f0_voiced = f0[voiced]
        if f0_voiced.numel() == 0:
            f0_mean = 0.0
            f0_std = 0.0
            f0_range = 0.0
            df0_mean = 0.0
        else:
            f0_mean = f0_voiced.mean().item()
            f0_std = f0_voiced.std(unbiased=False).item()
            f0_range = (f0_voiced.max() - f0_voiced.min()).item()
            # `df0_mean = mean(|f0_t - f0_{t-1}|)`，
            # 用来描述说话过程中基频变化的活跃程度。
            df0 = torch.diff(f0_voiced)
            df0_mean = df0.abs().mean().item() if df0.numel() else 0.0
    else:
        voiced_ratio = 0.0
        f0_mean = 0.0
        f0_std = 0.0
        f0_range = 0.0
        df0_mean = 0.0

    rms_mean = rms.mean().item()
    rms_std = rms.std(unbiased=False).item()
    rms_range = (rms.max() - rms.min()).item() if rms.numel() else 0.0
    # `drms_mean = mean(|rms_t - rms_{t-1}|)`，
    # 表示音强在时间上的起伏速度。
    drms = torch.diff(rms)
    drms_mean = drms.abs().mean().item() if drms.numel() else 0.0

    duration_sec = wav.numel() / float(cfg.sample_rate)

    feat = torch.tensor(
        [
            voiced_ratio,
            f0_mean,
            f0_std,
            f0_range,
            df0_mean,
            rms_mean,
            rms_std,
            rms_range,
            drms_mean,
            duration_sec,
        ],
        dtype=torch.float32,
    )
    return feat
