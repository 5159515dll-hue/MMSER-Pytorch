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


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    """按掩码计算均值。"""

    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum(dim=dim).clamp(min=1.0)
    return (x * mask_f).sum(dim=dim) / denom


def _masked_std(x: torch.Tensor, mask: torch.Tensor, *, dim: int, mean: torch.Tensor) -> torch.Tensor:
    """按掩码计算标准差。"""

    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum(dim=dim).clamp(min=1.0)
    centered = x - mean.unsqueeze(dim)
    var = (centered * centered * mask_f).sum(dim=dim) / denom
    return torch.sqrt(var.clamp_min(1e-9))


def _masked_minmax_range(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """按掩码计算 max-min。"""

    pos_inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    neg_inf = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    x_min = torch.where(mask, x, pos_inf).min(dim=1).values
    x_max = torch.where(mask, x, neg_inf).max(dim=1).values
    valid = mask.any(dim=1)
    out = torch.zeros_like(x_min)
    out[valid] = x_max[valid] - x_min[valid]
    return out


def extract_prosody_features_gpu(
    wav: torch.Tensor,
    cfg: ProsodyConfig,
    *,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """在当前设备上提取 batched 韵律特征。

    说明
    - 这是 `extract_prosody_features` 的 CUDA-first / torch-only 版本。
    - 它不依赖 torchaudio 的 CPU pitch API，而是使用短时频谱峰值做 pitch proxy；
      目标是任务等价而不是与旧 CPU 路线逐数值一致。
    - 输入支持 `(L,)` 或 `(B, L)`，输出维度保持为 10。
    """

    x = wav.detach().to(torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"Expected wav to have shape (L,) or (B, L), got {tuple(x.shape)}")

    batch_size, wav_len = x.shape
    if lengths is None:
        lengths = torch.full((batch_size,), wav_len, device=x.device, dtype=torch.long)
    else:
        lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1, max=wav_len)

    frame_len = max(1, int(cfg.sample_rate * cfg.frame_ms / 1000.0))
    hop_len = max(1, int(cfg.sample_rate * cfg.hop_ms / 1000.0))
    if wav_len < frame_len:
        pad = frame_len - wav_len
        x = torch.nn.functional.pad(x, (0, pad))
        wav_len = x.shape[1]
        lengths = lengths.clamp(max=wav_len)

    n_frames = max(1, 1 + (wav_len - frame_len) // hop_len)
    frames = x.unfold(dimension=1, size=frame_len, step=hop_len)  # (B, T, frame_len)
    frame_starts = torch.arange(n_frames, device=x.device, dtype=torch.long) * hop_len
    frame_valid = frame_starts.unsqueeze(0) < lengths.unsqueeze(1)

    rms = torch.sqrt(torch.mean(frames * frames, dim=-1) + 1e-9)
    rms_mean = _masked_mean(rms, frame_valid, dim=1)
    rms_std = _masked_std(rms, frame_valid, dim=1, mean=rms_mean)
    rms_range = _masked_minmax_range(rms, frame_valid)

    if n_frames > 1:
        drms = torch.diff(rms, dim=1).abs()
        drms_mask = frame_valid[:, 1:] & frame_valid[:, :-1]
        drms_mean = _masked_mean(drms, drms_mask, dim=1)
    else:
        drms_mean = torch.zeros((batch_size,), device=x.device, dtype=torch.float32)

    if cfg.use_pitch:
        window = torch.hann_window(frame_len, device=x.device, dtype=x.dtype)
        spec = torch.fft.rfft(frames * window.view(1, 1, -1), dim=-1)
        mag = spec.abs()
        freqs = torch.fft.rfftfreq(frame_len, d=1.0 / float(cfg.sample_rate)).to(device=x.device, dtype=x.dtype)
        freq_mask = (freqs >= float(cfg.fmin)) & (freqs <= float(cfg.fmax))
        if bool(freq_mask.any().item()):
            band = mag[..., freq_mask]
            band_freqs = freqs[freq_mask]
            peak_idx = band.argmax(dim=-1)
            f0 = band_freqs[peak_idx]
            voiced_thresh = rms_mean.unsqueeze(1) * 0.5
            voiced = frame_valid & torch.isfinite(f0) & (rms > voiced_thresh)
        else:
            f0 = torch.zeros((batch_size, n_frames), device=x.device, dtype=torch.float32)
            voiced = torch.zeros((batch_size, n_frames), device=x.device, dtype=torch.bool)
    else:
        f0 = torch.zeros((batch_size, n_frames), device=x.device, dtype=torch.float32)
        voiced = torch.zeros((batch_size, n_frames), device=x.device, dtype=torch.bool)

    voiced_ratio = voiced.to(torch.float32).mean(dim=1)
    if bool(voiced.any().item()):
        f0_mean = _masked_mean(f0, voiced, dim=1)
        f0_std = _masked_std(f0, voiced, dim=1, mean=f0_mean)
        f0_range = _masked_minmax_range(f0, voiced)
        if n_frames > 1:
            df0 = torch.diff(f0, dim=1).abs()
            df0_mask = voiced[:, 1:] & voiced[:, :-1]
            df0_mean = _masked_mean(df0, df0_mask, dim=1)
        else:
            df0_mean = torch.zeros((batch_size,), device=x.device, dtype=torch.float32)
    else:
        f0_mean = torch.zeros((batch_size,), device=x.device, dtype=torch.float32)
        f0_std = torch.zeros((batch_size,), device=x.device, dtype=torch.float32)
        f0_range = torch.zeros((batch_size,), device=x.device, dtype=torch.float32)
        df0_mean = torch.zeros((batch_size,), device=x.device, dtype=torch.float32)

    duration_sec = lengths.to(torch.float32) / float(cfg.sample_rate)
    feat = torch.stack(
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
        dim=1,
    ).to(torch.float32)
    return feat[0] if wav.ndim == 1 else feat


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
