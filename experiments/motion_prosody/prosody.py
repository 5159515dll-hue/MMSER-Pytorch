from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class ProsodyConfig:
    sample_rate: int = 24000
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    fmin: float = 50.0
    fmax: float = 400.0
    use_pitch: bool = True


def _frame_audio(wav: torch.Tensor, frame_len: int, hop_len: int) -> torch.Tensor:
    # wav: (L,)
    if wav.numel() < frame_len:
        pad = frame_len - wav.numel()
        wav = torch.nn.functional.pad(wav, (0, pad))
    n_frames = 1 + (wav.numel() - frame_len) // hop_len
    frames = wav.unfold(0, frame_len, hop_len)  # (n_frames, frame_len)
    return frames


def _rms(frames: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(frames * frames, dim=-1) + 1e-9)


def extract_prosody_features(wav: torch.Tensor, cfg: ProsodyConfig) -> torch.Tensor:
    """Return a small prosody feature vector (float32).

    Uses torchaudio if available for pitch; otherwise falls back to energy-only stats.

    Output dims (currently 10):
      [voiced_ratio, f0_mean, f0_std, f0_range, df0_mean,
       rms_mean, rms_std, rms_range, drms_mean, duration_sec]
    """
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
