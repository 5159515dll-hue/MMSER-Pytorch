from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AudioAugConfig:
    sample_rate: int = 24000
    p_gain: float = 0.5
    gain_db: float = 6.0
    p_speed: float = 0.3
    speed_range: float = 0.05  # +/- 5%
    p_crop: float = 0.5
    crop_sec: float = 4.0


def normalize_wav(
    wav: torch.Tensor,
    *,
    target_rms: float = 0.1,
    remove_dc: bool = True,
    clamp: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize waveform amplitude.

    Supports wav shaped (L,) or (B,L). Returns float32.
    """
    x = wav.detach().to(torch.float32)

    if x.dim() == 1:
        if remove_dc:
            x = x - x.mean()
        rms = torch.sqrt(torch.mean(x * x) + eps)
        x = x * (target_rms / rms)
        return x.clamp(-clamp, clamp)

    if x.dim() == 2:
        if remove_dc:
            x = x - x.mean(dim=1, keepdim=True)
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + eps)
        x = x * (target_rms / rms)
        return x.clamp(-clamp, clamp)

    raise ValueError(f"Expected wav to have dim 1 or 2, got {x.shape}")


def augment_wav(wav: torch.Tensor, cfg: AudioAugConfig) -> torch.Tensor:
    """wav: (B,L) float32"""
    # Avoid in-place modifications affecting other references to the same storage.
    out = wav.clone()

    # random gain
    if cfg.p_gain > 0:
        mask = torch.rand(out.shape[0], device=out.device) < cfg.p_gain
        if mask.any():
            k = int(mask.sum().item())
            db = (torch.rand(k, device=out.device) * 2 - 1) * cfg.gain_db
            gain = torch.pow(10.0, db / 20.0).view(-1, 1)
            out[mask] = out[mask] * gain

    # random speed perturbation (resample-based)
    if cfg.p_speed > 0:
        mask = torch.rand(out.shape[0], device=out.device) < cfg.p_speed
        if mask.any():
            try:
                import torchaudio

                for i in torch.where(mask)[0].tolist():
                    factor = 1.0 + (torch.rand(1, device=out.device).item() * 2 - 1) * cfg.speed_range
                    # resample to sr*factor then back to sr
                    sr = cfg.sample_rate
                    wav_i = out[i]
                    # torchaudio resample on CUDA can be unstable depending on build; run resample on CPU.
                    wav_cpu = wav_i.detach().to("cpu")
                    wav_cpu = torchaudio.functional.resample(wav_cpu, sr, int(sr * factor))
                    wav_cpu = torchaudio.functional.resample(wav_cpu, int(sr * factor), sr)
                    wav_new = wav_cpu.to(out.device)
                    out[i, : wav_new.numel()] = wav_new[: out.shape[1]]
            except Exception:
                pass

    # random crop/pad to fixed length (helps reduce speaker cues)
    if cfg.p_crop > 0:
        mask = torch.rand(out.shape[0], device=out.device) < cfg.p_crop
        if mask.any():
            max_len = out.shape[1]
            crop_len = min(max_len, int(cfg.sample_rate * cfg.crop_sec))
            for i in torch.where(mask)[0].tolist():
                if max_len <= crop_len:
                    continue
                start = torch.randint(0, max_len - crop_len + 1, (1,), device=out.device).item()
                # Clone to avoid overlapping memory with out[i] which we modify in-place below.
                seg = out[i, start : start + crop_len].clone()
                out[i].zero_()
                out[i, :crop_len] = seg

    return out
