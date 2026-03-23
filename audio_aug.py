"""音频增强与归一化工具。

这个模块的职责很单一：
1. 把原始波形统一到稳定的幅值尺度，减少录音音量差异带来的训练噪声。
2. 在训练时做轻量随机扰动，弱化说话速度、响度和长时上下文等捷径特征。

之所以把这些逻辑单独拆出来，是为了让训练入口只关心“何时增强”，
而不需要重复理解“如何增强”。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AudioAugConfig:
    """控制音频增强强度的配置对象。

    各字段都尽量保持“可解释的物理量”：
    - `gain_db` 用分贝表示响度扰动范围；
    - `speed_range` 表示相对变速比例；
    - `crop_sec` 用秒数表示随机裁剪长度。
    """

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
    """把波形标准化到统一的 RMS 幅值。

    这里做的不是“峰值归一化”，而是 RMS 归一化。原因是 RMS 更接近
    人耳感知到的平均能量，也更适合作为语音模型输入前的稳定化步骤。

    处理步骤：
    1. 转成 `float32`，避免整数波形或混合精度引入数值问题。
    2. 可选地移除直流分量（减去均值），避免录音设备偏置影响能量统计。
    3. 计算 RMS：`sqrt(mean(x^2) + eps)`。
    4. 按比例缩放到 `target_rms`。
    5. 最后做截断，避免极端峰值破坏后续编码器稳定性。

    Args:
        wav: 单条 `(L,)` 或批量 `(B, L)` 波形。
        target_rms: 目标均方根幅值。
        remove_dc: 是否去掉直流偏置。
        clamp: 最大绝对幅值裁剪阈值。
        eps: 避免 RMS 分母为 0 的数值稳定项。

    Returns:
        归一化后的 `float32` 波形，形状与输入一致。
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
    """对批量波形做轻量随机增强。

    当前增强策略偏保守，目标不是合成激进的新语音，而是轻微扰动那些
    容易让模型偷懒的浅层线索：
    - 随机增益：减少“录音更响 = 某一类情绪”的偶然相关。
    - 随机变速：减弱说话速度和个体节奏的捷径。
    - 随机裁剪：降低模型记住整段语音长时结构的倾向。

    注意这里所有增强都保持输出长度不变，便于后续 batch 拼接。

    Args:
        wav: 形状 `(B, L)` 的批量波形。
        cfg: 增强配置。

    Returns:
        增强后的波形副本，不会原地修改输入张量。
    """
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
                    # 先把采样率改成 sr*factor，再重采样回 sr。
                    # 这相当于时间轴被压缩/拉伸，从而实现轻量变速。
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
                # 先截取一个随机片段，再写回到长度固定的缓冲区前半段。
                # 这相当于“随机保留局部语音 + 其余补零”，能减弱长时说话人线索。
                seg = out[i, start : start + crop_len].clone()
                out[i].zero_()
                out[i, :crop_len] = seg

    return out
