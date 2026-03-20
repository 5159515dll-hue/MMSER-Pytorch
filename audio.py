'''
Description: 
Author: Dai Lu Lu
version: 1.0
Date: 2025-12-01 22:26:44
LastEditors: Dai Lu Lu
LastEditTime: 2026-01-22 18:06:05
'''
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

os.environ.setdefault('TRANSFORMERS_NO_TORCHVISION', '1')
# 强制使用官方 Hugging Face 端点，避免镜像 SSL 异常
os.environ.setdefault('HF_ENDPOINT', 'https://huggingface.co')

try:
    import torch
    from transformers import AutoModel
    WAVLM_AVAILABLE = True
    WAVLM_IMPORT_ERROR = None
except Exception as e:
    WAVLM_AVAILABLE = False
    WAVLM_IMPORT_ERROR = e

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except Exception:
    TORCHAUDIO_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except Exception:
    SOUNDFILE_AVAILABLE = False

from experiments.motion_prosody.audio_aug import normalize_wav
from experiments.motion_prosody.prosody import ProsodyConfig, extract_prosody_features
from experiments.motion_prosody.models import ProsodyMLP

# ================== 中文支持 ==================
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 黑体
rcParams['axes.unicode_minus'] = False
# ================================================

def _load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    if TORCHAUDIO_AVAILABLE:
        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(dim=0).cpu().numpy()
        if int(sr) != int(target_sr):
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), int(sr), int(target_sr)
            ).cpu().numpy()
            sr = int(target_sr)
        return wav.astype(np.float32), int(sr)
    if SOUNDFILE_AVAILABLE:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if int(sr) != int(target_sr):
            src_len = int(x.shape[0])
            tgt_len = max(1, int(round(src_len * float(target_sr) / float(sr))))
            if src_len > 1 and tgt_len > 1:
                t_src = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
                t_tgt = np.linspace(0.0, 1.0, num=tgt_len, endpoint=False)
                x = np.interp(t_tgt, t_src, x).astype("float32", copy=False)
            sr = int(target_sr)
        return x.astype(np.float32), int(sr)
    raise ImportError("未检测到 torchaudio 或 soundfile，无法加载音频。")


# ========== 与 motion_prosody 对齐的音频加载与预处理 ==========
audio_path = Path("1.wav")  # 替换为你的wav文件路径
target_sr = 24000
max_audio_sec = 6.0

raw_wav, sample_rate = _load_audio(audio_path, target_sr)

# 截断到 max_audio_sec
max_len = int(sample_rate * max_audio_sec)
if raw_wav.shape[0] > max_len:
    raw_wav = raw_wav[:max_len]

# normalize_wav: 去直流 + RMS 归一化 + clamp
wav_tensor = torch.from_numpy(raw_wav)
wav_norm = normalize_wav(wav_tensor, target_rms=0.1).cpu().numpy()

# 时间轴
time_raw = np.arange(0, len(raw_wav)) / sample_rate
time_norm = np.arange(0, len(wav_norm)) / sample_rate

# 频谱图（用于可视化，不影响项目逻辑）
spec = torch.stft(
    torch.from_numpy(wav_norm),
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    window=torch.hann_window(1024),
    return_complex=True,
)
spec_mag = spec.abs().cpu().numpy()
spec_db = 20 * np.log10(spec_mag + 1e-10)
freqs = np.linspace(0, sample_rate / 2, spec_db.shape[0])
times = np.arange(spec_db.shape[1]) * (256 / sample_rate)

# ====== 图1：音频预处理展示（与项目一致） ======
fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

axes[0].plot(time_raw, raw_wav)
axes[0].set_xlabel("时间 (秒)")
axes[0].set_ylabel("幅度")
axes[0].set_title(f"加载后波形（采样率 {sample_rate} Hz）")

axes[1].plot(time_norm, wav_norm)
axes[1].set_xlabel("时间 (秒)")
axes[1].set_ylabel("幅度")
axes[1].set_title("normalize_wav 归一化后波形（RMS=0.1）")

mesh = axes[2].pcolormesh(times, freqs, spec_db, shading="gouraud")
axes[2].set_xlabel("时间 (秒)")
axes[2].set_ylabel("频率 (Hz)")
axes[2].set_title("归一化后频谱图（对数幅度，仅可视化）")
fig.colorbar(mesh, ax=axes[2], label="分贝 (dB)")

# ====== WavLM 特征提取 ======
if not WAVLM_AVAILABLE:
    import sys
    raise ImportError(
        'WavLM 依赖未能正确导入。\n'
        f'当前解释器: {sys.executable}\n'
        f'错误详情: {WAVLM_IMPORT_ERROR}'
    )

wavlm_model = AutoModel.from_pretrained("microsoft/wavlm-large")
wavlm_model.eval()

with torch.no_grad():
    wav_in = torch.from_numpy(wav_norm).to(torch.float32).unsqueeze(0)
    wavlm_output = wavlm_model(input_values=wav_in)

wavlm_features = wavlm_output.last_hidden_state.squeeze(0).cpu().numpy()  # (T, D)
wavlm_mean = wavlm_features.mean(axis=0)
wavlm_std = wavlm_features.std(axis=0)
wavlm_pooled = np.concatenate([wavlm_mean, wavlm_std], axis=0)

# ====== Prosody 特征提取（与项目一致） ======

prosody_cfg = ProsodyConfig(sample_rate=sample_rate)
frame_len = int(prosody_cfg.sample_rate * prosody_cfg.frame_ms / 1000.0)
hop_len = int(prosody_cfg.sample_rate * prosody_cfg.hop_ms / 1000.0)

wav_t = torch.from_numpy(wav_norm).to(torch.float32)
if wav_t.numel() < frame_len:
    wav_t = torch.nn.functional.pad(wav_t, (0, frame_len - wav_t.numel()))

frames = wav_t.unfold(0, frame_len, hop_len)  # (n_frames, frame_len)
rms = torch.sqrt(torch.mean(frames * frames, dim=-1) + 1e-9).cpu().numpy()

if TORCHAUDIO_AVAILABLE:
    f0 = torchaudio.functional.detect_pitch_frequency(
        wav_t.unsqueeze(0),
        sample_rate=prosody_cfg.sample_rate,
        frame_time=prosody_cfg.frame_ms / 1000.0,
        win_length=frame_len,
        freq_low=int(prosody_cfg.fmin),
        freq_high=int(prosody_cfg.fmax),
    ).squeeze(0).cpu().numpy()
else:
    f0 = np.zeros_like(rms, dtype=np.float32)

prosody_time = np.arange(len(rms)) * (hop_len / prosody_cfg.sample_rate)
prosody_features = np.vstack([f0[: len(rms)], rms])

# ====== 图2：WavLM 五个关键阶段的图形展示 ======
fig2, axes2 = plt.subplots(5, 1, figsize=(12, 16), constrained_layout=True)

# 1) 输入波形（统一采样率与长度）
axes2[0].plot(time_norm, wav_norm)
axes2[0].set_xlabel('时间 (秒)')
axes2[0].set_ylabel('幅度')
axes2[0].set_title('输入波形（统一采样率 + normalize_wav）')

# 2) WavLM 编码器 Transformer 堆叠（用特征图表示）
# 归一化显示：按特征维做标准化并裁剪，提升对比度
feat_mean = wavlm_features.mean(axis=0, keepdims=True)
feat_std = wavlm_features.std(axis=0, keepdims=True) + 1e-6
display_features = (wavlm_features - feat_mean) / feat_std
display_features = np.clip(display_features, -3, 3)

im1 = axes2[1].imshow(display_features.T, aspect='auto', origin='lower', cmap='magma', vmin=-3, vmax=3)
axes2[1].set_xlabel('时间步')
axes2[1].set_ylabel('特征维度')
axes2[1].set_title('WavLM 编码器 Transformer 堆叠（特征图）')
fig2.colorbar(im1, ax=axes2[1], label='特征值')

# 3) 帧级隐表示序列（取一个特征维度随时间变化）
frame_series = wavlm_features[:, 0]
axes2[2].plot(frame_series)
axes2[2].set_xlabel('时间步')
axes2[2].set_ylabel('特征值')
axes2[2].set_title('帧级隐表示序列（示例维度）')

# 4) 统计池化（均值 + 标准差）
axes2[3].plot(wavlm_mean, label='均值')
axes2[3].plot(wavlm_std, label='标准差')
axes2[3].set_xlabel('特征维度')
axes2[3].set_ylabel('数值')
axes2[3].set_title('统计池化（均值 + 标准差）')
axes2[3].legend()

# 5) 全局声学表征（拼接均值与标准差）
global_repr = wavlm_pooled
axes2[4].plot(global_repr)
axes2[4].set_xlabel('特征维度')
axes2[4].set_ylabel('数值')
axes2[4].set_title('全局声学表征（均值 + 标准差拼接）')

# ====== 图3：Prosody 特征图与曲线 ======
fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

im2 = axes3[0].imshow(
    prosody_features,
    aspect='auto',
    origin='lower',
    extent=[prosody_time[0], prosody_time[-1], 0, prosody_features.shape[0]],
)
axes3[0].set_xlabel('时间 (秒)')
axes3[0].set_ylabel('特征类型')
axes3[0].set_title('Prosody 特征图（基频 + 能量）')
axes3[0].set_yticks([0.5, 1.5])
axes3[0].set_yticklabels(['基频', '能量'])
fig3.colorbar(im2, ax=axes3[0], label='特征值')

axes3[1].plot(prosody_time, f0[:len(rms)], label='基频')
axes3[1].plot(prosody_time, rms, label='能量')
axes3[1].set_xlabel('时间 (秒)')
axes3[1].set_ylabel('数值')
axes3[1].set_title('Prosody 提取后的特征曲线')
axes3[1].legend()

# ====== 图4：Prosody 分帧处理（帧长与步长） ======
fig4, ax4 = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
im3 = ax4.imshow(frames.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
ax4.set_xlabel('帧索引')
ax4.set_ylabel('帧内采样点')
ax4.set_title(f'分帧处理（帧长={frame_len}，步长={hop_len}）')
fig4.colorbar(im3, ax=ax4, label='幅度')

# ====== 图5：Prosody 统计图（F0 / RMS / 时长） ======
fig5, axes5 = plt.subplots(4, 1, figsize=(12, 12), constrained_layout=True)

# F0 估计基频统计
f0_valid = f0[np.isfinite(f0)]
f0_valid = f0_valid[f0_valid > 0]
axes5[0].hist(f0_valid, bins=50, color='#4C72B0', alpha=0.85)
axes5[0].set_xlabel('基频 F0 (Hz)')
axes5[0].set_ylabel('帧数')
axes5[0].set_title('F0 估计基频统计（直方图）')

# 能量统计（RMS）
axes5[1].hist(rms, bins=50, color='#55A868', alpha=0.85)
axes5[1].set_xlabel('RMS 能量')
axes5[1].set_ylabel('帧数')
axes5[1].set_title('能量统计（RMS 直方图）')

# 时长统计（秒）
total_duration = len(wav_norm) / sample_rate
axes5[2].bar(
    ['总时长(秒)', '步长(秒)', '帧长(秒)'],
    [total_duration, hop_len / sample_rate, frame_len / sample_rate],
    color=['#C44E52', '#CCB974', '#64B5CD'],
)
axes5[2].set_ylabel('秒')
axes5[2].set_title('时长统计（单位：秒）')

# 帧数统计
num_frames = frames.shape[0]
axes5[3].bar(['帧数'], [num_frames], color='#8172B2')
axes5[3].set_ylabel('帧数')
axes5[3].set_title('帧数统计')

# ====== 图6：Prosody 10D 向量（与项目一致） ======
prosody_vec = extract_prosody_features(wav_t, prosody_cfg).cpu().numpy()
prosody_labels = [
    'voiced_ratio',
    'f0_mean',
    'f0_std',
    'f0_range',
    'df0_mean',
    'rms_mean',
    'rms_std',
    'rms_range',
    'drms_mean',
    'duration_sec',
]

fig6, ax6 = plt.subplots(1, 1, figsize=(12, 3), constrained_layout=True)
heat = ax6.imshow(prosody_vec.reshape(1, -1), aspect='auto', cmap='viridis')
ax6.set_yticks([])
ax6.set_xticks(np.arange(len(prosody_labels)))
ax6.set_xticklabels(prosody_labels, rotation=20, ha='right')
ax6.set_title('Prosody 10D 向量（项目定义）')
fig6.colorbar(heat, ax=ax6, label='数值')

for i, v in enumerate(prosody_vec):
    ax6.text(i, 0, f'{v:.3g}', ha='center', va='center', color='white', fontsize=9)

# ====== 图7：ProsodyMLP 编码（与模型结构一致） ======
mlp = ProsodyMLP(in_dim=10, out_dim=64)
mlp.eval()
with torch.no_grad():
    emb = mlp(torch.from_numpy(prosody_vec).unsqueeze(0)).squeeze(0).cpu().numpy()

fig7, axes7 = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
im7 = axes7[0].imshow(emb.reshape(1, -1), aspect='auto', cmap='magma')
axes7[0].set_yticks([])
axes7[0].set_xlabel('嵌入维度')
axes7[0].set_title('ProsodyMLP 输出（未训练权重）')
fig7.colorbar(im7, ax=axes7[0], label='激活值')

axes7[1].plot(emb, marker='o')
axes7[1].set_xlabel('嵌入维度')
axes7[1].set_ylabel('数值')
axes7[1].set_title('Prosody 嵌入向量（示意）')

# ====== 图8：全局声学表征 Projection（Linear） ======
proj_dim = 512
torch.manual_seed(42)
proj_wavlm = torch.nn.Linear(wavlm_pooled.shape[0], proj_dim, bias=True)
proj_wavlm.eval()
with torch.no_grad():
    wavlm_proj = proj_wavlm(torch.from_numpy(wavlm_pooled).unsqueeze(0)).squeeze(0).cpu().numpy()

fig8, ax8 = plt.subplots(1, 1, figsize=(10, 3), constrained_layout=True)
im8 = ax8.imshow(wavlm_proj.reshape(1, -1), aspect='auto', cmap='magma')
ax8.set_yticks([])
ax8.set_xlabel('投影维度')
ax8.set_title('全局声学表征 → Projection(Linear)')
fig8.colorbar(im8, ax=ax8, label='数值')

# ====== 图9：韵律嵌入 Projection（Linear） ======
torch.manual_seed(42)
proj_prosody = torch.nn.Linear(emb.shape[0], proj_dim, bias=True)
proj_prosody.eval()
with torch.no_grad():
    prosody_proj = proj_prosody(torch.from_numpy(emb).unsqueeze(0)).squeeze(0).cpu().numpy()

fig9, ax9 = plt.subplots(1, 1, figsize=(10, 3), constrained_layout=True)
im9 = ax9.imshow(prosody_proj.reshape(1, -1), aspect='auto', cmap='magma')
ax9.set_yticks([])
ax9.set_xlabel('投影维度')
ax9.set_title('韵律嵌入 → Projection(Linear)')
fig9.colorbar(im9, ax=ax9, label='数值')

# ====== 图10：特征拼接（Concatenation） ======
concat_feat = np.concatenate([wavlm_proj, prosody_proj], axis=0)
fig10, ax10 = plt.subplots(1, 1, figsize=(12, 3), constrained_layout=True)
im10 = ax10.imshow(concat_feat.reshape(1, -1), aspect='auto', cmap='viridis')
ax10.set_yticks([])
ax10.set_xlabel('拼接后维度')
ax10.set_title('特征拼接（WavLM Projection ⊕ Prosody Projection）')
fig10.colorbar(im10, ax=ax10, label='数值')

# ====== 图11：门控加权融合（Gated Fusion） ======
torch.manual_seed(42)
gate_net = torch.nn.Linear(proj_dim * 2, proj_dim, bias=True)
gate_net.eval()
with torch.no_grad():
    cat_in = torch.from_numpy(concat_feat).unsqueeze(0)
    gate = torch.sigmoid(gate_net(cat_in)).squeeze(0).cpu().numpy()
    gated_fused = gate * wavlm_proj + (1.0 - gate) * prosody_proj

fig11, axes11 = plt.subplots(2, 1, figsize=(12, 5), constrained_layout=True)
im11a = axes11[0].imshow(gate.reshape(1, -1), aspect='auto', cmap='plasma', vmin=0, vmax=1)
axes11[0].set_yticks([])
axes11[0].set_xlabel('门控维度')
axes11[0].set_title('门控权重（Sigmoid 输出）')
fig11.colorbar(im11a, ax=axes11[0], label='权重')

im11b = axes11[1].imshow(gated_fused.reshape(1, -1), aspect='auto', cmap='magma')
axes11[1].set_yticks([])
axes11[1].set_xlabel('融合后维度')
axes11[1].set_title('门控加权融合结果')
fig11.colorbar(im11b, ax=axes11[1], label='数值')

# ====== 图12：融合策略选择（示意） ======
fig12, ax12 = plt.subplots(1, 1, figsize=(6, 2), constrained_layout=True)
ax12.axis('off')
ax12.text(0.5, 0.5, 'Fusion Strategy\n(选择 concat 或 gated)', ha='center', va='center', fontsize=12)

# ====== 分支A：Concat -> Fusion MLP -> Joint Representation -> Classifier ======
torch.manual_seed(42)
fusion_mlp_a = torch.nn.Sequential(
    torch.nn.Linear(concat_feat.shape[0], proj_dim),
    torch.nn.ReLU(),
)
joint_head_a = torch.nn.Linear(proj_dim, proj_dim)
classifier_a = torch.nn.Linear(proj_dim, 7, bias=True)
fusion_mlp_a.eval()
joint_head_a.eval()
classifier_a.eval()

with torch.no_grad():
    fusion_a = fusion_mlp_a(torch.from_numpy(concat_feat).unsqueeze(0)).squeeze(0)
    z1 = joint_head_a(fusion_a).squeeze(0)
    logits_a = classifier_a(z1.unsqueeze(0)).squeeze(0)
    fusion_a = fusion_a.cpu().numpy()
    z1 = z1.cpu().numpy()
    logits_a = logits_a.cpu().numpy()

fig13, ax13 = plt.subplots(1, 1, figsize=(10, 3), constrained_layout=True)
im13 = ax13.imshow(fusion_a.reshape(1, -1), aspect='auto', cmap='magma')
ax13.set_yticks([])
ax13.set_xlabel('融合MLP输出维度')
ax13.set_title('分支A：Fusion MLP 输出')
fig13.colorbar(im13, ax=ax13, label='数值')

fig14, ax14 = plt.subplots(1, 1, figsize=(10, 3), constrained_layout=True)
im14 = ax14.imshow(z1.reshape(1, -1), aspect='auto', cmap='magma')
ax14.set_yticks([])
ax14.set_xlabel('联合表示维度')
ax14.set_title('分支A：Joint Representation')
fig14.colorbar(im14, ax=ax14, label='数值')

fig15, ax15 = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
ax15.bar(np.arange(logits_a.shape[0]), logits_a, color='#4C72B0')
ax15.set_xlabel('类别')
ax15.set_ylabel('logits')
ax15.set_title('分支A：Classifier 输出（Emotion logits）')
ax15.set_xticks(np.arange(logits_a.shape[0]))

# ====== 分支B：Gate Network -> Gate Weights -> Weighted Merge -> Fusion MLP -> Joint Representation -> Classifier ======
torch.manual_seed(42)
fusion_mlp_b = torch.nn.Sequential(
    torch.nn.Linear(gated_fused.shape[0], proj_dim),
    torch.nn.ReLU(),
)
joint_head_b = torch.nn.Linear(proj_dim, proj_dim)
classifier_b = torch.nn.Linear(proj_dim, 7, bias=True)
fusion_mlp_b.eval()
joint_head_b.eval()
classifier_b.eval()

with torch.no_grad():
    fusion_b = fusion_mlp_b(torch.from_numpy(gated_fused).unsqueeze(0)).squeeze(0)
    z2 = joint_head_b(fusion_b).squeeze(0)
    logits_b = classifier_b(z2.unsqueeze(0)).squeeze(0)
    fusion_b = fusion_b.cpu().numpy()
    z2 = z2.cpu().numpy()
    logits_b = logits_b.cpu().numpy()

fig16, ax16 = plt.subplots(1, 1, figsize=(10, 3), constrained_layout=True)
im16 = ax16.imshow(fusion_b.reshape(1, -1), aspect='auto', cmap='magma')
ax16.set_yticks([])
ax16.set_xlabel('融合MLP输出维度')
ax16.set_title('分支B：Fusion MLP 输出')
fig16.colorbar(im16, ax=ax16, label='数值')

fig17, ax17 = plt.subplots(1, 1, figsize=(10, 3), constrained_layout=True)
im17 = ax17.imshow(z2.reshape(1, -1), aspect='auto', cmap='magma')
ax17.set_yticks([])
ax17.set_xlabel('联合表示维度')
ax17.set_title('分支B：Joint Representation')
fig17.colorbar(im17, ax=ax17, label='数值')

fig18, ax18 = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
ax18.bar(np.arange(logits_b.shape[0]), logits_b, color='#55A868')
ax18.set_xlabel('类别')
ax18.set_ylabel('logits')
ax18.set_title('分支B：Classifier 输出（Emotion logits）')
ax18.set_xticks(np.arange(logits_b.shape[0]))


plt.show()