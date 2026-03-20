<!--
 * @Description: 
 * @Author: Dai Lu Lu
 * @version: 1.0
 * @Date: 2026-01-13 15:01:57
 * @LastEditors: Dai Lu Lu
 * @LastEditTime: 2026-01-21 12:11:25
-->
# Motion + Prosody (独立实验目录)

## 项目概述

目标：在不改动现有主流程的前提下，提供一个新的两段式 pipeline：
- **预解码**：从视频提取人脸区域光流 + RGB clip（默认双路），从音频提取波形并计算显式 prosody（音调/能量动态统计），可选缓存 WavLM 音频嵌入。
- **训练（默认配置）**：**声学表征（WavLM）+ 显式韵律（Prosody）并行；光流 + RGB 双表征视频；蒙古文文本（XLM-R）**。支持冻结音频/文本/视频分支用于消融。

> **平台说明**：该目录面向 CUDA/Linux 环境；macOS 上如果 `torchaudio` 二进制不匹配会报错；另外 MPS 不支持 Conv3D（视频编码器），推理时建议 `--skip-video-encoder`。

---

## 技术架构详解

### 1. 多模态特征提取（预解码阶段）

#### 1.1 视频运动特征（Optical Flow + RGB）
- **方法**：Dense Optical Flow 计算相邻帧间的光流，获得人脸区域的运动向量
- **编码**：转换为 3-channel 光流张量 (B, 3, T, H, W)
  - X/Y 方向运动分量
  - 运动幅度
- **作用**：捕捉面部微表情和肢体动作，减弱"静态身份捷径"的影响
- **参数**：
  - `--num-frames`：光流帧数（推荐 32/64，更多→更好精度但更慢）
  - `--flow-size`：光流分辨率（96/112 常用，影响细节保留）
  - `--rgb-size`：RGB 分辨率（默认 224，适配 VideoMAE）
  - `--video-repr`：缓存视频表示（flow / rgb / both，默认 both）

#### 1.2 音频特征
**a) 原始波形 + WavLM 编码（默认）**
- 使用 WavLM-Large 预训练模型（768D）
- 对原始音频 waveform 进行自监督学习表征
- 池化策略：**均值 + 标准差**联合 → 1536 维特征向量
- 优势：自监督学习已捕捉言语声学信息，不需要 Mel-spectrogram 手工设计
- 可选在预解码阶段缓存为 `audio_emb`（通过 `--audio-repr wavlm/both`），训练/推理可直接读取缓存特征

**b) 显式韵律特征（Prosody）**
- **计算维度** (10D)：
  - **音调相关**（5D）：
    - `voiced_ratio`：浊音比例（0-1）
    - `f0_mean / f0_std / f0_range`：基频均值/标差/范围（Hz）
    - `df0_mean`：基频动态变化率（Hz/frame）
  - **能量相关**（4D）：
    - `rms_mean / rms_std / rms_range`：RMS 能量均值/标差/范围
    - `drms_mean`：能量动态变化率
  - **时间信息**（1D）：
    - `duration_sec`：总时长（秒）

- **实现细节**：
  - 帧长 20ms，跳步 10ms
  - 基频检测范围 50-400 Hz（针对人类言语优化）
  - 基频提取：`torchaudio.functional.detect_pitch_frequency`（自适应算法）
  - 所有特征做 z-score 归一化

- **创新点**：相比 Mel-spectrogram，直接提取音调/能量统计量有以下优势：
  - 特征维度极低（10D vs 128D Mel），计算快
  - 语义更清晰（可解释性强）
  - 避免了频谱数据中隐含的说话人身份信息

#### 1.3 文本特征
- **模型**：XLM-R（xlm-roberta-large，默认）
- **输入**：蒙古文本文本
- **输出**：CLS token 表征（768D）
- **特性**：多语言预训练，对低资源语言有较好支持

### 2. 融合模型架构（FusionClassifier）

```
输入分支：
├─ Flow (B,3,T,H,W) ──→ FlowVideoEncoder ──→ 256D
├─ RGB (B,T,3,H,W) ──→ VideoMAE ──→ 768D
├─ Audio (B,L) ──→ WavLMEncoder ──→ 1536D
├─ Prosody (B,10) ──→ ProsodyMLP ──→ 64D
└─ Text (B,128) ──→ XLMRTextEncoder ──→ 768D
                     ↓
         融合层 (门控位移/可选Concat) ──→ 隐层(1024D) ──→ 情感分类(7)
                                          ├─→ 强度回归(1) [可选]
```

**模型设计要点**：
- **各分支独立编码**：便于模块化、冻结与灵活组合
- **默认门控位移融合（gated_text）**：用非语言模态生成“位移向量”，
  对文本表征做残差微调，缓解强文本主导导致的模态贡献失衡
  - **可选 concat 融合**：用于对比实验或快速 baseline
  
- **可选强度回归**：支持同时做分类 + 连续情感强度预测（多任务学习）
- **梯度流控制**：可独立冻结各分支编码器，稳定训练或节省显存

---

### 3. 数据增强与鲁棒性（AudioAugConfig）

**音频增强**（可选，通过 `--audio-aug` 启用）：
- **随机增益**（概率 50%）：±6dB 动态范围增强
- **随机速率变换**（概率 30%）：±5% 重采样（减弱说话速度线索）
- **随机裁剪**（概率 50%）：4s 片段（减弱长期结构线索 & 说话人特征）

> 说明：若预解码缓存了 `audio_emb`（`--audio-repr wavlm/both`），训练时默认优先使用缓存特征，`--audio-aug` 将被忽略。

**prosody 处理**：
- 默认从增强音频重新计算 prosody（`--recompute-prosody-on-aug`）
- 可关闭以保持缓存一致性（`--no-recompute-prosody-on-aug`）

---

## 创新点与特色

### 1. **显式韵律特征**
- 不依赖 Mel-spectrogram 这类"黑盒"频谱，而是提取音调/能量动态的**清晰语义量**
- 显著降低维度（10D vs 128D），加快计算
- **可解释性强**：直接对应言语学中的超分音特征

### 2. **光流 + RGB + WavLM 的多模态融合**
- 视频运动直接编码（3D CNN），不依赖骨架检测或 MediaPipe
- WavLM 自监督表征避免了监督 ASR 对方言/口音的偏差
- 两者结合可互补减弱说话人身份信息（运动 vs 音频）

### 3. **缓存预处理 Pipeline**
- 预解码将原始视频/音频转为固定格式 PyTorch tensors
- 避免训练时反复解码（快速，I/O 友好）
- 支持断点续传与分布式预处理（shard-based）

### 4. **情感强度回归（可选多任务）**
- 在情感分类基础上，额外预测 0-5 连续强度值
- XLSX 中强度列会自动检测（支持多种表头约定）
- 缺失强度标注自动 mask，不影响训练

### 5. **模块化冻结机制**
- 可独立冻结 `--freeze-audio/video/prosody/text` 进行迁移学习或稳定微调
- 支持情感强度损失权重调节（`--intensity-weight`）

---

## 未来创新方向

### 1. **交叉模态注意机制（Cross-Modal Attention）**
- 当前使用简单 concat，可升级为：
  - 多头交叉注意：让各模态间互相关注，动态融合权重
  - 自适应模态融合权重学习
- **期待效果**：更好地利用多模态的互补性

### 2. **对比学习预训练（Contrastive Pre-training）**
- 在大量无标注多模态数据上预训练融合编码器
- 通过对比学习强化多模态表征对齐
- 后续微调需要的标注数据可显著减少

### 3. **情感生成模型**
- 从融合特征生成回对应的音频/视频
- 作为：
  - 特征质量检验（重建损失）
  - 无监督表征学习的自监督任务

### 4. **时间序列建模**
- 当前按单样本处理；可改进为：
  - RNN/Transformer 对整段视频的时间动态建模
  - 捕捉情感的"上升/下降/平台"阶段
  - 更好利用连续性线索

### 5. **说话人无关性约束（Speaker-Agnostic Regularization）**
- 显式约束：情感表征与说话人 embedding 的正交性
- 对抗学习：加入说话人判别器，使模型难以利用身份线索
- **针对现有数据标签≈说话人的问题**

### 6. **多语言迁移学习**
- 当前 XLM-R 支持多语言，可：
  - 在高资源情感语料（如英文 IEMOCAP）上预训练
  - 迁移到蒙古文/其他低资源语言
  - 跨语言情感表征对齐

### 7. **动态情感强度曲线预测**
- 从固定序列预测单一强度值，升级为：
  - 预测强度随时间的变化曲线
  - 捕捉"开始弱 → 中间强 → 结尾弱"的动态
  - 更接近真实对话的情感演进

---

## 核心模块说明

### 1. predecode_motion_audio.py - 多模态预解码
**职责**：
- 从源 XLSX 逐条读取数据（视频路径、标签、文本、强度等）
- 对每条数据：
  - 提取光流（Dense Optical Flow）并编码为 (3, T, H, W) 张量
  - 提取人脸区域 RGB clip（T, 3, H, W）用于 VideoMAE
  - 加载音频波形并计算 prosody 特征向量
  - 从 XLM-R tokenizer 获得文本 token IDs
- 批量输出为 `.pt` shards（PyTorch 序列化格式），便于快速加载

**输入**：XLSX 文件 + 视频音频文件  
**输出**：缓存目录（多个 .pt 文件）

### 2. train_motion_audio.py - 多任务融合训练
**职责**：
- 从缓存加载预处理好的多模态数据
- 初始化 FusionClassifier（5 个独立编码器：flow + RGB + audio + prosody + text）
- 支持两种训练模式：
  - **纯分类**（7 类情感）
  - **分类 + 强度回归**（多任务，需 `--use-intensity`）
- 梯度下降优化，可冻结指定分支以稳定或省显存
- 周期保存 checkpoint，记录训练曲线

**输入**：缓存数据集  
**输出**：checkpoint（best.pt）+ 指标文件（metrics.json）

### 3. batch_inference_motion_prosody.py - 批量推理
**职责**：
- 从缓存或 XLSX 读取测试数据
- 加载训练好的 checkpoint
- 对每条样本：
  - 前向传播，得到情感分类 logits 和强度预测（若有）
  - 计算准确率、F1 分数等指标
  - 若有 GT 标注，计算 MSE/MAE/CCC（强度）
- 输出 JSONL 格式（每行一个 JSON，便于后续分析）

**输入**：缓存或 XLSX + 训练好的 checkpoint  
**输出**：inference_results.jsonl + metrics.json

---

本目录 3 个核心命令：

- 数据集预处理：`predecode_motion_audio.py`（生成缓存 shards）
- 模型训练：`train_motion_audio.py`（分类 / 分类+强度回归）
- 批量推理：`batch_inference_motion_prosody.py`（从 XLSX 或从缓存）

---

## 快速开始（3 步）

1) 预解码（生成缓存）

```bash
python3 experiments/motion_prosody/predecode_motion_audio.py \
  --data-root databases --xlsx databases/video_databases.xlsx \
  --num-frames 64 --flow-size 112 --rgb-size 224 --video-repr both \
  --audio-repr wavlm --audio-model microsoft/wavlm-large \
  --sample-rate 24000 --max-audio-sec 6.0 \
  --shard-size 100 \
  --num-workers 2 --mp-start-method forkserver --mp-chunksize 32 \
  --print-result all --result-every 1 --log-every 1 \
  --output outputs/predecoded/motion_prosody_f64_sr24000
```

2) 训练（只做分类）

```bash
python3 experiments/motion_prosody/train_motion_audio.py \
  --cached-dataset outputs/predecoded/motion_prosody_f64_sr24000 \
  --epochs 50 --batch-size 32 --num-workers 16 \
  --pin-memory --persistent-workers --prefetch-factor 4 \
  --lr 1e-4 --output-dir outputs/motion_prosody \
  --video-backbone dual --audio-model microsoft/wavlm-large \
  --text-model xlm-roberta-large --max-text-len 128
```

3) 推理（推荐：从缓存推理）

```bash
python3 experiments/motion_prosody/batch_inference_motion_prosody.py \
  --cached-dataset outputs/predecoded/motion_prosody_f64_sr24000 \
  --checkpoint outputs/motion_prosody/checkpoints/best.pt \
  --output outputs/motion_prosody/inference_results.jsonl \
  --video-backbone dual --audio-model microsoft/wavlm-large \
  --text-model xlm-roberta-large --max-text-len 128
```

---

## 数据与 XLSX 约定（强度列在哪里）

- XLSX 至少需要列：`序号 / 情感类别 / 蒙文`。
- 情感强度列（可选）：支持表头 `情感强度/强度/intensity/Intensity`；也支持“蒙文后面那一列”的位置模式 `[序号, 蒙文, 情感强度, 中文, 情感类别]`。
- 预解码会把强度写入缓存 sample 的 `intensity`；缺失/非数值会被当作无标注（mask 掉）。

---

## 1) 预解码（生成缓存）

### 核心特性

预解码将原始视频和音频转换为固定格式的 PyTorch 张量，包括：
1. **光流张量**：(3, T, H, W) - 从连续帧对计算光流
2. **RGB 张量**：(T, 3, H, W) - 人脸区域 RGB clip（用于 VideoMAE）
3. **音频波形**：(L,) - 原始采样音频，供 WavLM 编码器使用（若 `--audio-repr raw/both`）
4. **WavLM 嵌入**：(1536,) - 预解码缓存的音频特征（若 `--audio-repr wavlm/both`）
5. **Prosody 特征**：(10,) - 预计算的韵律统计（音调、能量、时长等）
6. **文本 IDs**：(max_text_len,) - XLM-R tokenizer 的 token IDs
7. **标签与元数据**：情感类别、强度（若有）、说话人 ID 等

### 为什么需要预解码？

- **性能**：训练时无需反复解码视频（极慢），直接读取 tensor（毫秒级）
- **I/O 优化**：Sharding 使数据并行加载稳定
- **可重复性**：预处理结果一致，避免随机解码差异
- **分布式友好**：支持多进程预处理（`--num-workers`）

### 命令与参数解读

你现在这条命令 **不需要为了"启用/不启用情感强度"而改任何参数**：

- XLSX 有强度列：自动写入 `intensity`
- XLSX 没强度列：自动为缺失（NaN），后续训练可仍然只做分类

**常用可调参数与效果**：

| 参数 | 默认值 | 含义 | 调优建议 |
|-----|-------|------|--------|
| `--num-frames` | 64 | 光流帧数 | 32/64 常用；大→更精细但更慢 |
| `--flow-size` | 112 | 光流分辨率 | 96/112 常用；大→细节多但更慢 |
| `--rgb-size` | 224 | RGB 分辨率 | 224 为 VideoMAE 标准 |
| `--video-repr` | both | 预解码视频表示 | flow / rgb / both |
| `--audio-repr` | raw | 预解码音频表示 | raw / wavlm / both |
| `--audio-model` | microsoft/wavlm-large | 预解码音频模型 | HF WavLM 模型名 |
| `--sample-rate` | 24000 | 音频采样率 (Hz) | **需与后续训练/推理一致** |
| `--max-audio-sec` | 6.0 | 最长音频秒数 | 大→缓存大；根据数据实际长度设定 |
| `--shard-size` | 100 | 每个 shard 的样本数 | 大→单文件大，I/O 少；小→读写多 |
| `--num-workers` | 2 | 预解码进程数 | 大→快但吃 CPU；根据机器核数调 |
| `--mp-start-method` | spawn | 多进程启动方式 | Linux: `forkserver`；macOS/Windows: `spawn` |
| `--pool-retry` | 0 | worker 崩溃后的重试次数 | >0 可在崩溃后自动重建进程池 |
| `--pool-retry-delay` | 5.0 | 重试等待秒数 | 崩溃后等待再重试 |
| `--print-result` | off | 输出详度 | `all` 便于调试；`off` 加快速度 |

**范例**：

```bash
# 快速预解码（低分辨率、少帧）
python3 experiments/motion_prosody/predecode_motion_audio.py \
  --data-root databases --xlsx databases/video_databases.xlsx \
  --num-frames 32 --flow-size 96 --sample-rate 24000 --max-audio-sec 4.0 \
  --audio-repr wavlm --audio-model microsoft/wavlm-large \
  --shard-size 50 --num-workers 4 \
  --output outputs/predecoded/motion_prosody_fast

# 高精度预解码（更多帧、更大分辨率）
python3 experiments/motion_prosody/predecode_motion_audio.py \
  --data-root databases --xlsx databases/video_databases.xlsx \
  --num-frames 64 --flow-size 112 --sample-rate 24000 --max-audio-sec 6.0 \
  --audio-repr wavlm --audio-model microsoft/wavlm-large \
  --shard-size 100 --num-workers 8 \
  --output outputs/predecoded/motion_prosody_hd
```

---

## 2) 模型训练（分类 / 分类+强度回归）

### 训练流程

1. **加载缓存数据**：从 shard 读入所有样本到内存（或使用 streaming 加载）
2. **数据划分**：Train/Val 分割（随机或顺序）
3. **优化循环**：
  - 前向：各编码器独立编码，默认门控位移融合（或 concat），分类头输出 logits
   - 损失：交叉熵（分类）+ 可选 MSE/Huber（强度回归）
   - 反向 + 优化器更新
4. **验证评估**：计算 accuracy、F1、强度 MSE/MAE/CCC
5. **Checkpoint 保存**：记录最佳模型

### 模式 1：纯分类

```bash
python3 experiments/motion_prosody/train_motion_audio.py \
  --cached-dataset outputs/predecoded/motion_prosody_f64_sr24000 \
  --epochs 50 --batch-size 32 --num-workers 16 \
  --pin-memory --persistent-workers --prefetch-factor 4 \
  --lr 1e-4 --output-dir outputs/motion_prosody \
  --text-model xlm-roberta-large --max-text-len 128
```

**输出**：
- `outputs/motion_prosody/checkpoints/best.pt`（最佳 checkpoint）
- `outputs/motion_prosody/metrics.json`（训练曲线）

### 模式 2：分类 + 情感强度回归（多任务）

```bash
python3 experiments/motion_prosody/train_motion_audio.py \
  --cached-dataset outputs/predecoded/motion_prosody_f64_sr24000 \
  --epochs 50 --batch-size 32 --num-workers 16 \
  --pin-memory --persistent-workers --prefetch-factor 4 \
  --lr 1e-4 --output-dir outputs/motion_prosody \
  --video-backbone dual --audio-model microsoft/wavlm-large \
  --text-model xlm-roberta-large --max-text-len 128 \
  --use-intensity --intensity-loss mse --intensity-weight 1.0
```

**强度回归参数**：

| 参数 | 默认值 | 含义 |
|-----|-------|------|
| `--use-intensity` | - | 启用强度回归分支 |
| `--intensity-loss` | mse | 损失类型：`mse` \| `huber` |
| `--intensity-weight` | 1.0 | 强度损失权重（相对于分类损失） |

### 关键训练参数

| 参数 | 默认值 | 含义 | 调优建议 |
|-----|-------|------|--------|
| `--batch-size` | 8 | batch 大小 | 32/64 常用；大→更稳定但更吃显存 |
| `--lr` | 3e-4 | 学习率 | 冻结分支时可用 1e-4；微调用 1e-5 |
| `--weight-decay` | 1e-4 | L2 正则 | 防止过拟合 |
| `--epochs` | 50 | 训练轮数 | 根据 Val loss 早停（通常 30-50） |
| `--train-split` | 0.8 | 训练集比例 | 80/20 常用 |
| `--split-mode` | random | 切分方式 | `random`：随机；`sequential`：按顺序 |

### 门控位移融合（gated_text）超参说明

**默认融合模式**：`--fusion-mode gated_text`

该策略用非语言模态（视频/音频/韵律）生成“位移向量”$\Delta$和门控$g$，对文本表示做残差微调：

$$
\Delta = f_{\Delta}(v, a, p)\qquad
g = \sigma\left(\frac{f_g(v, a, p)}{T}\right)\qquad
t' = t + (\alpha \cdot g) \odot (\beta \cdot \Delta)
$$

- $T$: gate temperature
- $\alpha$: gate scale
- $\beta$: delta scale

**参数与含义**：

| 参数 | 默认 | 作用 | 建议范围 |
|---|---:|---|---|
| `--fusion-mode` | gated_text | 融合模式（gated_text / concat） | gated_text |
| `--gate-temperature` | 1.0 | 门控温度（越小越“硬”） | 0.5~2.0 |
| `--gate-scale` | 1.0 | 门控强度系数 | 0.5~2.0 |
| `--delta-scale` | 1.0 | 位移幅度系数 | 0.5~2.0 |
| `--modality-dropout` | 0.1 | 非文本模态随机丢弃概率 | 0.0~0.3 |

**使用示例**：

```bash
# 更“硬”的门控（提升非语言影响）
python3 experiments/motion_prosody/train_motion_audio.py \
  --cached-dataset outputs/predecoded/motion_prosody_f64_sr24000 \
  --video-backbone dual --audio-model microsoft/wavlm-large \
  --text-model xlm-roberta-large --fusion-mode gated_text \
  --gate-temperature 0.7 --gate-scale 1.2 --delta-scale 1.1 \
  --modality-dropout 0.1

# 回退到传统 concat 融合
python3 experiments/motion_prosody/train_motion_audio.py \
  --cached-dataset outputs/predecoded/motion_prosody_f64_sr24000 \
  --fusion-mode concat --modality-dropout 0.0
```

### 冻结与迁移学习

减少参数更新、稳定训练、节省显存：

```bash
# 冻结视频和音频编码器，只训练融合头和 prosody
python3 experiments/motion_prosody/train_motion_audio.py \
  --cached-dataset outputs/predecoded/motion_prosody_f64_sr24000 \
  --freeze-video --freeze-audio \
  --epochs 50 --batch-size 32 --num-workers 16 \
  --lr 1e-3 --output-dir outputs/motion_prosody_finetune \
  --text-model xlm-roberta-large --max-text-len 128
```

**冻结选项**：
- `--freeze-audio`：冻结 WavLM（保留预训练）
- `--freeze-video`：冻结 3D CNN 光流编码器
- `--freeze-prosody`：冻结 prosody MLP
- `--freeze-text`：冻结 XLM-R

### 音频增强（提升鲁棒性）

```bash
python3 experiments/motion_prosody/train_motion_audio.py \
  --cached-dataset outputs/predecoded/motion_prosody_f64_sr24000 \
  --audio-aug \
  --epochs 50 --batch-size 32 --num-workers 16 \
  --lr 1e-4 --output-dir outputs/motion_prosody_aug \
  --text-model xlm-roberta-large --max-text-len 128
```

**音频增强包括**：
- 随机增益（±6 dB）：模拟录音条件变化
- 随机速率（±5%）：模拟说话速度变化
- 随机裁剪（4s）：减弱长期结构和说话人线索

**选项**：
- `--no-recompute-prosody-on-aug`：使用原始 prosody（更快，但 prosody 与增强音频不完全对齐）

### 性能优化选项（CUDA 特定）

```bash
python3 experiments/motion_prosody/train_motion_audio.py \
  --cached-dataset outputs/predecoded/motion_prosody_f64_sr24000 \
  --amp \
  --pin-memory --persistent-workers --prefetch-factor 4 \
  --epochs 50 --batch-size 32 --num-workers 16 \
  --lr 1e-4 --output-dir outputs/motion_prosody_fast \
  --text-model xlm-roberta-large --max-text-len 128
```

**优化参数**：
- `--amp`：混合精度训练（仅 CUDA；大幅加速）
- `--pin-memory`：数据固定在 GPU 内存（加快 H2D 传输）
- `--persistent-workers`：工作进程跨 epoch 复用（减少启动开销）
- `--prefetch-factor`：预取的 batch 数量（提高 I/O 吞吐）

---

## 3) 推理（从 XLSX 或从缓存）

### 推理模式对比

| 来源 | 速度 | 依赖 | 适用场景 |
|-----|------|------|--------|
| 缓存（推荐） | ⭐⭐⭐ 快 | 无 OpenCV | 批量测试、评估 |
| XLSX 在线解码 | ⭐ 慢 | 需 OpenCV | 单次新数据、实时演示 |

### 推理方式 1：从缓存推理（推荐）

最快、最稳定的方式：

```bash
python3 experiments/motion_prosody/batch_inference_motion_prosody.py \
  --cached-dataset outputs/predecoded/motion_prosody_f64_sr24000 \
  --checkpoint outputs/motion_prosody/checkpoints/best.pt \
  --output outputs/motion_prosody/inference_results.jsonl \
  --video-backbone dual --audio-model microsoft/wavlm-large \
  --text-model xlm-roberta-large --max-text-len 128
```

**流程**：
1. 加载缓存中所有样本
2. 遍历，逐条前向
3. 输出 JSONL（每行一条预测）

**优势**：
- 预解码已固定特征，无随机性
- 速度快（毫秒/条）
- 可复现

### 推理方式 2：从 XLSX 在线推理

在线从视频/音频解码（较慢）：

```bash
python3 experiments/motion_prosody/batch_inference_motion_prosody.py \
  --data-root databases --xlsx databases/video_databases.xlsx \
  --checkpoint outputs/motion_prosody/checkpoints/best.pt \
  --output outputs/motion_prosody/inference_results_online.jsonl \
  --max-text-len 128
```

**流程**：
1. 读 XLSX 获得视频/音频路径、文本等
2. 实时计算光流、prosody（同预解码逻辑）
3. 输出 JSONL

**劣势**：
- 逐条解码视频（慢，通常 10-50 条/分钟）
- 需要 OpenCV
- 存在解码随机性（可通过 `--seed` 固定）

### macOS/MPS 兼容性

MPS（Metal Performance Shaders）不支持 3D Conv（视频编码器）。推荐跳过：

```bash
python3 experiments/motion_prosody/batch_inference_motion_prosody.py \
  --cached-dataset outputs/predecoded/motion_prosody_f64_sr24000 \
  --checkpoint outputs/motion_prosody/checkpoints/best.pt \
  --output outputs/motion_prosody/inference_results_nomps.jsonl \
  --skip-video-encoder \
  --max-text-len 128
```

**效果**：
- 不使用视频特征，仅用 audio + prosody + text
- 精度略低，但能在 MPS 上运行

### 输出格式（JSONL）

每行一条 JSON：

```json
{
  "stem": "样本ID",
  "label": 2,
  "pred_label": 2,
  "logits": [0.1, 0.2, ..., 0.05],
  "confidence": 0.87,
  "pred_intensity": 3.2,
  "intensity_gt": 3.0,
  "mn": "蒙文文本"
}
```

**字段说明**：
- `stem`：样本唯一标识符（通常是视频文件名）
- `label`：GT 情感标签（0-6）
- `pred_label`：预测情感标签
- `logits`：7 个情感的 logit 分数
- `confidence`：softmax 后最高概率
- `pred_intensity`：预测强度（0-5，仅当训练时启用 `--use-intensity`）
- `intensity_gt`：GT 强度（仅当有标注）
- `mn`：蒙古文文本

### 评估指标（JSONL 末尾汇总）

推理完成后，JSONL 末尾会追加一条汇总行：

```json
{
  "_type": "metrics",
  "accuracy": 0.75,
  "f1_macro": 0.72,
  "f1_weighted": 0.74,
  "confusion_matrix": [[30, 2, ...], ...],
  "per_class_f1": [0.70, 0.75, 0.72, 0.78, 0.68, 0.71, 0.73],
  "intensity_metrics": {
    "mse": 0.45,
    "mae": 0.38,
    "ccc": 0.82
  }
}
```

**指标解释**：
- `accuracy`：整体准确率
- `f1_macro`：7 类情感的 F1 均值（不考虑样本不均衡）
- `f1_weighted`：加权 F1（按类别样本数加权）
- `per_class_f1`：各情感类别的 F1 分数
- `intensity_metrics`（若有）：
  - `mse`：均方误差
  - `mae`：平均绝对误差
  - `ccc`：Concordance Correlation Coefficient（相关性）

## 你应该怎么解读结果

在你的数据设定（标签≈说话人）下，这套 pipeline 的意义主要是：
- 通过运动特征/韵律特征弱化“静态身份捷径”，让结果更接近“情绪动态”。
- 但它**不能从根本上解决 speaker-confound**；要获得可信的情绪识别，需要每类多说话人覆盖。
### 常见问题排查

**Q1：为什么预解码很慢？**  
- A：视频解码和光流计算本身计算量大。建议：
  - 增加 `--num-workers`（多进程）
  - 降低 `--flow-size` 或 `--num-frames`
  - 减少 `--max-audio-sec`

**Q2：训练时显存不足？**  
- A：尝试：
  - 减少 `--batch-size`
  - 启用 `--freeze-video/audio/text`（减少需要梯度的参数）
  - 在 CUDA 上启用 `--amp`（混合精度）

**Q3：缓存文件很大？**  
- A：预期行为。光流(float32)、音频(float32)、prosody 等都占空间。可：
  - 减小 `--shard-size` 但这会增加文件数量
  - 降低 `--flow-size` / `--num-frames`

**Q4：在 macOS 上 torchaudio 导入失败？**  
- A：torchaudio 在 macOS 的二进制包通常有版本不匹配问题。解决：
  ```bash
  python3 -m pip uninstall -y torchaudio torchvision torch
  python3 -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
  ```

**Q5：在 macOS/MPS 上推理时出现 Conv3D 错误？**  
- A：MPS 不支持 3D Conv。使用：
  ```bash
  --skip-video-encoder
  ```
  这会禁用视频编码器，仅用音频/prosody/文本。

### 调试建议（命令级）

- **预解码调试**：
  - 使用 `--print-result all --result-every 1 --log-every 1` 查看每条样本是否成功
  - 若 OpenCV 慢或崩溃，降低 `--num-frames` / `--flow-size`，或用 `--video-repr rgb` 单独测试

- **训练调试**：
  - 先用小批量：`--batch-size 4`，确认能跑通
  - 使用 `--freeze-video --freeze-audio` 快速验证融合头
  - 若 loss 不下降，先切 `--fusion-mode concat` 做 baseline

- **推理调试**：
  - 优先使用缓存模式（`--cached-dataset`）减少解码变量
  - 如果报 `rgb_missing/flow_missing`，检查预解码是否 `--video-repr both`

---

## 完整工作流示例

### 场景：新项目从零开始
unset HF_ENDPOINT
export HF_TOKEN=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
export HF_ENDPOINT=https://hf-mirror.com
**第 1 步：准备数据**
- 将视频放在 `databases/emotions/` 下
- 创建 XLSX：`databases/video_databases.xlsx`
- 列包括：`序号, 情感类别, 蒙文, [可选强度]`

**第 2 步：预解码（启用光流 + RGB + prosody + WavLM 缓存）**
```bash
FORCE_FACE_CPU=1 HF_ENDPOINT=$HF_ENDPOINT HF_TOKEN=$HF_TOKEN \
python3 experiments/motion_prosody/predecode_motion_audio.py \
  --data-root databases --xlsx databases/video_databases.xlsx \
  --num-frames 64 --flow-size 112 --rgb-size 224 --video-repr both \
  --audio-repr both --audio-model microsoft/wavlm-large \
  --sample-rate 24000 --max-audio-sec 6.0 \
  --shard-size 25 \
  --pool-retry 100 --pool-retry-delay 3.0 \
  --num-workers 24 --mp-start-method spawn \
  --print-result all --result-every 5 \
  --output outputs/predecoded/my_project_v1
```
> 说明：该命令完成“光流 + RGB + prosody + WavLM 嵌入（audio_emb）”的预处理缓存；
> 训练/推理将优先读取 `audio_emb`，无需再跑 WavLM 编码。

**第 3 步：训练（默认：WavLM + Prosody 并行，双路视频，蒙古文文本）**
```bash
python3 experiments/motion_prosody/train_motion_audio.py \
  --cached-dataset outputs/predecoded/my_project_v1 \
  --epochs 50 --batch-size 32 --num-workers 16 \
  --pin-memory --persistent-workers --prefetch-factor 4 \
  --lr 1e-4 --output-dir outputs/my_project_v1_model \
  --video-backbone dual --audio-model microsoft/wavlm-large \
  --text-model xlm-roberta-large --max-text-len 128
```

**可选：冻结分支做消融**
```bash
python3 experiments/motion_prosody/train_motion_audio.py \
  --cached-dataset outputs/predecoded/my_project_v1 \
  --freeze-audio --freeze-video --freeze-text \
  --epochs 20 --batch-size 32 --lr 1e-3 \
  --output-dir outputs/my_project_v1_ablation \
  --video-backbone dual --audio-model microsoft/wavlm-large \
  --text-model xlm-roberta-large --max-text-len 128
```

**第 4 步：推理与评估（默认：从原始视频/音频/文本在线解码）**
```bash
python3 experiments/motion_prosody/batch_inference_motion_prosody.py \
  --data-root databases --xlsx databases/video_databases.xlsx \
  --checkpoint outputs/my_project_v1_model/checkpoints/best.pt \
  --output outputs/my_project_v1_model/results.jsonl \
  --video-backbone dual --audio-model microsoft/wavlm-large \
  --text-model xlm-roberta-large --max-text-len 128
```

**第 5 步：分析结果**
- 查看 `results.jsonl` 末尾的汇总指标
- 分析 confusion matrix，找出哪些情感容易混淆
- 若精度不理想，参考"常见问题"尝试调参

### 场景：继续迭代（已有 baseline 模型）

**尝试冻结音视频分支，只微调融合层**：
```bash
python3 experiments/motion_prosody/train_motion_audio.py \
  --cached-dataset outputs/predecoded/my_project_v1 \
  --freeze-audio --freeze-video \
  --epochs 20 --batch-size 32 --lr 1e-3 \
  --output-dir outputs/my_project_v1_finetune \
  --text-model xlm-roberta-large --max-text-len 128
```

**尝试启用音频增强与强度回归**：
```bash
python3 experiments/motion_prosody/train_motion_audio.py \
  --cached-dataset outputs/predecoded/my_project_v1 \
  --epochs 50 --batch-size 32 --lr 1e-4 \
  --audio-aug \
  --use-intensity --intensity-loss mse --intensity-weight 0.5 \
  --output-dir outputs/my_project_v1_intensity \
  --text-model xlm-roberta-large --max-text-len 128
```

---

## 依赖与环境

### 必需包

```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0 (或 >=2.2.2 for better macOS support)
transformers>=4.37.2
sentencepiece
opencv-python
numpy
pandas
tqdm
```

### 安装命令（推荐）

```bash
# 基础 PyTorch（调整 CUDA 版本/CPU-only）
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 macOS/CPU
python3 -m pip install torch torchvision torchaudio

# 其他包
python3 -m pip install transformers sentencepiece opencv-python pandas tqdm
```

### 已知兼容版本

- PyTorch 2.2.2 + torchvision 0.17.2 + torchaudio 2.2.2（CUDA 11.8 或 CPU）
- transformers 4.37+
- Python 3.9-3.11

---

## 性能基准（参考）

在 NVIDIA A100 上的典型时间（VideoMAE + WavLM + XLM-R，双路视频）：

| 操作 | 样本数 | 耗时 | 速度 |
|-----|-------|------|------|
| 预解码（64 帧，112 光流 / 224 RGB）| 1000 | ~2-3h | 5-8 条/秒 |
| 训练 1 epoch（batch 32） | 800 | ~1-2 分钟 | 取决于样本长度 |
| 推理（缓存） | 200 | ~10-20s | 10-20 条/秒 |
| 推理（XLSX 在线） | 200 | ~5-10 分钟 | 0.3-0.7 条/秒 |

**影响因素**：
- CPU 核数（预解码并行度）
- GPU 显存（batch size）
- 磁盘速度（I/O）
- 网络延迟（模型下载）
- 预训练模型大小（VideoMAE/WavLM/XLM-R 会显著增加显存与推理耗时）
