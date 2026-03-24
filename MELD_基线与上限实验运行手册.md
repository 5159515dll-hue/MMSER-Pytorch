# MELD GPU 基线与上限实验运行手册

本文档整理 `MMSER-Pytorch` 在 `MELD` 数据集上的四组 **GPU 流式实验** 命令与解释。

这版手册只保留当前主线实现对应的 GPU 路线：

- 训练与推理统一走 `gpu_stream`
- 中间不再写 `raw/feature cache` shard
- 共享前置步骤只保留 manifest、音频 sidecar 和模型下载

有两个必须明确的实现事实：

1. 当前 `gpu_stream` 视频预处理使用的是 **中心裁剪 + GPU resize**，不是旧 CPU 路线的 Haar 人脸裁剪。
2. 当前第四组实验里的动态分支使用的是 **`torch_motion` GPU 动态分支**，不是旧 CPU 路线的 Farneback optical flow，因此不能宣称与旧 CPU 光流数值等价。

如果你要写论文或报告，应该把这四组结果理解为：

- 当前仓库 **GPU 版 MELD 实验口径**
- 不是旧 CPU 缓存式实验的逐点复现

## 四组实验总览

| 实验 | 推荐命名 | 模态/分支 | 冻结策略 | 主要证明内容 | 推荐服务器 |
| --- | --- | --- | --- | --- | --- |
| 实验一 | `MMSER-Pytorch-practical-gpu (A+T+P, zero-video-input, frozen)` | 音频 + 文本 + 韵律，视频输入置零 | 冻结 `audio/text/flow` | 证明 GPU 流式主线下，不依赖 feature cache 也能跑出稳定基线 | 64GB 显存，12 核 CPU |
| 实验二 | `MMSER-Pytorch-frozen-gpu (RGB+A+T)` | RGB + 音频 + 文本 + 韵律 | 冻结 `audio/rgb/text` | 证明 RGB 模态纳入后，冻结编码器的三模态融合是否优于实验一 | 64GB 显存，16 核 CPU |
| 实验三 | `MMSER-Pytorch-upper-gpu (RGB+A+T, trainable)` | RGB + 音频 + 文本 + 韵律 | 不冻结主编码器 | 证明放开 RGB/A/T 编码器后，三模态上限能否继续提高 | 80GB+ 显存，24 核 CPU |
| 实验四 | `MMSER-Pytorch-dual-upper-gpu (GPU-motion+RGB+A+T)` | GPU 动态分支 + RGB + 音频 + 文本 + 韵律 | 不冻结主编码器 | 证明当前 GPU 动态分支相对于 RGB-only 上限是否还有增益 | 80GB+ 显存，24 核+CPU，NVMe |

## 共享准备步骤

下面四组实验都共享这一段准备流程。只要同一台服务器已经做过一次，就不需要每次重复。

```bash
cd /root/private_data/MMSER-Pytorch
git pull

export LD_LIBRARY_PATH="/opt/dtk/cuda/cuda-12/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib:/usr/local/lib64:/opt/mpi/lib:/opt/hwloc/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}"
export ROCM_PATH=/opt/dtk
export HF_ENDPOINT=https://hf-mirror.com

# 第一次下载模型时不要开离线模式。
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

python3 build_split_manifest.py \
  --dataset-kind meld \
  --data-root /root/private_data/MELD.Raw/MELD.Raw \
  --metadata-root /root/private_data/MELD.metadata \
  --audio-cache-root outputs/datasets/meld/audio_cache \
  --output outputs/benchmarks/meld/splits/default_manifest.json

python3 prepare_dataset_media.py \
  --dataset-kind meld \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.json \
  --subset all \
  --num-workers auto

python3 filter_meld_manifest.py
python3 download.py
```

经过这一步后，正式实验统一使用：

- `outputs/benchmarks/meld/splits/default_manifest.filtered.json`

说明：

- `prepare_dataset_media.py` 仍然会用到 CPU 和磁盘，因为当前 GPU 主线仍依赖 `audio_path` sidecar。
- 这一步是一次性准备，不属于四组主实验命令的一部分。
- 如果模型已经下载完成，后续正式训练前可以再加：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## MELD GPU 主实验以后不要再用的旧命令

下面这些命令属于旧的 CPU 缓存式 MELD 路线。对于本文这四组 GPU 主实验，后续都不要再用：

- `python3 build_feature_cache.py ...`
- `python3 validate_cached_shards.py ...`
- `python3 train.py --feature-cache ...`
- `python3 batch_inference.py --feature-cache ...`
- 任何依赖 `outputs/benchmarks/meld/feature_cache_*` 的训练或推理命令

原因很简单：

- 它们仍然会先走一次重型媒体读取、特征编码和磁盘写 shard
- 这不是当前文档要跑的 GPU 流式主线
- 这也是你之前感觉“第一个命令就很慢”的根本原因

因此，当前这四组正式实验的起点统一都是：

- `python3 train.py --pipeline gpu_stream --cache-mode none ...`

推理统一都是：

- `python3 batch_inference.py --pipeline gpu_stream --cache-mode none ...`

---

## 实验一：MELD GPU 实用基线

### 命名建议

中文：
`MMSER-Pytorch 在 MELD 官方划分上的 GPU 实用基线（音频+文本+韵律，视频输入置零，冻结预训练编码器）`

英文：
`MMSER-Pytorch Practical GPU Baseline on MELD (Audio + Text + Prosody, Zeroed Video Input, Frozen Encoders)`

表格短名：
`MMSER-Pytorch-practical-gpu (A+T+P, zero-video-input, frozen)`

### 使用了哪些分支

- 音频分支：使用
- 文本分支：使用
- 韵律分支：使用
- 视频输入：置零
- 动态分支：结构存在，但输入置零且编码器冻结
- RGB 分支：不使用

### 冻结了哪些模块

- `audio` 编码器：冻结
- `text` 编码器：冻结
- `flow` 编码器：冻结
- 实际训练部分主要是：
  - 融合头
  - prosody 小分支
  - 顶层小模块

### 这组实验能证明什么

- 证明 `MMSER-Pytorch` 在 `MELD` 官方 split 上，可以直接走 GPU 流式训练，不依赖 feature cache
- 证明 `音频 + 文本 + 韵律 + gated_text` 在 GPU 主线上是可迁移、可复现的
- 这是最适合作为第一张基线表的 GPU 版结果

### 这组实验不能证明什么

- 它不是“物理删除视频分支”的纯音频文本模型，而是**视频输入置零**
- 不能证明 RGB 分支有效
- 不能证明 GPU 动态分支有效
- 不能代表完整多模态上限

### 推荐服务器

- GPU：1 卡，显存 `>= 64GB`
- CPU：`>= 12` 核
- 内存：`>= 96GB`
- 磁盘：SSD 即可

### 命令

这是实验一的最终正确 GPU 命令。不要再为它额外执行 `build_feature_cache.py`。

```bash
rm -rf outputs/benchmarks/meld/run_gpu_practical_no_video

python3 train.py \
  --pipeline gpu_stream \
  --cache-mode none \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --output-dir outputs/benchmarks/meld/run_gpu_practical_no_video \
  --epochs 60 \
  --device auto \
  --amp-mode bf16 \
  --batch-size 32 \
  --num-workers 0 \
  --video-backbone flow \
  --flow-backend torch_motion \
  --audio-model microsoft/wavlm-large \
  --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
  --text-model FacebookAI/xlm-roberta-large \
  --fusion-mode gated_text \
  --freeze-audio \
  --freeze-flow \
  --freeze-text \
  --task-mode confounded_7way \
  --text-policy full \
  --ablation no-video \
  --sample-rate 24000 \
  --max-audio-sec 6 \
  --monitor val_f1 \
  --early-stop-patience 10 \
  --lr 1e-3 \
  --benchmark-tag meld_gpu_practical_no_video

python3 batch_inference.py \
  --pipeline gpu_stream \
  --cache-mode none \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --subset val \
  --checkpoint outputs/benchmarks/meld/run_gpu_practical_no_video/checkpoints/best.pt \
  --output outputs/benchmarks/meld/run_gpu_practical_no_video/inference_val.jsonl \
  --device auto \
  --amp-mode bf16 \
  --batch-size 64 \
  --num-workers 0 \
  --video-backbone flow \
  --flow-backend torch_motion \
  --audio-model microsoft/wavlm-large \
  --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
  --text-model FacebookAI/xlm-roberta-large \
  --task-mode confounded_7way \
  --text-policy full \
  --ablation no-video \
  --sample-rate 24000 \
  --max-audio-sec 6

python3 batch_inference.py \
  --pipeline gpu_stream \
  --cache-mode none \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --subset test \
  --checkpoint outputs/benchmarks/meld/run_gpu_practical_no_video/checkpoints/best.pt \
  --output outputs/benchmarks/meld/run_gpu_practical_no_video/inference_test.jsonl \
  --device auto \
  --amp-mode bf16 \
  --batch-size 64 \
  --num-workers 0 \
  --video-backbone flow \
  --flow-backend torch_motion \
  --audio-model microsoft/wavlm-large \
  --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
  --text-model FacebookAI/xlm-roberta-large \
  --task-mode confounded_7way \
  --text-policy full \
  --ablation no-video \
  --sample-rate 24000 \
  --max-audio-sec 6
```

### 结果如何看

```bash
python3 -m json.tool outputs/benchmarks/meld/run_gpu_practical_no_video/metrics.json
python3 -m json.tool outputs/benchmarks/meld/run_gpu_practical_no_video/inference_val.metrics.json
python3 -m json.tool outputs/benchmarks/meld/run_gpu_practical_no_video/inference_test.metrics.json
```

重点看：

- `outputs/benchmarks/meld/run_gpu_practical_no_video/inference_test.metrics.json`
- `accuracy_on_ok`
- `macro_f1_on_ok`
- `pipeline` 是否为 `gpu_stream`
- `cache_mode` 是否为 `none`

### 总结

这是最稳、最容易先出结果的一组 GPU 基线。它的价值在于“低磁盘写入 + GPU 流式主线可复现”，而不是代表完整多模态上限。

---

## 实验二：MELD GPU 三模态冻结基线

### 命名建议

中文：
`MMSER-Pytorch 在 MELD 官方划分上的 GPU 三模态冻结基线（RGB+音频+文本）`

英文：
`MMSER-Pytorch Tri-Modal Frozen GPU Baseline on MELD (RGB + Audio + Text)`

表格短名：
`MMSER-Pytorch-frozen-gpu (RGB+A+T)`

### 使用了哪些分支

- RGB 分支：使用
- 音频分支：使用
- 文本分支：使用
- 韵律分支：使用
- GPU 动态分支：不使用

### 冻结了哪些模块

- `audio` 编码器：冻结
- `rgb` 编码器：冻结
- `text` 编码器：冻结
- 训练部分主要是融合头和顶层小模块

### 这组实验能证明什么

- 证明当前 GPU 流式主线下，真正把 RGB 视频模态纳入后是否优于实验一
- 证明 `RGB + Audio + Text + Prosody` 这条冻结编码器三模态路线在 `MELD` 上的表现

### 这组实验不能证明什么

- 还不能证明端到端训练上限
- 还没有把 GPU 动态分支纳入
- 当前视频预处理是 **center crop**，不等价旧 CPU 路线的人脸裁剪

### 推荐服务器

- GPU：1 卡，显存 `>= 64GB`
- CPU：`>= 16` 核
- 内存：`>= 128GB`
- 磁盘：SSD / NVMe

### 命令

这是实验二的最终正确 GPU 命令。不要再为它生成 `feature_cache_tri_*` 一类目录。

```bash
rm -rf outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen

python3 train.py \
  --pipeline gpu_stream \
  --cache-mode none \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --output-dir outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen \
  --epochs 30 \
  --device auto \
  --amp-mode bf16 \
  --batch-size 8 \
  --num-workers 0 \
  --video-backbone videomae \
  --num-frames 16 \
  --rgb-size 224 \
  --audio-model microsoft/wavlm-large \
  --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
  --text-model FacebookAI/xlm-roberta-large \
  --fusion-mode gated_text \
  --freeze-audio \
  --freeze-rgb \
  --freeze-text \
  --task-mode confounded_7way \
  --text-policy full \
  --sample-rate 24000 \
  --max-audio-sec 6 \
  --monitor val_f1 \
  --early-stop-patience 6 \
  --benchmark-tag meld_gpu_tri_rgb_audio_text_frozen

python3 batch_inference.py \
  --pipeline gpu_stream \
  --cache-mode none \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --subset val \
  --checkpoint outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen/checkpoints/best.pt \
  --output outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen/inference_val.jsonl \
  --device auto \
  --amp-mode bf16 \
  --batch-size 16 \
  --num-workers 0 \
  --video-backbone videomae \
  --num-frames 16 \
  --rgb-size 224 \
  --audio-model microsoft/wavlm-large \
  --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
  --text-model FacebookAI/xlm-roberta-large \
  --task-mode confounded_7way \
  --text-policy full \
  --sample-rate 24000 \
  --max-audio-sec 6

python3 batch_inference.py \
  --pipeline gpu_stream \
  --cache-mode none \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --subset test \
  --checkpoint outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen/checkpoints/best.pt \
  --output outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen/inference_test.jsonl \
  --device auto \
  --amp-mode bf16 \
  --batch-size 16 \
  --num-workers 0 \
  --video-backbone videomae \
  --num-frames 16 \
  --rgb-size 224 \
  --audio-model microsoft/wavlm-large \
  --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
  --text-model FacebookAI/xlm-roberta-large \
  --task-mode confounded_7way \
  --text-policy full \
  --sample-rate 24000 \
  --max-audio-sec 6
```

### 结果如何看

```bash
python3 -m json.tool outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen/metrics.json
python3 -m json.tool outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen/inference_val.metrics.json
python3 -m json.tool outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen/inference_test.metrics.json
```

重点看：

- 相比实验一，`test macro_f1` 是否提升
- `per_class_recall` 是否更均衡
- `pipeline` 是否为 `gpu_stream`

### 总结

这是当前 GPU 路线里“真正把 RGB 模态带上”的第一组三模态基线，比实验一更能体现多模态融合能力。

---

## 实验三：MELD GPU RGB 三模态上限

### 命名建议

中文：
`MMSER-Pytorch 在 MELD 官方划分上的 GPU RGB 三模态上限探索（RGB+音频+文本，编码器参与训练）`

英文：
`MMSER-Pytorch RGB Tri-Modal GPU Upper-Bound on MELD (RGB + Audio + Text, Trainable Encoders)`

表格短名：
`MMSER-Pytorch-upper-gpu (RGB+A+T, trainable)`

### 使用了哪些分支

- RGB 分支：使用，并参与训练
- 音频分支：使用，并参与训练
- 文本分支：使用，并参与训练
- 韵律分支：使用
- GPU 动态分支：不使用

### 冻结了哪些模块

- 不冻结 `audio/rgb/text`
- 这是比实验二更接近完整能力的 RGB 三模态设置

### 这组实验能证明什么

- 证明放开 RGB/A/T 主编码器后，三模态能力上限能否超过实验二
- 证明当前 `MMSER-Pytorch` 在 `MELD` 上的更完整三模态能力

### 这组实验不能证明什么

- 还没有把 GPU 动态分支纳入
- 当前视频预处理仍然是 **center crop**，不等价旧 CPU 人脸裁剪

### 推荐服务器

- GPU：1 卡，显存 `>= 80GB`
- CPU：`>= 24` 核
- 内存：`>= 128GB`
- 磁盘：NVMe

### 命令

这是实验三的最终正确 GPU 命令。它直接从原始媒体流式训练，不需要任何中间 shard。

```bash
rm -rf outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text

python3 train.py \
  --pipeline gpu_stream \
  --cache-mode none \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --output-dir outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text \
  --epochs 20 \
  --device auto \
  --amp-mode bf16 \
  --batch-size 2 \
  --num-workers 0 \
  --video-backbone videomae \
  --num-frames 16 \
  --rgb-size 224 \
  --audio-model microsoft/wavlm-large \
  --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
  --text-model FacebookAI/xlm-roberta-large \
  --fusion-mode gated_text \
  --task-mode confounded_7way \
  --text-policy full \
  --sample-rate 24000 \
  --max-audio-sec 6 \
  --monitor val_f1 \
  --early-stop-patience 5 \
  --benchmark-tag meld_gpu_upper_rgb_audio_text

python3 batch_inference.py \
  --pipeline gpu_stream \
  --cache-mode none \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --subset val \
  --checkpoint outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text/checkpoints/best.pt \
  --output outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text/inference_val.jsonl \
  --device auto \
  --amp-mode bf16 \
  --batch-size 8 \
  --num-workers 0 \
  --video-backbone videomae \
  --num-frames 16 \
  --rgb-size 224 \
  --audio-model microsoft/wavlm-large \
  --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
  --text-model FacebookAI/xlm-roberta-large \
  --task-mode confounded_7way \
  --text-policy full \
  --sample-rate 24000 \
  --max-audio-sec 6

python3 batch_inference.py \
  --pipeline gpu_stream \
  --cache-mode none \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --subset test \
  --checkpoint outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text/checkpoints/best.pt \
  --output outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text/inference_test.jsonl \
  --device auto \
  --amp-mode bf16 \
  --batch-size 8 \
  --num-workers 0 \
  --video-backbone videomae \
  --num-frames 16 \
  --rgb-size 224 \
  --audio-model microsoft/wavlm-large \
  --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
  --text-model FacebookAI/xlm-roberta-large \
  --task-mode confounded_7way \
  --text-policy full \
  --sample-rate 24000 \
  --max-audio-sec 6
```

### 结果如何看

```bash
python3 -m json.tool outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text/metrics.json
python3 -m json.tool outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text/inference_val.metrics.json
python3 -m json.tool outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text/inference_test.metrics.json
```

重点看：

- 相比实验二，`test macro_f1` 是否继续提升
- 是否值得继续把 GPU 动态分支也纳入

### 总结

这是当前 GPU 路线里 RGB 三模态的正式上限版本。比实验二更接近完整能力，但训练成本会明显更高。

---

## 实验四：MELD GPU 完整双视频分支上限

### 命名建议

中文：
`MMSER-Pytorch 在 MELD 官方划分上的 GPU 双视频分支上限探索（GPU 动态分支 + RGB + 音频 + 文本）`

英文：
`MMSER-Pytorch Dual-Video GPU Upper-Bound on MELD (GPU Motion + RGB + Audio + Text)`

表格短名：
`MMSER-Pytorch-dual-upper-gpu (GPU-motion+RGB+A+T)`

### 使用了哪些分支

- GPU 动态分支：使用，并参与训练
- RGB 分支：使用，并参与训练
- 音频分支：使用，并参与训练
- 文本分支：使用，并参与训练
- 韵律分支：使用

### 冻结了哪些模块

- 默认不冻结主编码器
- 这是四组里最接近完整工程能力的一组

### 这组实验能证明什么

- 证明当前 GPU 动态分支相对于 RGB-only 上限是否还有额外收益
- 证明完整 `dual` 视频设计在 `MELD` 上是否值得保留

### 这组实验不能证明什么

- 它证明的是 **当前 GPU 动态分支实现** 的价值
- 它不等价于旧 CPU 路线的 Haar + Farneback 光流
- 不能直接与旧 CPU 光流结果做数值等价比较

### 推荐服务器

- GPU：1 卡，显存 `>= 80GB`
- CPU：`>= 24` 核，越高越好
- 内存：`>= 192GB`
- 磁盘：NVMe

### 命令

这是实验四的最终正确 GPU 命令。它对应当前 GPU 版 `dual(torch_motion + RGB)` 主线，不要再混用旧 CPU 光流缓存命令。

```bash
rm -rf outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion

python3 train.py \
  --pipeline gpu_stream \
  --cache-mode none \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --output-dir outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion \
  --epochs 20 \
  --device auto \
  --amp-mode bf16 \
  --batch-size 1 \
  --num-workers 0 \
  --video-backbone dual \
  --flow-backend torch_motion \
  --num-frames 32 \
  --flow-size 112 \
  --rgb-size 224 \
  --audio-model microsoft/wavlm-large \
  --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
  --text-model FacebookAI/xlm-roberta-large \
  --fusion-mode gated_text \
  --task-mode confounded_7way \
  --text-policy full \
  --sample-rate 24000 \
  --max-audio-sec 6 \
  --monitor val_f1 \
  --early-stop-patience 5 \
  --benchmark-tag meld_gpu_upper_dual_torch_motion

python3 batch_inference.py \
  --pipeline gpu_stream \
  --cache-mode none \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --subset val \
  --checkpoint outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion/checkpoints/best.pt \
  --output outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion/inference_val.jsonl \
  --device auto \
  --amp-mode bf16 \
  --batch-size 4 \
  --num-workers 0 \
  --video-backbone dual \
  --flow-backend torch_motion \
  --num-frames 32 \
  --flow-size 112 \
  --rgb-size 224 \
  --audio-model microsoft/wavlm-large \
  --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
  --text-model FacebookAI/xlm-roberta-large \
  --task-mode confounded_7way \
  --text-policy full \
  --sample-rate 24000 \
  --max-audio-sec 6

python3 batch_inference.py \
  --pipeline gpu_stream \
  --cache-mode none \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --subset test \
  --checkpoint outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion/checkpoints/best.pt \
  --output outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion/inference_test.jsonl \
  --device auto \
  --amp-mode bf16 \
  --batch-size 4 \
  --num-workers 0 \
  --video-backbone dual \
  --flow-backend torch_motion \
  --num-frames 32 \
  --flow-size 112 \
  --rgb-size 224 \
  --audio-model microsoft/wavlm-large \
  --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
  --text-model FacebookAI/xlm-roberta-large \
  --task-mode confounded_7way \
  --text-policy full \
  --sample-rate 24000 \
  --max-audio-sec 6
```

### 结果如何看

```bash
python3 -m json.tool outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion/metrics.json
python3 -m json.tool outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion/inference_val.metrics.json
python3 -m json.tool outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion/inference_test.metrics.json
```

重点看：

- `test macro_f1` 是否超过实验三
- `per_class_recall` 是否更均衡
- GPU 动态分支是否带来实际增益

### 总结

这是当前仓库里最接近“完整 GPU 版 `MMSER-Pytorch` 工程能力”的实验。它不是旧 CPU 光流路线的复现，而是当前 GPU 动态分支路线的正式上限实验。

---

## 推荐的执行顺序

如果你第一次做 GPU 版 MELD 实验，建议固定按下面顺序执行：

1. 实验一：先确认 GPU 流式主线稳定
2. 实验二：确认 RGB 模态是否真的带来增益
3. 实验三：确认放开 RGB/A/T 编码器后是否继续提升
4. 实验四：最后判断 GPU 动态分支值不值得保留

## 四组实验分别能证明什么

- 实验一证明：
  在不依赖 feature cache 的前提下，`MMSER-Pytorch` 可以直接从 `MELD` 原始媒体流式训练，并给出一条稳定、可复现、低磁盘写入的 GPU 基线。
- 实验二证明：
  把 RGB 模态纳入后，冻结编码器的三模态融合是否比实验一更强。
- 实验三证明：
  不再冻结 RGB/A/T 主编码器后，完整三模态能力上限能否进一步提高。
- 实验四证明：
  当前 GPU 动态分支相对于 RGB-only 上限是否还能带来收益，是不是值得保留 `dual` 设计。

## 结果比较顺序

真正最关键的不是某一组单点最高分，而是下面三条增益链条是否成立：

- `实验二 > 实验一`
  说明 RGB 模态有实际贡献
- `实验三 > 实验二`
  说明放开 RGB/A/T 编码器有实际贡献
- `实验四 > 实验三`
  说明当前 GPU 动态分支有实际贡献

如果某一级没有明显提升，就应该重新评估那个分支或训练成本是否值得保留。
