# MELD GPU 基线与上限实验运行手册

本文档整理 `MMSER-Pytorch` 在 `MELD` 数据集上的四组 **GPU 流式实验** 命令与解释。

正式写论文、报告或对外表格前，请先通读根目录的 [PAPER_GRADE_PROTOCOL.md](/Users/dailulu/projects/MMSER-Pytorch/PAPER_GRADE_PROTOCOL.md)。本手册只描述 MELD 主线命令，不替代仓库级的论文口径约束。

这版手册只保留当前主线实现对应的 GPU 路线：

- 训练与推理统一走 `gpu_stream`
- 中间不再写旧 `raw/feature cache` shard
- 允许使用新的 `mainline input cache` 作为论文级等价加速路径
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

当前推荐默认使用服务器全局 Python 环境，不再额外创建 `.venv-server` 或 conda 环境。`setup_ubuntu_server.sh` 会在全局环境上补装缺失依赖，并把模型缓存固定到仓库本地目录。

```bash
cd /root/private_data/MMSER-Pytorch
git pull

export HF_ENDPOINT=https://hf-mirror.com

# 第一次下载模型时不要开离线模式。
#unset HF_HUB_OFFLINE
#unset TRANSFORMERS_OFFLINE

USE_GLOBAL_ENV=1 INSTALL_LEGACY_EXTRAS=0 source setup_ubuntu_server.sh
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# 先确认真正存放 MELD MP4 的目录层级。
find ../MELD.Raw -type f -name '*.mp4' | head

python3 build_split_manifest.py \
  --dataset-kind meld \
  --data-root ../MELD.Raw \
  --metadata-root ../MELD.metadata \
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

- 当前手册默认使用全局环境模式：`USE_GLOBAL_ENV=1`。只有在你明确想隔离依赖、并且服务器本身没有环境兼容问题时，才考虑脚本里的 conda 路线。
- `prepare_dataset_media.py` 仍然会用到 CPU 和磁盘，因为当前 GPU 主线仍依赖 `audio_path` sidecar。
- 这一步是一次性准备，不属于四组主实验命令的一部分。
- `filter_meld_manifest.py` 只做本地 manifest 过滤，不会下载任何模型或远程资源。
- `download.py` 默认把 Hugging Face 模型快照下载到仓库本地的 `.hf-cache/hub/`。同一台机器第二次运行时，会先校验本地缓存的 `config`、`tokenizer` 和 `safetensors` 头是否完整；只有校验通过才直接复用，否则会重新下载损坏或不完整的快照。
- 当前主线在 `train.py` / `batch_inference.py` 里会优先解析仓库本地 `.hf-cache/hub/` 中已验证的 Hugging Face 快照。只要 `download.py` 已成功跑过，后续正式训练和推理即使并发开多个进程，也不应该再因为 `AutoTokenizer` / `AutoModel` 初始化先去 `HEAD huggingface.co` 而卡住。
- 当前仓库已经在 `download.py`、`train.py`、`batch_inference.py` 以及相关 Hugging Face 入口里补了 `torch.utils._pytree` 兼容层，用来兼容只暴露 `_register_pytree_node` 的旧版全局 `torch`。如果你遇到 `register_pytree_node` 缺失报错，先 `git pull` 再重试，不要第一反应就重装整套全局 CUDA/PyTorch 环境。
- 当前主线会显式关闭 `microsoft/wavlm-large` 这类 Hugging Face 音频编码器在训练态的内部 SpecAugment / time masking。原因有两个：一是它不属于本文档的正式实验变量；二是在部分 `torch + transformers + CUDA` 组合上，它会触发 `masked_spec_embed` 相关的 CUDA 索引断言。若你遇到这类报错，先 `git pull`，不要自行回退到旧代码。
- 当前 `dual/flow` 分支的 `FlowVideoEncoder` 已改为确定性友好的空间 `AvgPool3d` 版本，不再使用会在 `torch.use_deterministic_algorithms(True)` 下直接报错的 `MaxPool3d` backward。论文级正式结果不要混用这次修改前训练出来的旧 `dual` checkpoint；第四组请在最新代码上从头重跑。
- `--data-root` 必须指向实际包含 `train_splits`、`dev_splits_complete` 或 `output_repeated_splits_test` 的目录。很多服务器应该填 `../MELD.Raw`，不要机械照抄成 `../MELD.Raw/MELD.Raw`。
- 如果 `build_split_manifest.py` 打印 `usable_rows: 0`，说明 MP4 根目录填错了；先修正 `--data-root`，再重新生成 manifest。
- 下文四组实验的音频统一使用 `16kHz`，以对齐 `microsoft/wavlm-large` 的预训练采样率。
- 训练/推理主线已经不再公开 `--pipeline` / `--cache-mode` 这类旧运行时参数；当前应该直接使用最短的 manifest/gpu_stream 命令。
- 训练/推理开始后，控制台会额外打印 `gpu_stream_backends`、`gpu_stream_prepare_stats`，训练日志还会带 `train_prepare_s`、`val_prepare_s`，它们用于判断瓶颈是在 CPU ingress 还是模型前向。
- `--num-workers auto` 不是硬性要求，而是默认推荐值。如果像 `K100_AI` 这类旧内核/特殊驱动服务器在首个 batch 附近出现 `Segmentation fault (core dumped)`，通常就是 worker 子进程和底层音频/视频库不兼容，此时应把该机器上的训练与推理统一固定成 `--num-workers 0`。
- 再次进入同一台机器上的同一仓库时，仍然直接执行 `USE_GLOBAL_ENV=1 source setup_ubuntu_server.sh` 即可。依赖如果已经装好，脚本会自动跳过重复安装。
- 如果模型已经下载完成，后续正式训练前可以再加：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 可选的主线输入缓存加速路径

如果你的 GPU 服务器 CPU 明显喂不满 GPU，正式实验推荐使用新的 `mainline input cache`。它和旧 `feature cache` 不是一回事：

- 新 `input cache` 只缓存当前主线等价输入：`load_audio()` 输出波形、已经按主线规则 crop/resize 完的 RGB clip、以及 `full/masked` 文本 token
- 它不会缓存 prosody、legacy flow、音频 embedding 或文本 embedding
- 因此它仍然属于论文级等价加速路径，只要 `input_cache_contract` 与当前运行参数匹配

推荐做法是：

1. 在 CPU 服务器上构建缓存
2. 把缓存目录整体拷到 GPU 服务器
3. 四组正式实验统一通过 `--input-cache` 读取

建议固定使用下面三套缓存目录：

- 实验一：`outputs/benchmarks/meld/input_cache/audio_text_sr16000_len6_v1`
- 实验二/三：`outputs/benchmarks/meld/input_cache/rgb16_audio_text_sr16000_len6_v1`
- 实验四：`outputs/benchmarks/meld/input_cache/rgb32_audio_text_sr16000_len6_v1`

CPU 服务器构建命令：

```bash
python3 build_mainline_input_cache.py \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --output-dir outputs/benchmarks/meld/input_cache/audio_text_sr16000_len6_v1 \
  --subset all \
  --sample-rate 16000 \
  --max-audio-sec 6 \
  --text-model FacebookAI/xlm-roberta-large \
  --max-text-len 128 \
  --num-workers auto

python3 build_mainline_input_cache.py \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --output-dir outputs/benchmarks/meld/input_cache/rgb16_audio_text_sr16000_len6_v1 \
  --subset all \
  --sample-rate 16000 \
  --max-audio-sec 6 \
  --text-model FacebookAI/xlm-roberta-large \
  --max-text-len 128 \
  --include-video \
  --num-frames 16 \
  --rgb-size 224 \
  --num-workers auto

python3 build_mainline_input_cache.py \
  --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
  --output-dir outputs/benchmarks/meld/input_cache/rgb32_audio_text_sr16000_len6_v1 \
  --subset all \
  --sample-rate 16000 \
  --max-audio-sec 6 \
  --text-model FacebookAI/xlm-roberta-large \
  --max-text-len 128 \
  --include-video \
  --num-frames 32 \
  --rgb-size 224 \
  --num-workers auto
```

如果你已经在旧代码上开始构建视频缓存，或者在 CPU 服务器上遇到 `PytorchStreamWriter failed writing file data/1` / `unexpected pos` 这类写盘错误，先删除未完成的视频缓存目录后再重跑。当前主线的视频缓存会保存更小的预处理后 RGB clip，不再推荐继续保留旧的半成品目录。

GPU 服务器上只需要把目录拷过来，后续训练/推理命令统一附带对应的 `--input-cache ...`。如果你不使用这个加速路径，删掉 `--input-cache` 那一行即可；正式论文口径仍然成立，只是 CPU 压力会更高。

## 统一的停止与报告协议

从这一版开始，四组 MELD GPU 实验不再采用“单次 run 里连续 5 轮没涨就停”的经验式口径，而统一采用下面这套更稳妥的协议：

- headline 结果一律来自 `5` 个固定随机种子：`13 17 23 42 3407`
- 每个 seed 都独立训练一次，只允许用验证集选择 `checkpoints/best.pt`
- 同一组 `5` 个 seed 必须统一 `--input-cache` 使用方式，不能一部分走缓存、一部分走原始流式路径
- 测试集只在该 seed 训练结束后跑一次，不能参与早停、调参或挑 checkpoint
- 主监控指标固定为 `val_f1`
- “真正提升”定义为：当前 `val_f1 > 历史最好 val_f1 + 0.002`
- 如果当前 `val_f1` 与历史最好值的差距不超过 `0.002`，则允许用更低的 `val_loss` 替换 checkpoint，但这种替换**不会**重置早停 patience
- 学习率调度固定为 `ReduceLROnPlateau`：监控 `val_loss`，`factor=0.5`，`patience=4`，`min_lr=1e-6`
- 早停必须同时满足三条：
  - 已达到该实验的 `min_epochs`
  - 已经发生至少 `2` 次 LR 降档
  - `val_f1` 连续 `stop_patience` 轮没有超过历史最好值 `+ 0.002`

这套口径的核心目标不是“尽快停”，而是尽量避免因为验证集抖动、短期尖峰或单次随机波动，把结果做得不可信。

统一说明：

- `batch_inference.py` 的 checkpoint 统一使用 `checkpoints/best.pt`
- 正式实验里不要使用 `--allow-incompatible-checkpoint`；它只用于调试，且会让输出自动失去论文级资格
- 推理统计摘要统一写成 `inference_val.metrics.json` 和 `inference_test.metrics.json`
- 训练、推理和聚合产物里都应检查 `paper_grade.eligible=true`
- 正式 headline 结果必须保证 train / val / test 共享同一 `manifest_sha256`
- 单个 seed 的最好结果只能作为明细，不应直接作为论文表格 headline
- 正式对外汇报时，应使用多 seed 汇总结果里的 `mean ± std`、`95% CI` 和相邻实验之间的配对显著性检验
- 如果某次 run 的 `paper_grade.eligible=false`，即使数值更高，也不能放进正式论文表

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

- `python3 train.py --split-manifest ...`

推理统一都是：

- `python3 batch_inference.py --split-manifest ...`

这里的主线运行方式要理解成：

- 不再生成 raw/feature shard 这类**磁盘缓存**
- 不是“禁止进程内 RAM 级复用”

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

这是实验一的正式论文级运行口径。不要再为它额外执行 `build_feature_cache.py`；如果想减轻 GPU 服务器 CPU 压力，直接使用下面的 `--input-cache`。
训练、`val` 推理、`test` 推理分别按下面三个 loop 执行；每个 loop 会把 `5` 个 seed 全部跑完。
当前 GPU 主线已经把媒体 ingress 挪到 `DataLoader worker`，默认推荐 `--num-workers auto`。但你当前这类 `K100_AI`/旧内核服务器已经实测在 worker 模式下会于首个 batch 附近触发 `Segmentation fault (core dumped)`，因此本机应把下面命令里的 `--num-workers auto` 改成 `--num-workers 0`。

训练：

```bash
SEEDS="13 17 23 42 3407"
INPUT_CACHE=outputs/benchmarks/meld/input_cache/audio_text_sr16000_len6_v1

for seed in $SEEDS; do
  rm -rf outputs/benchmarks/meld/run_gpu_practical_no_video_seed${seed}

  python3 train.py \
    --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
    --input-cache ${INPUT_CACHE} \
    --output-dir outputs/benchmarks/meld/run_gpu_practical_no_video_seed${seed} \
    --epochs 80 \
    --device auto \
    --amp-mode bf16 \
    --batch-size 16 \
    --num-workers auto \
    --video-backbone flow \
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
    --sample-rate 16000 \
    --max-audio-sec 6 \
    --monitor val_f1 \
    --early-stop-patience 12 \
    --early-stop-min-epochs 20 \
    --early-stop-min-delta 0.002 \
    --early-stop-after-lr-drops 2 \
    --lr-scheduler plateau \
    --lr-plateau-patience 4 \
    --lr-plateau-factor 0.5 \
    --lr-min 1e-6 \
    --lr 1e-3 \
    --seed ${seed} \
    --benchmark-tag meld_gpu_practical_no_video_seed${seed}
done
```

验证集推理：

```bash
SEEDS="13 17 23 42 3407"
INPUT_CACHE=outputs/benchmarks/meld/input_cache/audio_text_sr16000_len6_v1

for seed in $SEEDS; do
  python3 batch_inference.py \
    --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
    --input-cache ${INPUT_CACHE} \
    --subset val \
    --checkpoint outputs/benchmarks/meld/run_gpu_practical_no_video_seed${seed}/checkpoints/best.pt \
    --output outputs/benchmarks/meld/run_gpu_practical_no_video_seed${seed}/inference_val.jsonl \
    --device auto \
    --amp-mode bf16 \
    --batch-size 16 \
    --num-workers auto \
    --video-backbone flow \
    --audio-model microsoft/wavlm-large \
    --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
    --text-model FacebookAI/xlm-roberta-large \
    --task-mode confounded_7way \
    --text-policy full \
    --ablation no-video \
    --sample-rate 16000 \
    --max-audio-sec 6 \
    --benchmark-tag meld_gpu_practical_no_video_seed${seed}
done
```

测试集推理：

```bash
SEEDS="13 17 23 42 3407"
INPUT_CACHE=outputs/benchmarks/meld/input_cache/audio_text_sr16000_len6_v1

for seed in $SEEDS; do
  python3 batch_inference.py \
    --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
    --input-cache ${INPUT_CACHE} \
    --subset test \
    --checkpoint outputs/benchmarks/meld/run_gpu_practical_no_video_seed${seed}/checkpoints/best.pt \
    --output outputs/benchmarks/meld/run_gpu_practical_no_video_seed${seed}/inference_test.jsonl \
    --device auto \
    --amp-mode bf16 \
    --batch-size 16 \
    --num-workers auto \
    --video-backbone flow \
    --audio-model microsoft/wavlm-large \
    --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
    --text-model FacebookAI/xlm-roberta-large \
    --task-mode confounded_7way \
    --text-policy full \
    --ablation no-video \
    --sample-rate 16000 \
    --max-audio-sec 6 \
    --benchmark-tag meld_gpu_practical_no_video_seed${seed}
done
```

### 结果如何看

```bash
python3 aggregate_multi_seed_results.py \
  --group practical=outputs/benchmarks/meld/run_gpu_practical_no_video_seed* \
  --output-dir outputs/benchmarks/meld/reports/practical_5seed

python3 -m json.tool outputs/benchmarks/meld/reports/practical_5seed/multi_seed_summary.json
python3 -m json.tool outputs/benchmarks/meld/reports/practical_5seed/pairwise_significance.json
```

重点看：

- `multi_seed_summary.json` 里的 `test_macro_f1.mean/std/95% CI`
- 每个 seed 的 `metrics.json` 里 `stop.reason`、`stop.lr_drop_epochs` 和 `best.best_val_loss`
- 这一组主要作为后续三组的统计比较基线

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

这是实验二的正式论文级运行口径。不要再为它生成旧 `feature_cache_tri_*` 一类目录；正式加速方式改用下面的 `--input-cache`。
训练、`val` 推理、`test` 推理分别按下面三个 loop 执行，全部使用同一组 `5` 个 seed。
当前 GPU 主线已经把媒体 ingress 挪到 `DataLoader worker`，默认推荐 `--num-workers auto`。但你当前这类 `K100_AI`/旧内核服务器已经实测在 worker 模式下会于首个 batch 附近触发 `Segmentation fault (core dumped)`，因此本机应把下面命令里的 `--num-workers auto` 改成 `--num-workers 0`。

训练：

```bash
SEEDS="13 17 23 42 3407"
INPUT_CACHE=outputs/benchmarks/meld/input_cache/rgb16_audio_text_sr16000_len6_v1

for seed in $SEEDS; do
  rm -rf outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen_seed${seed}

  python3 train.py \
    --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
    --input-cache ${INPUT_CACHE} \
    --output-dir outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen_seed${seed} \
    --epochs 50 \
    --device auto \
    --amp-mode bf16 \
    --batch-size 16 \
    --num-workers auto \
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
    --sample-rate 16000 \
    --max-audio-sec 6 \
    --monitor val_f1 \
    --early-stop-patience 10 \
    --early-stop-min-epochs 15 \
    --early-stop-min-delta 0.002 \
    --early-stop-after-lr-drops 2 \
    --lr-scheduler plateau \
    --lr-plateau-patience 4 \
    --lr-plateau-factor 0.5 \
    --lr-min 1e-6 \
    --seed ${seed} \
    --benchmark-tag meld_gpu_tri_rgb_audio_text_frozen_seed${seed}
done
```

验证集推理：

```bash
SEEDS="13 17 23 42 3407"
INPUT_CACHE=outputs/benchmarks/meld/input_cache/rgb16_audio_text_sr16000_len6_v1

for seed in $SEEDS; do
  python3 batch_inference.py \
    --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
    --input-cache ${INPUT_CACHE} \
    --subset val \
    --checkpoint outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen_seed${seed}/checkpoints/best.pt \
    --output outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen_seed${seed}/inference_val.jsonl \
    --device auto \
    --amp-mode bf16 \
    --batch-size 16 \
    --num-workers auto \
    --video-backbone videomae \
    --num-frames 16 \
    --rgb-size 224 \
    --audio-model microsoft/wavlm-large \
    --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
    --text-model FacebookAI/xlm-roberta-large \
    --task-mode confounded_7way \
    --text-policy full \
    --sample-rate 16000 \
    --max-audio-sec 6 \
    --benchmark-tag meld_gpu_tri_rgb_audio_text_frozen_seed${seed}
done
```

测试集推理：

```bash
SEEDS="13 17 23 42 3407"
INPUT_CACHE=outputs/benchmarks/meld/input_cache/rgb16_audio_text_sr16000_len6_v1

for seed in $SEEDS; do
  python3 batch_inference.py \
    --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
    --input-cache ${INPUT_CACHE} \
    --subset test \
    --checkpoint outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen_seed${seed}/checkpoints/best.pt \
    --output outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen_seed${seed}/inference_test.jsonl \
    --device auto \
    --amp-mode bf16 \
    --batch-size 16 \
    --num-workers auto \
    --video-backbone videomae \
    --num-frames 16 \
    --rgb-size 224 \
    --audio-model microsoft/wavlm-large \
    --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
    --text-model FacebookAI/xlm-roberta-large \
    --task-mode confounded_7way \
    --text-policy full \
    --sample-rate 16000 \
    --max-audio-sec 6 \
    --benchmark-tag meld_gpu_tri_rgb_audio_text_frozen_seed${seed}
done
```

### 结果如何看

```bash
python3 aggregate_multi_seed_results.py \
  --group tri_frozen=outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen_seed* \
  --output-dir outputs/benchmarks/meld/reports/tri_frozen_5seed

python3 -m json.tool outputs/benchmarks/meld/reports/tri_frozen_5seed/multi_seed_summary.json
```

重点看：

- 相比实验一，`test macro_f1 mean` 是否提升
- `95% CI` 是否与实验一分开
- 每个 seed 的提升方向是否大体一致

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

这是实验三的正式论文级运行口径。它不需要旧中间 shard；如果想减少 GPU 服务器 CPU 压力，直接复用实验二同一套 `rgb16` 输入缓存。
训练、`val` 推理、`test` 推理分别按下面三个 loop 执行，全部使用同一组 `5` 个 seed。
当前 GPU 主线已经把媒体 ingress 挪到 `DataLoader worker`，默认推荐 `--num-workers auto`。但你当前这类 `K100_AI`/旧内核服务器已经实测在 worker 模式下会于首个 batch 附近触发 `Segmentation fault (core dumped)`，因此本机应把下面命令里的 `--num-workers auto` 改成 `--num-workers 0`。

训练：

```bash
SEEDS="13 17 23 42 3407"
INPUT_CACHE=outputs/benchmarks/meld/input_cache/rgb16_audio_text_sr16000_len6_v1

for seed in $SEEDS; do
  rm -rf outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text_seed${seed}

  python3 train.py \
    --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
    --input-cache ${INPUT_CACHE} \
    --output-dir outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text_seed${seed} \
    --epochs 40 \
    --device auto \
    --amp-mode bf16 \
    --batch-size 16 \
    --num-workers auto \
    --video-backbone videomae \
    --num-frames 16 \
    --rgb-size 224 \
    --audio-model microsoft/wavlm-large \
    --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
    --text-model FacebookAI/xlm-roberta-large \
    --fusion-mode gated_text \
    --task-mode confounded_7way \
    --text-policy full \
    --sample-rate 16000 \
    --max-audio-sec 6 \
    --monitor val_f1 \
    --early-stop-patience 12 \
    --early-stop-min-epochs 12 \
    --early-stop-min-delta 0.002 \
    --early-stop-after-lr-drops 2 \
    --lr-scheduler plateau \
    --lr-plateau-patience 4 \
    --lr-plateau-factor 0.5 \
    --lr-min 1e-6 \
    --seed ${seed} \
    --benchmark-tag meld_gpu_upper_rgb_audio_text_seed${seed}
done
```

验证集推理：

```bash
SEEDS="13 17 23 42 3407"
INPUT_CACHE=outputs/benchmarks/meld/input_cache/rgb16_audio_text_sr16000_len6_v1

for seed in $SEEDS; do
  python3 batch_inference.py \
    --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
    --input-cache ${INPUT_CACHE} \
    --subset val \
    --checkpoint outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text_seed${seed}/checkpoints/best.pt \
    --output outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text_seed${seed}/inference_val.jsonl \
    --device auto \
    --amp-mode bf16 \
    --batch-size 16 \
    --num-workers auto \
    --video-backbone videomae \
    --num-frames 16 \
    --rgb-size 224 \
    --audio-model microsoft/wavlm-large \
    --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
    --text-model FacebookAI/xlm-roberta-large \
    --task-mode confounded_7way \
    --text-policy full \
    --sample-rate 16000 \
    --max-audio-sec 6 \
    --benchmark-tag meld_gpu_upper_rgb_audio_text_seed${seed}
done
```

测试集推理：

```bash
SEEDS="13 17 23 42 3407"
INPUT_CACHE=outputs/benchmarks/meld/input_cache/rgb16_audio_text_sr16000_len6_v1

for seed in $SEEDS; do
  python3 batch_inference.py \
    --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
    --input-cache ${INPUT_CACHE} \
    --subset test \
    --checkpoint outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text_seed${seed}/checkpoints/best.pt \
    --output outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text_seed${seed}/inference_test.jsonl \
    --device auto \
    --amp-mode bf16 \
    --batch-size 16 \
    --num-workers auto \
    --video-backbone videomae \
    --num-frames 16 \
    --rgb-size 224 \
    --audio-model microsoft/wavlm-large \
    --audio-model-revision e4e472c491084b2c6fb9736099130aa805159c62 \
    --text-model FacebookAI/xlm-roberta-large \
    --task-mode confounded_7way \
    --text-policy full \
    --sample-rate 16000 \
    --max-audio-sec 6 \
    --benchmark-tag meld_gpu_upper_rgb_audio_text_seed${seed}
done
```

### 结果如何看

```bash
python3 aggregate_multi_seed_results.py \
  --group upper_rgb=outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text_seed* \
  --output-dir outputs/benchmarks/meld/reports/upper_rgb_5seed

python3 -m json.tool outputs/benchmarks/meld/reports/upper_rgb_5seed/multi_seed_summary.json
```

重点看：

- 相比实验二，`test macro_f1 mean` 是否继续提升
- `95% CI` 与实验二是否有明显分离
- 各 seed 的提升方向是否一致到足以支持“值得继续上 dual 分支”的判断

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

这是实验四的正式论文级运行口径。它对应当前 GPU 版 `dual(torch_motion + RGB)` 主线，不要再混用旧 CPU 光流缓存命令；正式加速方式改用下面的 `rgb32` 输入缓存。
训练、`val` 推理、`test` 推理分别按下面三个 loop 执行，全部使用同一组 `5` 个 seed。
当前 GPU 主线已经把媒体 ingress 挪到 `DataLoader worker`，默认推荐 `--num-workers auto`。如果你的服务器在 worker 模式下于首个 batch 附近触发 `Segmentation fault (core dumped)`，应把下面命令里的 `--num-workers auto` 改成 `--num-workers 0`。

训练：

```bash
SEEDS="13 17 23 42 3407"
INPUT_CACHE=outputs/benchmarks/meld/input_cache/rgb32_audio_text_sr16000_len6_v1

for seed in $SEEDS; do
  rm -rf outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion_seed${seed}

  python3 train.py \
    --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
    --input-cache ${INPUT_CACHE} \
    --output-dir outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion_seed${seed} \
    --epochs 60 \
    --device auto \
    --amp-mode bf16 \
    --batch-size 16 \
    --num-workers auto \
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
    --sample-rate 16000 \
    --max-audio-sec 6 \
    --monitor val_f1 \
    --early-stop-patience 12 \
    --early-stop-min-epochs 12 \
    --early-stop-min-delta 0.002 \
    --early-stop-after-lr-drops 2 \
    --lr-scheduler plateau \
    --lr-plateau-patience 4 \
    --lr-plateau-factor 0.5 \
    --lr-min 1e-6 \
    --seed ${seed} \
    --benchmark-tag meld_gpu_upper_dual_torch_motion_seed${seed}
done
```

验证集推理：

```bash
SEEDS="13 17 23 42 3407"
INPUT_CACHE=outputs/benchmarks/meld/input_cache/rgb32_audio_text_sr16000_len6_v1

for seed in $SEEDS; do
  python3 batch_inference.py \
    --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
    --input-cache ${INPUT_CACHE} \
    --subset val \
    --checkpoint outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion_seed${seed}/checkpoints/best.pt \
    --output outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion_seed${seed}/inference_val.jsonl \
    --device auto \
    --amp-mode bf16 \
    --batch-size 16 \
    --num-workers auto \
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
    --sample-rate 16000 \
    --max-audio-sec 6 \
    --benchmark-tag meld_gpu_upper_dual_torch_motion_seed${seed}
done
```

测试集推理：

```bash
SEEDS="13 17 23 42 3407"
INPUT_CACHE=outputs/benchmarks/meld/input_cache/rgb32_audio_text_sr16000_len6_v1

for seed in $SEEDS; do
  python3 batch_inference.py \
    --split-manifest outputs/benchmarks/meld/splits/default_manifest.filtered.json \
    --input-cache ${INPUT_CACHE} \
    --subset test \
    --checkpoint outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion_seed${seed}/checkpoints/best.pt \
    --output outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion_seed${seed}/inference_test.jsonl \
    --device auto \
    --amp-mode bf16 \
    --batch-size 16 \
    --num-workers auto \
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
    --sample-rate 16000 \
    --max-audio-sec 6 \
    --benchmark-tag meld_gpu_upper_dual_torch_motion_seed${seed}
done
```

### 结果如何看

```bash
python3 aggregate_multi_seed_results.py \
  --group dual_upper=outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion_seed* \
  --output-dir outputs/benchmarks/meld/reports/dual_upper_5seed

python3 -m json.tool outputs/benchmarks/meld/reports/dual_upper_5seed/multi_seed_summary.json
```

重点看：

- `test macro_f1 mean` 是否超过实验三
- `95% CI` 与实验三是否分离
- 相邻对比里是否满足 “均值为正 + p<0.05 + 至少 4/5 个 seed 为正增益”

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

## 四组结果的正式统计汇总

当四组实验都把 `5` 个 seed 的 train / val / test 跑完以后，统一执行下面这个汇总命令：

```bash
python3 aggregate_multi_seed_results.py \
  --group practical=outputs/benchmarks/meld/run_gpu_practical_no_video_seed* \
  --group tri_frozen=outputs/benchmarks/meld/run_gpu_tri_rgb_audio_text_frozen_seed* \
  --group upper_rgb=outputs/benchmarks/meld/run_gpu_upper_rgb_audio_text_seed* \
  --group dual_upper=outputs/benchmarks/meld/run_gpu_upper_dual_torch_motion_seed* \
  --output-dir outputs/benchmarks/meld/reports/full_comparison_5seed

python3 -m json.tool outputs/benchmarks/meld/reports/full_comparison_5seed/multi_seed_summary.json
python3 -m json.tool outputs/benchmarks/meld/reports/full_comparison_5seed/pairwise_significance.json
```

正式对外报告时，优先使用：

- `multi_seed_summary.json`
- `pairwise_significance.json`
- `multi_seed_summary.md`

## 结果比较顺序

真正最关键的不是某一组单点最高分，而是下面三条增益链条是否成立：

- `实验二 > 实验一`
  说明 RGB 模态有实际贡献
- `实验三 > 实验二`
  说明放开 RGB/A/T 编码器有实际贡献
- `实验四 > 实验三`
  说明 GPU 动态分支相对于 RGB-only 上限有实际贡献

只有当下面三条同时成立时，才建议在论文或报告里写成“显著提升”：

- `test macro_f1` 的 `mean` 为正增益
- `pairwise_significance.json` 里的 `p_value < 0.05`
- 至少 `4/5` 个共享 seed 呈现正增益

如果均值更高但不满足这三条，就只能写成：

- “数值上更高，但尚未获得统计显著性支持”
- `实验四 > 实验三`
  说明当前 GPU 动态分支有实际贡献

如果某一级没有明显提升，就应该重新评估那个分支或训练成本是否值得保留。
