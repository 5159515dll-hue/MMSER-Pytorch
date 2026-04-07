"""Manifest 驱动的主线训练入口。

这个文件负责把命令行参数、manifest 数据集、gpu_stream 预处理器、
多模态模型、优化器和 early stop 控制器串起来，形成完整的训练流程。
阅读顺序建议从 `main()` 开始，再回看 `_run_phase()` 理解单个 epoch
里一个 batch 是如何完成“取数据 -> 预处理 -> 前向 -> 反向 -> 统计”的。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# HuggingFace tokenizers may warn when DataLoader workers fork after tokenizers were used.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TASK_MODE_CHOICES = ("confounded_7way", "within_speaker")

torch: Any = None
DataLoader: Any = None
tqdm: Any = None
AudioAugConfig: Any = None
augment_wav: Any = None
normalize_wav: Any = None
ManifestIngressConfig: Any = None
StreamingManifestDataset: Any = None
CachedManifestDataset: Any = None
cache_manifest_text_tokens: Any = None
collate_manifest_items: Any = None
GpuStreamConfig: Any = None
GpuStreamPreprocessor: Any = None
autocast_context: Any = None
ccc: Any = None
prepare_manifest_items_for_task: Any = None
print_manifest_label_hist: Any = None
build_paper_grade: Any = None
build_run_contract: Any = None
build_run_provenance: Any = None
FLOW_VIDEO_ENCODER_VARIANT: Any = None
make_dataloader_worker_init_fn: Any = None
make_torch_generator: Any = None
resolve_ablation_flags: Any = None
save_metrics_and_plots: Any = None
set_seed: Any = None
to_jsonable: Any = None
write_best_val_inference_outputs: Any = None
write_results_summary: Any = None
build_validity_summary: Any = None
filter_manifest_items_for_task: Any = None
load_split_manifest: Any = None
manifest_sha256: Any = None
map_label_to_task_index: Any = None
resolve_task_label_names: Any = None
resolve_task_mode: Any = None
select_manifest_items: Any = None
classification_summary: Any = None
speaker_majority_baseline: Any = None
speaker_only_baseline: Any = None
FusionClassifier: Any = None
ProsodyConfig: Any = None
extract_prosody_features_gpu: Any = None
detect_runtime: Any = None
resolve_amp_mode: Any = None
resolve_batch_size: Any = None
resolve_prefetch_factor: Any = None
resolve_worker_count: Any = None
select_device: Any = None
resolve_text_policy: Any = None
build_input_cache_contract: Any = None
load_input_cache_meta: Any = None
validate_input_cache_contract: Any = None


def _ensure_project_root_on_path() -> None:
    """确保仓库根目录在 `sys.path` 中。

    训练入口经常会被用户从不同工作目录触发。这里把当前文件所在目录
    放到导入搜索路径前面，保证 `hf_compat.py`、`models.py` 等根目录模块
    都能被稳定导入。
    """

    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()

from hf_compat import ensure_transformers_torch_compat
from hf_loading import resolve_hf_pretrained_source
from run_store import RunAttemptStore, SignalCapture
from training_control import EarlyStopConfig, EarlyStopController


def _lazy_runtime_imports() -> None:
    """按需导入训练阶段真正需要的大模块。

    这样做有两个目的：
    1. `train.py --help` 这类轻量命令不需要提前加载整套 torch/transformers。
    2. 避免模块导入阶段就触发较重的 CUDA / Hugging Face 初始化。
    """

    global torch, DataLoader, tqdm
    global AudioAugConfig, augment_wav, normalize_wav
    global ManifestIngressConfig, StreamingManifestDataset, CachedManifestDataset, cache_manifest_text_tokens, collate_manifest_items
    global GpuStreamConfig, GpuStreamPreprocessor
    global autocast_context, ccc, prepare_manifest_items_for_task, print_manifest_label_hist
    global build_paper_grade, build_run_contract, build_run_provenance, FLOW_VIDEO_ENCODER_VARIANT
    global make_dataloader_worker_init_fn, make_torch_generator
    global resolve_ablation_flags, save_metrics_and_plots, set_seed, to_jsonable
    global write_best_val_inference_outputs, write_results_summary
    global build_validity_summary, filter_manifest_items_for_task, load_split_manifest
    global manifest_sha256, map_label_to_task_index, resolve_task_label_names, resolve_task_mode, select_manifest_items
    global classification_summary, speaker_majority_baseline, speaker_only_baseline
    global FusionClassifier, ProsodyConfig, extract_prosody_features_gpu
    global detect_runtime, resolve_amp_mode, resolve_batch_size, resolve_prefetch_factor, resolve_worker_count, select_device
    global resolve_text_policy
    global build_input_cache_contract, load_input_cache_meta, validate_input_cache_contract
    # 这里用 `torch is not None` 作为“是否已经完成整批导入”的哨兵值，
    # 防止在一个进程里重复执行整套导入和全局绑定。
    if torch is not None:
        return

    import torch as _torch
    from torch.utils.data import DataLoader as _DataLoader
    from tqdm import tqdm as _tqdm

    from audio_aug import AudioAugConfig as _AudioAugConfig, augment_wav as _augment_wav, normalize_wav as _normalize_wav
    from data import (
        ManifestIngressConfig as _ManifestIngressConfig,
        StreamingManifestDataset as _StreamingManifestDataset,
        CachedManifestDataset as _CachedManifestDataset,
        cache_manifest_text_tokens as _cache_manifest_text_tokens,
        collate_manifest_items as _collate_manifest_items,
    )
    from gpu_stream import GpuStreamConfig as _GpuStreamConfig, GpuStreamPreprocessor as _GpuStreamPreprocessor
    from mainline_utils import (
        autocast_context as _autocast_context,
        build_paper_grade as _build_paper_grade,
        build_run_contract as _build_run_contract,
        build_run_provenance as _build_run_provenance,
        ccc as _ccc,
        FLOW_VIDEO_ENCODER_VARIANT as _FLOW_VIDEO_ENCODER_VARIANT,
        make_dataloader_worker_init_fn as _make_dataloader_worker_init_fn,
        make_torch_generator as _make_torch_generator,
        prepare_manifest_items_for_task as _prepare_manifest_items_for_task,
        print_manifest_label_hist as _print_manifest_label_hist,
        resolve_ablation_flags as _resolve_ablation_flags,
        save_metrics_and_plots as _save_metrics_and_plots,
        set_seed as _set_seed,
        to_jsonable as _to_jsonable,
        write_best_val_inference_outputs as _write_best_val_inference_outputs,
        write_results_summary as _write_results_summary,
    )
    from manifest_utils import (
        build_validity_summary as _build_validity_summary,
        filter_manifest_items_for_task as _filter_manifest_items_for_task,
        load_split_manifest as _load_split_manifest,
        manifest_sha256 as _manifest_sha256,
        map_label_to_task_index as _map_label_to_task_index,
        resolve_task_label_names as _resolve_task_label_names,
        resolve_task_mode as _resolve_task_mode,
        select_manifest_items as _select_manifest_items,
    )
    from metrics_utils import (
        classification_summary as _classification_summary,
        speaker_majority_baseline as _speaker_majority_baseline,
        speaker_only_baseline as _speaker_only_baseline,
    )
    from models import FusionClassifier as _FusionClassifier
    from prosody import ProsodyConfig as _ProsodyConfig, extract_prosody_features_gpu as _extract_prosody_features_gpu
    from runtime_adapt import (
        detect_runtime as _detect_runtime,
        resolve_amp_mode as _resolve_amp_mode,
        resolve_batch_size as _resolve_batch_size,
        resolve_prefetch_factor as _resolve_prefetch_factor,
        resolve_worker_count as _resolve_worker_count,
        select_device as _select_device,
    )
    from text_policy_utils import resolve_text_policy as _resolve_text_policy
    from input_cache import (
        build_input_cache_contract as _build_input_cache_contract,
        load_input_cache_meta as _load_input_cache_meta,
        validate_input_cache_contract as _validate_input_cache_contract,
    )

    torch = _torch
    DataLoader = _DataLoader
    tqdm = _tqdm
    AudioAugConfig = _AudioAugConfig
    augment_wav = _augment_wav
    normalize_wav = _normalize_wav
    ManifestIngressConfig = _ManifestIngressConfig
    StreamingManifestDataset = _StreamingManifestDataset
    CachedManifestDataset = _CachedManifestDataset
    cache_manifest_text_tokens = _cache_manifest_text_tokens
    collate_manifest_items = _collate_manifest_items
    GpuStreamConfig = _GpuStreamConfig
    GpuStreamPreprocessor = _GpuStreamPreprocessor
    autocast_context = _autocast_context
    build_paper_grade = _build_paper_grade
    build_run_contract = _build_run_contract
    build_run_provenance = _build_run_provenance
    FLOW_VIDEO_ENCODER_VARIANT = _FLOW_VIDEO_ENCODER_VARIANT
    ccc = _ccc
    make_dataloader_worker_init_fn = _make_dataloader_worker_init_fn
    make_torch_generator = _make_torch_generator
    prepare_manifest_items_for_task = _prepare_manifest_items_for_task
    print_manifest_label_hist = _print_manifest_label_hist
    resolve_ablation_flags = _resolve_ablation_flags
    save_metrics_and_plots = _save_metrics_and_plots
    set_seed = _set_seed
    to_jsonable = _to_jsonable
    write_best_val_inference_outputs = _write_best_val_inference_outputs
    write_results_summary = _write_results_summary
    build_validity_summary = _build_validity_summary
    filter_manifest_items_for_task = _filter_manifest_items_for_task
    load_split_manifest = _load_split_manifest
    manifest_sha256 = _manifest_sha256
    map_label_to_task_index = _map_label_to_task_index
    resolve_task_label_names = _resolve_task_label_names
    resolve_task_mode = _resolve_task_mode
    select_manifest_items = _select_manifest_items
    classification_summary = _classification_summary
    speaker_majority_baseline = _speaker_majority_baseline
    speaker_only_baseline = _speaker_only_baseline
    FusionClassifier = _FusionClassifier
    ProsodyConfig = _ProsodyConfig
    extract_prosody_features_gpu = _extract_prosody_features_gpu
    detect_runtime = _detect_runtime
    resolve_amp_mode = _resolve_amp_mode
    resolve_batch_size = _resolve_batch_size
    resolve_prefetch_factor = _resolve_prefetch_factor
    resolve_worker_count = _resolve_worker_count
    select_device = _select_device
    resolve_text_policy = _resolve_text_policy
    build_input_cache_contract = _build_input_cache_contract
    load_input_cache_meta = _load_input_cache_meta
    validate_input_cache_contract = _validate_input_cache_contract


def parse_args() -> argparse.Namespace:
    """定义并解析训练 CLI 参数。

    参数分三类：
    - 任务/数据口径：例如 `task_mode`、`speaker_id`、`text_policy`
    - 训练控制：例如 epoch、学习率、early stop、AMP
    - 兼容参数：旧脚本可能还会传入的隐藏参数，这些参数会在后面统一校验
    """

    p = argparse.ArgumentParser(description="Train the manifest-driven gpu_stream mainline")
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--task-mode", type=str, default="confounded_7way", choices=list(TASK_MODE_CHOICES))
    p.add_argument("--speaker-id", type=str, default="")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=str, default="auto")
    p.add_argument("--num-workers", type=str, default="auto")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--monitor", type=str, default="val_f1", choices=["val_acc", "val_f1"])
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--early-stop-min-epochs", type=int, default=10)
    p.add_argument("--early-stop-min-delta", type=float, default=0.002)
    p.add_argument("--early-stop-after-lr-drops", type=int, default=2)
    p.add_argument("--lr-scheduler", type=str, default="plateau", choices=["none", "plateau"])
    p.add_argument("--lr-plateau-patience", type=int, default=4)
    p.add_argument("--lr-plateau-factor", type=float, default=0.5)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/motion_prosody"))
    p.add_argument("--benchmark-tag", type=str, default="default")
    p.add_argument("--input-cache", type=Path, default=None, help="Optional mainline input cache directory built by build_mainline_input_cache.py")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--freeze-audio", action="store_true")
    p.add_argument("--freeze-video", action="store_true")
    p.add_argument("--freeze-flow", action="store_true")
    p.add_argument("--freeze-rgb", action="store_true")
    p.add_argument("--freeze-prosody", action="store_true")
    p.add_argument("--video-backbone", type=str, default="dual", choices=["flow", "videomae", "dual"])
    p.add_argument("--video-model", type=str, default="MCG-NJU/videomae-large")
    p.add_argument("--audio-model", type=str, default="microsoft/wavlm-large")
    p.add_argument("--audio-model-revision", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--fusion-mode", type=str, default="gated_text", choices=["gated_text", "concat"])
    p.add_argument("--gate-temperature", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--gate-scale", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--delta-scale", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--modality-dropout", type=float, default=0.1)
    p.add_argument("--text-model", type=str, default="xlm-roberta-large")
    p.add_argument("--text-policy", type=str, default="full", choices=["full", "mask_emotion_cues", "drop"])
    p.add_argument("--freeze-text", action="store_true")
    p.add_argument("--max-text-len", type=int, default=128)
    p.add_argument("--audio-aug", action="store_true")
    p.set_defaults(recompute_prosody_on_aug=True)
    p.add_argument(
        "--no-recompute-prosody-on-aug",
        dest="recompute_prosody_on_aug",
        action="store_false",
        help="When using --audio-aug, keep the precomputed prosody vectors.",
    )
    p.add_argument("--prosody-no-pitch", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "text-only", "audio-only", "video-only", "no-text", "no-audio", "no-video"],
    )
    p.add_argument("--num-frames", type=int, default=64)
    p.add_argument("--flow-size", type=int, default=112, help=argparse.SUPPRESS)
    p.add_argument("--rgb-size", type=int, default=224, help=argparse.SUPPRESS)
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--max-audio-sec", type=float, default=6.0)
    p.add_argument("--audio-backend", type=str, default="auto", choices=["auto", "torchaudio", "soundfile"], help=argparse.SUPPRESS)
    p.add_argument("--video-decode-backend", type=str, default="auto", choices=["auto", "decord", "cpu"], help=argparse.SUPPRESS)
    p.add_argument("--flow-backend", type=str, default="torch_motion", choices=["torch_motion", "legacy"], help=argparse.SUPPRESS)
    p.add_argument("--amp-mode", type=str, default="auto", choices=["auto", "off", "fp16", "bf16"])
    p.add_argument("--use-intensity", action="store_true")
    p.add_argument("--intensity-loss", type=str, default="mse", choices=["mse", "mae"])
    p.add_argument("--intensity-weight", type=float, default=1.0)

    # Hidden compatibility args. They either translate to current behavior or fail fast.
    p.add_argument("--zero-video", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--zero-audio", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--zero-text", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cached-dataset", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--feature-cache", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--pipeline", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--cache-mode", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--ram-cache-size", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--train-split", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--split-mode", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--prefetch-factor", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--persistent-workers", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--pin-memory", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--amp", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def _validate_compat_args(args: argparse.Namespace) -> None:
    """拒绝已经退役的旧参数组合。

    主线现在只接受 manifest + gpu_stream 路径。这里专门把旧版缓存数据集、
    pipeline 开关、手工性能开关等参数拦下来，避免用户以为这些参数还生效，
    从而导致“实际执行的实验”和“命令行看起来描述的实验”不一致。
    """

    if args.cached_dataset is not None or args.feature_cache is not None:
        raise RuntimeError(
            "Mainline training no longer supports --cached-dataset or --feature-cache. "
            "Use --split-manifest with the gpu_stream path."
        )
    pipeline = str(args.pipeline or "").strip().lower()
    if pipeline and pipeline not in {"auto", "gpu_stream"}:
        raise RuntimeError(f"Unsupported retired --pipeline value: {args.pipeline}")
    cache_mode = str(args.cache_mode or "").strip().lower()
    if cache_mode and cache_mode not in {"none"}:
        raise RuntimeError(
            "Mainline training no longer supports cache modes other than --cache-mode none. "
            "The manifest/gpu_stream path is now the only active runtime."
        )
    if args.ram_cache_size not in {None, 0}:
        raise RuntimeError("--ram-cache-size is retired in gpu_stream-only training.")
    for name in ("train_split", "split_mode", "prefetch_factor"):
        if getattr(args, name) is not None:
            raise RuntimeError(f"--{name.replace('_', '-')} is retired in gpu_stream-only training.")
    if bool(args.persistent_workers) or bool(args.pin_memory) or bool(args.amp):
        raise RuntimeError("Legacy performance toggles are retired; mainline training now resolves them automatically.")
    if int(args.early_stop_patience) < 0:
        raise RuntimeError("--early-stop-patience must be >= 0.")
    if int(args.early_stop_min_epochs) < 0:
        raise RuntimeError("--early-stop-min-epochs must be >= 0.")
    if float(args.early_stop_min_delta) < 0.0:
        raise RuntimeError("--early-stop-min-delta must be >= 0.")
    if int(args.early_stop_after_lr_drops) < 0:
        raise RuntimeError("--early-stop-after-lr-drops must be >= 0.")
    if str(args.lr_scheduler) == "none" and int(args.early_stop_after_lr_drops) > 0:
        raise RuntimeError("--early-stop-after-lr-drops requires --lr-scheduler plateau.")
    if int(args.lr_plateau_patience) < 0:
        raise RuntimeError("--lr-plateau-patience must be >= 0.")
    if not 0.0 < float(args.lr_plateau_factor) < 1.0:
        raise RuntimeError("--lr-plateau-factor must be in (0, 1).")
    if float(args.lr_min) < 0.0:
        raise RuntimeError("--lr-min must be >= 0.")


def _load_tokenizer(model_name: str) -> Any:
    """加载文本 tokenizer。

    这里先调用 `ensure_transformers_torch_compat()`，是为了在旧版 torch
    全局环境里补齐 `transformers` 依赖的 pytree API，再走真正的
    `AutoTokenizer.from_pretrained(...)`。
    """

    try:
        ensure_transformers_torch_compat()
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required for text-enabled training.") from e
    source, load_kwargs = resolve_hf_pretrained_source(str(model_name))
    return AutoTokenizer.from_pretrained(source, **load_kwargs)


def _startup_log(message: str) -> None:
    """打印启动阶段日志，帮助定位静默初始化卡点。"""

    print(f"[startup] {message}", flush=True)


@contextmanager
def _startup_stage(label: str) -> Any:
    """为启动阶段打印开始/完成/失败与耗时。"""

    _startup_log(f"{label} [start]")
    started = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        _startup_log(f"{label} [failed after {elapsed:.1f}s]")
        raise
    elapsed = time.perf_counter() - started
    _startup_log(f"{label} [done in {elapsed:.1f}s]")


def _current_lr(optimizer: Any) -> float:
    """读取优化器第一个参数组的当前学习率。"""

    param_groups = getattr(optimizer, "param_groups", [])
    if not param_groups:
        return 0.0
    return float(param_groups[0].get("lr", 0.0))


class NumericStabilityError(RuntimeError):
    """Raised when a training batch or epoch produces non-finite values."""


class DeterminismCompatibilityError(RuntimeError):
    """Raised when strict deterministic mode hits an unsupported CUDA op."""


def _assert_finite_tensor(name: str, value: Any, *, phase: str) -> None:
    """在训练过程中强制拦截 NaN/Inf。

    论文级协议要求一旦出现非有限值就立刻失败，而不是继续把异常 run
    当成有效结果写盘。因此 logits、loss、回归输出等关键张量都会经过这里。
    """

    if value is None:
        return
    if isinstance(value, torch.Tensor):
        if bool(torch.isfinite(value).all().item()):
            return
        raise NumericStabilityError(f"{phase}: non-finite tensor detected in {name}")
    scalar = float(value)
    if math.isfinite(scalar):
        return
    raise NumericStabilityError(f"{phase}: non-finite scalar detected in {name}: {scalar}")


def _assert_finite_stats(name: str, stats: dict[str, Any]) -> None:
    """检查一个 epoch 汇总指标是否仍然是有限值。"""

    for key in ("loss", "accuracy", "macro_f1", "prepare_sec", "mse", "mae", "ccc"):
        if key not in stats:
            continue
        value = stats.get(key, None)
        if value is None:
            continue
        scalar = float(value)
        if not math.isfinite(scalar):
            raise NumericStabilityError(f"{name}: non-finite summary metric {key}={scalar}")


def _wrap_runtime_error(exc: RuntimeError, *, phase: str) -> None:
    """把底层 RuntimeError 重新分类成更明确的主线异常。

    目前最重要的分支是“严格 deterministic 模式遇到不支持的 CUDA 算子”。
    这样主循环可以把它记成 `failed_determinism`，而不是笼统的普通报错。
    """

    message = str(exc)
    if "does not have a deterministic implementation" in message:
        raise DeterminismCompatibilityError(f"{phase}: {message}") from exc
    raise exc


def _run_phase(
    *,
    phase: str,
    model: FusionClassifier,
    loader: DataLoader,
    preprocessor: GpuStreamPreprocessor,
    tokenizer: Any | None,
    args: argparse.Namespace,
    device: torch.device,
    amp_mode: str,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    prosody_cfg: ProsodyConfig,
    aug_cfg: AudioAugConfig,
) -> dict[str, Any]:
    """执行一个训练或验证 phase，并返回该 phase 的汇总统计。

    这个函数是训练主线里最值得阅读的部分。它完成的事情依次是：
    1. 通过 `GpuStreamPreprocessor` 把原始 manifest item 变成张量 batch
    2. 在训练态可选地做音频增强与韵律重算
    3. 执行模型前向，得到分类 logits 和可选强度回归输出
    4. 计算损失，并在训练态执行反向传播和优化器更新
    5. 累积分类/回归指标，以及验证阶段逐样本记录
    """

    _lazy_runtime_imports()
    is_train = optimizer is not None
    model.train(is_train)
    label_names = list(getattr(model, "_label_names", []))

    total = 0
    correct = 0
    loss_sum = 0.0
    prepare_sec = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    reg_true: list[float] = []
    reg_pred: list[float] = []
    records: list[dict[str, Any]] = []
    backend_logged = False

    # 训练和验证都复用同一个 phase 实现；是否显示进度条只取决于 tqdm 是否可用。
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=phase, unit="batch", dynamic_ncols=True)

    for batch_items in iterator:
        prep_started = time.perf_counter()
        # `prepare_batch` 负责把 list[manifest item] 变成模型消费的整批张量，
        # 其中会统一补齐文本 token、音频长度、prosody、flow/rgb 等字段。
        batch = preprocessor.prepare_batch(
            batch_items,
            tokenizer=tokenizer,
            text_policy=str(args.text_policy),
            max_text_len=int(args.max_text_len),
        )
        prepare_sec += float(time.perf_counter() - prep_started)

        # 每个 phase 只打印一次后端摘要，避免日志被每个 batch 淹没。
        if not backend_logged:
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "gpu_stream_backends": preprocessor.backend_summary(),
                        "gpu_stream_prepare_stats": preprocessor.consume_prepare_stats(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            backend_logged = True

        wav = batch["audio"]
        audio_lens = batch["audio_lens"]
        prosody = batch["prosody"]
        flow = batch.get("flow", None)
        rgb = batch.get("rgb", None)
        text_inputs = batch.get("text_inputs", None)
        labels = batch["labels"]

        # 训练态的音频增强只影响当前 batch；如果增强改变了波形，
        # 且用户允许，就要重新提取 prosody，保证韵律特征与增强后的音频一致。
        if is_train and bool(args.audio_aug) and (not bool(args.zero_audio)):
            wav = normalize_wav(wav, target_rms=0.1)
            wav = augment_wav(wav, aug_cfg)
            wav = normalize_wav(wav, target_rms=0.1)
            if bool(args.recompute_prosody_on_aug):
                prosody = extract_prosody_features_gpu(wav, prosody_cfg, lengths=audio_lens).to(torch.float32)

        # 训练时需要保留计算图，验证时显式关闭梯度，减少显存和计算开销。
        grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
        with grad_ctx, autocast_context(device, amp_mode):
            if bool(args.use_intensity):
                logits, pred_intensity = model(
                    flow,
                    wav,
                    prosody,
                    text_inputs=text_inputs,
                    return_intensity=True,
                    rgb=rgb,
                    audio_lens=audio_lens,
                )
            else:
                pred_intensity = None
                logits = model(
                    flow,
                    wav,
                    prosody,
                    text_inputs=text_inputs,
                    rgb=rgb,
                    audio_lens=audio_lens,
                )

            _assert_finite_tensor("logits", logits, phase=phase)
            _assert_finite_tensor("pred_intensity", pred_intensity, phase=phase)
            # 主任务始终是情绪分类；强度回归如果启用，只是在分类损失上追加一项。
            loss = loss_fn(logits, labels)
            if bool(args.use_intensity):
                intensity = batch["intensity"].to(torch.float32)
                intensity_mask = batch["intensity_mask"]
                # 有些样本可能没有强度标注，所以这里只在 mask 为真的子集上计算回归损失。
                if bool(intensity_mask.any().item()):
                    pred_f = pred_intensity.to(torch.float32)[intensity_mask]
                    gt_f = intensity[intensity_mask]
                    if str(args.intensity_loss).strip().lower() == "mae":
                        reg_loss = torch.abs(pred_f - gt_f).mean()
                    else:
                        reg_loss = torch.square(pred_f - gt_f).mean()
                    loss = loss + float(args.intensity_weight) * reg_loss
            _assert_finite_tensor("loss", loss, phase=phase)

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            try:
                # fp16 时优先走 GradScaler；bf16/off 则直接 backward。
                if scaler is not None and bool(scaler.is_enabled()):
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            except RuntimeError as exc:
                _wrap_runtime_error(exc, phase=phase)

        pred = logits.argmax(dim=1)
        batch_size = int(labels.shape[0])
        total += batch_size
        correct += int((pred == labels).sum().item())
        loss_sum += float(loss.item()) * batch_size

        labels_cpu = labels.detach().cpu().tolist()
        pred_cpu = pred.detach().cpu().tolist()
        y_true.extend(int(x) for x in labels_cpu)
        y_pred.extend(int(x) for x in pred_cpu)

        if bool(args.use_intensity) and pred_intensity is not None:
            intensity_cpu = batch["intensity"].detach().cpu().to(torch.float32)
            mask_cpu = batch["intensity_mask"].detach().cpu()
            pred_intensity_cpu = pred_intensity.detach().cpu().to(torch.float32)
            for idx in range(batch_size):
                if bool(mask_cpu[idx].item()):
                    reg_true.append(float(intensity_cpu[idx].item()))
                    reg_pred.append(float(pred_intensity_cpu[idx].item()))

        if not is_train:
            # 验证阶段会把每个样本的预测结果记下来，
            # 后续最佳 epoch 的 `inference_val.jsonl` 就来自这里。
            probs = torch.softmax(logits, dim=1)
            pred_probs = probs.gather(1, pred.view(-1, 1)).squeeze(1).detach().cpu().tolist()
            gt_intensity = batch["intensity"].detach().cpu().to(torch.float32)
            mask_cpu = batch["intensity_mask"].detach().cpu()
            pred_intensity_cpu = pred_intensity.detach().cpu().to(torch.float32) if pred_intensity is not None else None
            for idx, stem in enumerate(batch["stems"]):
                rec = {
                    "stem": str(stem),
                    "status": "ok",
                    "label": label_names[int(labels_cpu[idx])],
                    "pred": label_names[int(pred_cpu[idx])],
                    "match": bool(int(labels_cpu[idx]) == int(pred_cpu[idx])),
                    "probability": float(pred_probs[idx]),
                    "speaker_id": str(batch["speaker_id"][idx]),
                }
                if bool(args.use_intensity):
                    if bool(mask_cpu[idx].item()):
                        rec["intensity_gt"] = float(gt_intensity[idx].item())
                        rec["pred_intensity"] = float(pred_intensity_cpu[idx].item()) if pred_intensity_cpu is not None else None
                    else:
                        rec["intensity_gt"] = None
                        rec["pred_intensity"] = None
                records.append(rec)

    # 循环结束后统一从逐样本标签恢复出宏 F1、混淆矩阵等分类摘要。
    summary = classification_summary(y_true, y_pred, label_names)

    mse = 0.0
    mae = 0.0
    ccc_value = 0.0
    intensity_n = int(len(reg_true))
    # 强度指标和分类指标分开统计，避免没有强度标签的样本污染结果。
    if reg_true and reg_pred:
        rt = torch.tensor(reg_true, dtype=torch.float32)
        rp = torch.tensor(reg_pred, dtype=torch.float32)
        diff = (rp - rt).to(torch.float64)
        mse = float((diff * diff).mean().item())
        mae = float(diff.abs().mean().item())
        ccc_value = ccc(rp, rt)

    stats = {
        "loss": float(loss_sum / max(1, total)),
        "accuracy": float(correct / max(1, total)),
        "macro_f1": float(summary.get("macro_f1", 0.0)),
        "class_summary": summary,
        "prepare_sec": float(prepare_sec),
        "records": records,
        "intensity_n": intensity_n,
        "mse": mse,
        "mae": mae,
        "ccc": ccc_value,
    }
    _assert_finite_stats(phase, stats)
    return stats


def main() -> None:
    """主训练入口。

    整体流程：
    1. 解析并校验参数
    2. 固定随机性策略，解析 manifest 与任务口径
    3. 构建训练/验证数据集、tokenizer、DataLoader
    4. 构建模型、优化器、调度器、预处理器
    5. 循环执行 train/val phase，按验证集规则选 `best.pt`
    6. 写出 `last.pt`、metrics、图表和结果摘要
    """

    args = parse_args()
    _validate_compat_args(args)
    _lazy_runtime_imports()

    # 先把文本策略和 ablation 统一解析成当前主线认可的“最终零化开关”，
    # 这样后面的 dataset / model / inference 记录才能保持一致。
    args.text_policy = resolve_text_policy(args.text_policy)
    zero_video, zero_audio, zero_text = resolve_ablation_flags(  # 函数内部会根据 text_policy 强制 zero_text
        ablation=str(args.ablation),    # 但这里的 zero_* 仍然以命令行参数为准，保持用户输入和最终协议的一致性。
        zero_video=bool(args.zero_video),   # 这样用户即使传了 --zero-video，但选了 text-only，也会被主线改成不 zero 视频，避免协议和实际执行不一致。
        zero_audio=bool(args.zero_audio),
        zero_text=bool(args.zero_text),
    )
    if str(args.text_policy) == "drop": # 当文本策略是 drop 时，强制 zero_text=True，无论用户命令行输入如何，确保协议一致性。
        zero_text = True
    args.zero_video = bool(zero_video)  # 这里把最终的 zero_* 结果写回 args，确保后续代码和协议里都用这个统一的版本，而不是用户原始输入的版本。
    args.zero_audio = bool(zero_audio)
    args.zero_text = bool(zero_text)

    # 训练一开始就固定随机种子和 deterministic policy，
    # 其结果会写入 provenance，后续推理/聚合也会用来验证协议一致性。
    deterministic_policy = set_seed(int(args.seed)) # 返回当前的 deterministic policy 设定，包含 torch 和 cudnn 的相关开关状态，供后续记录和验证使用。
    profile = detect_runtime(args.device)   # 返回当前的 runtime profile，包含设备类型、torch 版本、cuda 版本、cudnn 版本等信息，供后续记录和验证使用。
    device = select_device(args.device)
    resolved_amp_mode = resolve_amp_mode(str(args.amp_mode), profile)   # 返回实际使用的 AMP 模式，供后续训练流程和协议记录使用。
    print(f"Using device: {device}", flush=True)
    print(f"Runtime profile: {json.dumps(profile.to_jsonable(), ensure_ascii=False)}", flush=True)

    split_manifest_path = args.split_manifest.expanduser()  # 用户输入的 manifest 路径，支持 ~ 之类的 shell 风格路径，先展开成绝对路径再使用。
    if not split_manifest_path.exists():    # 如果展开后的路径不存在，就立刻报错，避免后续加载时出现模糊的文件不存在错误。
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
    manifest = load_split_manifest(split_manifest_path) # 加载 manifest 内容，得到一个 dict 结构，包含 train/val 划分和每个样本的元信息。
    manifest_hash = manifest_sha256(split_manifest_path)    # 计算 manifest 文件的 sha256 哈希值，作为该 manifest 的唯一标识，供后续协议记录和验证使用。
    manifest_summary = manifest.get("summary", {})  # 从 manifest 中提取 summary 字段，里面可能包含一些预计算的统计信息，例如原始数据的标签分布、时长分布等，供后续协议记录和验证使用。
    dataset_kind = str(manifest.get("dataset_kind", "") or "")

    task_mode = resolve_task_mode(args.task_mode)
    task_speaker_id = str(args.speaker_id).strip().upper() or None
    label_names = resolve_task_label_names(task_mode, task_speaker_id)
    validity_summary = build_validity_summary(manifest_summary, task_mode, task_speaker_id)
    paper_grade_reasons: list[str] = []
    if not bool(deterministic_policy.get("deterministic_algorithms_enabled", False)):
        paper_grade_reasons.append("deterministic_algorithms_disabled")
    if bool(deterministic_policy.get("cudnn_enabled", False)) and bool(deterministic_policy.get("cudnn_benchmark", False)):
        paper_grade_reasons.append("cudnn_benchmark_enabled")
    # `paper_contract` 是训练与推理共享的“不可变实验契约”，
    # 之后的推理会拿 checkpoint 里的这份契约和当前命令/manifest 做逐项比对。
    run_contract = build_run_contract(
        split_manifest=split_manifest_path,
        manifest_sha256=manifest_hash,
        dataset_kind=dataset_kind,
        task_mode=task_mode,
        speaker_id=task_speaker_id,
        text_policy=str(args.text_policy),
        label_names=list(label_names),
        validity_summary=validity_summary,
        ablation=str(args.ablation),
        zero_video=bool(args.zero_video),
        zero_audio=bool(args.zero_audio),
        zero_text=bool(args.zero_text),
        use_intensity=bool(args.use_intensity),
        video_backbone=str(args.video_backbone),
        flow_encoder_variant=str(FLOW_VIDEO_ENCODER_VARIANT),
        text_model=str(args.text_model),
        max_text_len=int(args.max_text_len),
        sample_rate=int(args.sample_rate),
        max_audio_sec=float(args.max_audio_sec),
        num_frames=int(args.num_frames),
    )
    run_provenance = build_run_provenance(
        runtime_profile=profile.to_jsonable(),
        deterministic_policy=deterministic_policy,
        repo_root=Path(__file__).resolve().parent,
    )

    # 这里先按 subset 取样本，再根据任务模式映射成当前任务自己的标签索引。
    train_items = prepare_manifest_items_for_task(
        filter_manifest_items_for_task(select_manifest_items(manifest, "train"), task_mode, task_speaker_id),   # 先选出 train 子集的样本，再根据 task_mode 和 speaker_id 过滤成当前任务的样本，最后映射成当前任务的标签索引。
        task_mode=task_mode,
        speaker_id=task_speaker_id,
        map_label_to_task_index=map_label_to_task_index,
    )
    val_items = prepare_manifest_items_for_task(
        filter_manifest_items_for_task(select_manifest_items(manifest, "val"), task_mode, task_speaker_id), # 先选出 val 子集的样本，再根据 task_mode 和 speaker_id 过滤成当前任务的样本，最后映射成当前任务的标签索引。
        task_mode=task_mode,
        speaker_id=task_speaker_id,
        map_label_to_task_index=map_label_to_task_index,
    )
    if not train_items or not val_items:
        raise RuntimeError("Split manifest produced an empty train or val subset for the selected task.")

    print(f"Using split manifest: {split_manifest_path}", flush=True)
    print(f"Manifest sha256: {manifest_hash}", flush=True)
    print_manifest_label_hist("train", train_items, label_names)
    print_manifest_label_hist("val", val_items, label_names)

    speaker_baseline = None
    speaker_only = None
    if task_mode == "confounded_7way":
        speaker_baseline = speaker_majority_baseline(train_items, val_items, label_names)
        speaker_only = speaker_only_baseline(train_items, val_items, label_names)

    ingress_cfg = ManifestIngressConfig(
        sample_rate=int(args.sample_rate),
        max_audio_sec=float(args.max_audio_sec),
        audio_backend_mode=str(args.audio_backend),
        video_decode_backend=str(args.video_decode_backend),
        num_frames=int(args.num_frames),
        zero_audio=bool(args.zero_audio),
        zero_video=bool(args.zero_video),
        video_backbone=str(args.video_backbone),
    )
    need_audio = not bool(args.zero_audio)
    need_video = (not bool(args.zero_video)) and str(args.video_backbone) in {"flow", "videomae", "dual"}
    need_text = not bool(args.zero_text)
    input_cache_contract = None
    input_cache_dir = args.input_cache.expanduser() if args.input_cache is not None else None
    if input_cache_dir is not None:
        with _startup_stage(f"validate input cache contract: {input_cache_dir}"):
            cache_meta = load_input_cache_meta(input_cache_dir)
            input_cache_contract = build_input_cache_contract(cache_meta)
            cache_mismatch_reasons = validate_input_cache_contract(
                input_cache_contract,
                manifest_sha256=manifest_hash,
                dataset_kind=dataset_kind,
                sample_rate=int(args.sample_rate),
                max_audio_sec=float(args.max_audio_sec),
                num_frames=int(args.num_frames),
                rgb_size=int(args.rgb_size),
                text_model=str(args.text_model),
                max_text_len=int(args.max_text_len),
                need_audio=bool(need_audio),
                need_video=bool(need_video),
                need_text=bool(need_text),
                text_policy=str(args.text_policy),
            )
            if cache_mismatch_reasons:
                raise RuntimeError(
                    "Input cache contract mismatch: " + ", ".join(str(reason) for reason in cache_mismatch_reasons)
                )
        with _startup_stage("build cached train dataset"):
            train_ds = CachedManifestDataset(
                train_items,
                ingress=ingress_cfg,
                cache_dir=input_cache_dir,
                text_policy=str(args.text_policy),
                runtime_profile=profile,
                progress_logger=_startup_log,
            )
        with _startup_stage("build cached val dataset"):
            val_ds = CachedManifestDataset(
                val_items,
                ingress=ingress_cfg,
                cache_dir=input_cache_dir,
                text_policy=str(args.text_policy),
                runtime_profile=profile,
                progress_logger=_startup_log,
            )
    else:
        with _startup_stage("build streaming train dataset"):
            train_ds = StreamingManifestDataset(train_items, ingress=ingress_cfg)
        with _startup_stage("build streaming val dataset"):
            val_ds = StreamingManifestDataset(val_items, ingress=ingress_cfg)

    # 文本不被 zero 掉时，提前把整份 manifest 文本分词缓存好，
    # 避免每个 batch 都重复调用 tokenizer。
    tokenizer = None
    if not bool(args.zero_text) and input_cache_contract is None:
        with _startup_stage(f"load tokenizer + pretokenize manifest text: {args.text_model}"):
            tokenizer = _load_tokenizer(str(args.text_model))
            cache_manifest_text_tokens(train_ds.items, tokenizer, max_text_len=int(args.max_text_len), text_policy=str(args.text_policy))
            cache_manifest_text_tokens(val_ds.items, tokenizer, max_text_len=int(args.max_text_len), text_policy=str(args.text_policy))

    # batch size / worker 数都走自动解析，避免用户在不同机器上手工调很多次。
    with _startup_stage("resolve loader settings + build dataloaders"):
        resolved_batch_size = resolve_batch_size(
            args.batch_size,
            phase="train",
            profile=profile,
            feature_cache=False,
            video_backbone=str(args.video_backbone),
            freeze_audio=bool(args.freeze_audio),
            freeze_text=bool(args.freeze_text),
            freeze_flow=bool(args.freeze_video or args.freeze_flow),
            freeze_rgb=bool(args.freeze_video or args.freeze_rgb),
        )
        resolved_num_workers = resolve_worker_count(
            args.num_workers,
            phase="train",
            profile=profile,
            dataset_in_memory=bool(getattr(train_ds, "in_memory", False) and getattr(val_ds, "in_memory", False)),
            cache_backed=bool(input_cache_contract is not None),
            total_items=len(train_items) + len(val_items),
        )
        prefetch_factor = resolve_prefetch_factor("auto", num_workers=int(resolved_num_workers))
        dl_kwargs: dict[str, Any] = {
            "num_workers": int(resolved_num_workers),
            "collate_fn": collate_manifest_items,
            "pin_memory": bool(device.type == "cuda"),
        }
        if int(resolved_num_workers) > 0:
            dl_kwargs["persistent_workers"] = True
            if prefetch_factor is not None:
                dl_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

        # train/val 分别使用不同的 generator seed，避免两者的随机流完全相同。
        train_loader = DataLoader(
            train_ds,
            batch_size=int(resolved_batch_size),
            shuffle=True,
            worker_init_fn=make_dataloader_worker_init_fn(int(args.seed)),
            generator=make_torch_generator(int(args.seed)),
            **dl_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(resolved_batch_size),
            shuffle=False,
            worker_init_fn=make_dataloader_worker_init_fn(int(args.seed) + 1000),
            generator=make_torch_generator(int(args.seed) + 1000),
            **dl_kwargs,
        )

    # 模型初始化完全由命令行与 paper contract 决定，
    # 这里不会根据数据内容偷偷改动结构。
    with _startup_stage("construct fusion classifier"):
        model = FusionClassifier(
            num_classes=len(label_names),
            freeze_audio=bool(args.freeze_audio),
            freeze_video=bool(args.freeze_video),
            freeze_flow=bool(args.freeze_video or args.freeze_flow),
            freeze_rgb=bool(args.freeze_video or args.freeze_rgb),
            freeze_prosody=bool(args.freeze_prosody),
            text_model=str(args.text_model),
            freeze_text=bool(args.freeze_text),
            audio_model=str(args.audio_model),
            audio_model_revision=(str(args.audio_model_revision).strip() or None),
            video_backbone=str(args.video_backbone),
            video_model=str(args.video_model),
            fusion_mode=str(args.fusion_mode),
            modality_dropout=float(args.modality_dropout),
            gate_temperature=float(args.gate_temperature),
            gate_scale=float(args.gate_scale),
            delta_scale=float(args.delta_scale),
            intensity_head=bool(args.use_intensity),
            progress_callback=_startup_log,
        )
    with _startup_stage(f"move model to device: {device}"):
        model = model.to(device)
    setattr(model, "_label_names", list(label_names))

    with _startup_stage("initialize optimizer + scheduler + preprocessors"):
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        scheduler = None
        if str(args.lr_scheduler) == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=float(args.lr_plateau_factor),
                patience=int(args.lr_plateau_patience),
                min_lr=float(args.lr_min),
            )
        loss_fn = torch.nn.CrossEntropyLoss()

        stream_preprocessor = GpuStreamPreprocessor(
            GpuStreamConfig(
                device=device,
                video_backbone=str(args.video_backbone),
                sample_rate=int(args.sample_rate),
                max_audio_sec=float(args.max_audio_sec),
                audio_backend_mode=str(args.audio_backend),
                num_frames=int(args.num_frames),
                flow_size=int(args.flow_size),
                rgb_size=int(args.rgb_size),
                zero_video=bool(args.zero_video),
                zero_audio=bool(args.zero_audio),
                zero_text=bool(args.zero_text),
                prosody_use_pitch=(not bool(args.prosody_no_pitch)),
                video_decode_backend=str(args.video_decode_backend),
                flow_backend=str(args.flow_backend),
            )
        )
        aug_cfg = AudioAugConfig(sample_rate=int(args.sample_rate))
        prosody_cfg = ProsodyConfig(sample_rate=int(args.sample_rate), use_pitch=(not bool(args.prosody_no_pitch)))
        scaler = None
        if device.type == "cuda":
            amp_mod = getattr(torch, "amp", None)
            if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
                scaler = amp_mod.GradScaler("cuda", enabled=(resolved_amp_mode == "fp16"))  # type: ignore[attr-defined]
            else:
                scaler = torch.cuda.amp.GradScaler(enabled=(resolved_amp_mode == "fp16"))

    print(
        json.dumps(
            {
                "resolved_batch_size": int(resolved_batch_size),
                "resolved_num_workers": int(resolved_num_workers),
                "resolved_prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
                "pin_memory": bool(device.type == "cuda"),
                "pipeline": "gpu_stream",
                "amp_mode": resolved_amp_mode,
                "lr_scheduler": str(args.lr_scheduler),
                "input_cache": str(input_cache_dir) if input_cache_dir is not None else None,
                "input_cache_in_memory": bool(getattr(train_ds, "in_memory", False) and getattr(val_ds, "in_memory", False)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    out_dir = args.output_dir.expanduser()
    store = RunAttemptStore.create(
        out_dir,
        seed=int(args.seed),
        benchmark_tag=str(args.benchmark_tag),
        args_payload=to_jsonable(vars(args)),
        run_contract=run_contract,
        provenance=run_provenance,
        validity=validity_summary,
        input_cache_contract=input_cache_contract,
        deterministic_policy=deterministic_policy,
    )
    signal_capture = SignalCapture().install()
    metrics_dir = store.published_dir
    store.mark_running()

    # `history` 既服务于后续画图，也会成为 published metrics.json 的原始时间序列部分。
    history: dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
        "train_per_class_recall": [],
        "val_per_class_recall": [],
    }
    if bool(args.use_intensity):
        history.update(
            {
                "train_mse": [],
                "train_mae": [],
                "train_ccc": [],
                "train_intensity_n": [],
                "val_mse": [],
                "val_mae": [],
                "val_ccc": [],
                "val_intensity_n": [],
            }
        )

    early_stop = EarlyStopController(
        EarlyStopConfig(
            monitor_name=str(args.monitor),
            monitor_mode="max",
            patience=int(args.early_stop_patience),
            min_epochs=int(args.early_stop_min_epochs),
            min_delta=float(args.early_stop_min_delta),
            tie_break_name="val_loss",
            tie_break_mode="min",
            after_lr_drops=int(args.early_stop_after_lr_drops),
        )
    )
    best_epoch = 0
    best_acc = 0.0
    best_f1 = 0.0
    best_val_loss = 0.0
    best_monitor_value = 0.0
    significant_best_monitor_value = float("-inf")
    best_checkpoint_reason = ""
    best_val_summary: dict[str, Any] = {}
    best_train_summary: dict[str, Any] = {}
    best_val_records: list[dict[str, Any]] = []
    final_epoch = 0
    stop_reason = ""
    run_status = "completed"

    try:
        for epoch in range(1, int(args.epochs) + 1):
            final_epoch = int(epoch)
            epoch_lr = _current_lr(opt)
            try:
                train_stats = _run_phase(
                    phase=f"Epoch {epoch}/{args.epochs} [train]",
                    model=model,
                    loader=train_loader,
                    preprocessor=stream_preprocessor,
                    tokenizer=tokenizer,
                    args=args,
                    device=device,
                    amp_mode=resolved_amp_mode,
                    loss_fn=loss_fn,
                    optimizer=opt,
                    scaler=scaler,
                    prosody_cfg=prosody_cfg,
                    aug_cfg=aug_cfg,
                )
                val_stats = _run_phase(
                    phase=f"Epoch {epoch}/{args.epochs} [val]",
                    model=model,
                    loader=val_loader,
                    preprocessor=stream_preprocessor,
                    tokenizer=tokenizer,
                    args=args,
                    device=device,
                    amp_mode=resolved_amp_mode,
                    loss_fn=loss_fn,
                    optimizer=None,
                    scaler=None,
                    prosody_cfg=prosody_cfg,
                    aug_cfg=aug_cfg,
                )
            except NumericStabilityError as exc:
                run_status = "failed_numeric"
                stop_reason = f"failed_numeric: {exc}"
                paper_grade_reasons.append("numeric_failure")
                print(f"Stopping training due to numeric instability: {exc}", flush=True)
                break
            except DeterminismCompatibilityError as exc:
                run_status = "failed_determinism"
                stop_reason = f"failed_determinism: {exc}"
                paper_grade_reasons.append("determinism_unsupported_op")
                print(f"Stopping training due to deterministic-op incompatibility: {exc}", flush=True)
                break

            print(
                "Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} "
                "val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
                "lr={lr:.2e} train_prepare_s={train_prepare:.2f} val_prepare_s={val_prepare:.2f}".format(
                    epoch=epoch,
                    train_loss=train_stats["loss"],
                    train_acc=train_stats["accuracy"],
                    train_f1=train_stats["macro_f1"],
                    val_loss=val_stats["loss"],
                    val_acc=val_stats["accuracy"],
                    val_f1=val_stats["macro_f1"],
                    lr=epoch_lr,
                    train_prepare=train_stats["prepare_sec"],
                    val_prepare=val_stats["prepare_sec"],
                ),
                flush=True,
            )
            if bool(args.use_intensity):
                print(
                    "Intensity: train_mse={train_mse:.4f} train_mae={train_mae:.4f} train_ccc={train_ccc:.4f} (n={train_n}) "
                    "val_mse={val_mse:.4f} val_mae={val_mae:.4f} val_ccc={val_ccc:.4f} (n={val_n})".format(
                        train_mse=train_stats["mse"],
                        train_mae=train_stats["mae"],
                        train_ccc=train_stats["ccc"],
                        train_n=train_stats["intensity_n"],
                        val_mse=val_stats["mse"],
                        val_mae=val_stats["mae"],
                        val_ccc=val_stats["ccc"],
                        val_n=val_stats["intensity_n"],
                    ),
                    flush=True,
                )

            history["train_loss"].append(float(train_stats["loss"]))
            history["train_acc"].append(float(train_stats["accuracy"]))
            history["train_f1"].append(float(train_stats["macro_f1"]))
            history["val_loss"].append(float(val_stats["loss"]))
            history["val_acc"].append(float(val_stats["accuracy"]))
            history["val_f1"].append(float(val_stats["macro_f1"]))
            history["lr"].append(float(epoch_lr))
            history["train_per_class_recall"].append(dict(train_stats["class_summary"].get("per_class_recall", {})))
            history["val_per_class_recall"].append(dict(val_stats["class_summary"].get("per_class_recall", {})))
            if bool(args.use_intensity):
                history["train_mse"].append(float(train_stats["mse"]))
                history["train_mae"].append(float(train_stats["mae"]))
                history["train_ccc"].append(float(train_stats["ccc"]))
                history["train_intensity_n"].append(float(train_stats["intensity_n"]))
                history["val_mse"].append(float(val_stats["mse"]))
                history["val_mae"].append(float(val_stats["mae"]))
                history["val_ccc"].append(float(val_stats["ccc"]))
                history["val_intensity_n"].append(float(val_stats["intensity_n"]))

            monitor_value = float(val_stats["macro_f1"] if str(args.monitor) == "val_f1" else val_stats["accuracy"])
            decision = early_stop.observe(
                epoch=epoch,
                monitor_value=monitor_value,
                tie_break_value=float(val_stats["loss"]),
            )
            if decision.should_save_checkpoint:
                best_epoch = int(epoch)
                best_acc = float(val_stats["accuracy"])
                best_f1 = float(val_stats["macro_f1"])
                best_monitor_value = float(monitor_value)
                best_val_loss = float(val_stats["loss"])
                best_checkpoint_reason = str(decision.checkpoint_reason or "")
                significant_best_monitor_value = float(
                    early_stop.best_monitor_value if early_stop.best_monitor_value is not None else monitor_value
                )
                best_train_summary = dict(train_stats["class_summary"])
                best_val_summary = dict(val_stats["class_summary"])
                best_val_records = list(val_stats["records"])
                bundle_checkpoint_path = store.attempt_dir / "bundles" / f"best_epoch_{int(epoch):04d}" / "checkpoint.pt"
                ckpt = {
                    "model": model.state_dict(),
                    "epoch": int(epoch),
                    "best_acc": float(best_acc),
                    "best_f1": float(best_f1),
                    "best_monitor_name": str(args.monitor),
                    "best_monitor_value": float(best_monitor_value),
                    "significant_best_monitor_value": float(significant_best_monitor_value),
                    "best_val_loss": float(best_val_loss),
                    "checkpoint_reason": str(best_checkpoint_reason),
                    "args": to_jsonable(vars(args)),
                    "label_names": list(label_names),
                    "task_mode": task_mode,
                    "speaker_id": task_speaker_id,
                    "text_policy": str(args.text_policy),
                    "validity": validity_summary,
                    "resolved_runtime": {
                        "device": str(device),
                        "profile": profile.to_jsonable(),
                        "batch_size": int(resolved_batch_size),
                        "num_workers": int(resolved_num_workers),
                        "prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
                        "amp_mode": resolved_amp_mode,
                        "pipeline": "gpu_stream",
                        "input_cache": str(input_cache_dir) if input_cache_dir is not None else None,
                        "input_cache_in_memory": bool(getattr(train_ds, "in_memory", False) and getattr(val_ds, "in_memory", False)),
                    },
                    "benchmark_tag": str(args.benchmark_tag),
                    "dataset_kind": dataset_kind,
                    "split_manifest": str(split_manifest_path),
                    "manifest_sha256": manifest_hash,
                    "ablation": str(args.ablation),
                    "zero_video": bool(args.zero_video),
                    "zero_audio": bool(args.zero_audio),
                    "zero_text": bool(args.zero_text),
                    "best_val_summary": best_val_summary,
                    "paper_contract": run_contract,
                    "input_cache_contract": input_cache_contract,
                    "provenance": run_provenance,
                    "paper_grade": build_paper_grade(validity_summary=validity_summary, ineligibility_reasons=paper_grade_reasons),
                    "run_status": str(run_status),
                }
                val_metrics_summary = {
                    "split_manifest": str(split_manifest_path),
                    "subset": "val",
                    "benchmark_tag": str(args.benchmark_tag),
                    "checkpoint": str(bundle_checkpoint_path),
                    "task_mode": task_mode,
                    "speaker_id": task_speaker_id,
                    "text_policy": str(args.text_policy),
                    "label_names": list(label_names),
                    "attempted": len(best_val_records),
                    "ok": len(best_val_records),
                    "errors": 0,
                    "skipped": 0,
                    "accuracy_on_ok": float(best_acc),
                    "macro_f1_on_ok": float(best_f1),
                    "loss_on_ok": float(best_val_loss),
                    "best_epoch": int(best_epoch),
                    "checkpoint_reason": str(best_checkpoint_reason),
                    "per_class_recall": best_val_summary.get("per_class_recall", {}),
                    "confusion_matrix": best_val_summary.get("confusion_matrix", []),
                    "support": best_val_summary.get("support", {}),
                    "pred_counts": best_val_summary.get("pred_counts", {}),
                    "manifest_sha256": manifest_hash,
                    "manifest_summary": manifest_summary,
                    "validity": validity_summary,
                    "speaker_majority_baseline": speaker_baseline,
                    "speaker_only_baseline": speaker_only,
                    "pipeline": "gpu_stream",
                    "paper_contract": run_contract,
                    "input_cache_contract": input_cache_contract,
                    "provenance": run_provenance,
                    "paper_grade": build_paper_grade(validity_summary=validity_summary, ineligibility_reasons=paper_grade_reasons),
                    "run_status": str(run_status),
                }
                bundle_dir = store.publish_best_bundle(
                    torch_mod=torch,
                    epoch=epoch,
                    checkpoint_payload=ckpt,
                    records=best_val_records,
                    metrics_summary=val_metrics_summary,
                    selection_meta={
                        "checkpoint_reason": str(best_checkpoint_reason),
                        "best_monitor_value": float(best_monitor_value),
                        "best_val_loss": float(best_val_loss),
                    },
                )
                print(f"Saved best checkpoint bundle -> {bundle_dir / 'checkpoint.pt'}", flush=True)

            epoch_state = {
                "schema_version": "run_store_v1",
                "epoch": int(epoch),
                "lr": float(epoch_lr),
                "train": {
                    "loss": float(train_stats["loss"]),
                    "accuracy": float(train_stats["accuracy"]),
                    "macro_f1": float(train_stats["macro_f1"]),
                    "prepare_sec": float(train_stats["prepare_sec"]),
                    "class_summary": dict(train_stats["class_summary"]),
                },
                "val": {
                    "loss": float(val_stats["loss"]),
                    "accuracy": float(val_stats["accuracy"]),
                    "macro_f1": float(val_stats["macro_f1"]),
                    "prepare_sec": float(val_stats["prepare_sec"]),
                    "class_summary": dict(val_stats["class_summary"]),
                },
                "selection": {
                    "monitor_name": str(args.monitor),
                    "monitor_value": float(monitor_value),
                    "checkpoint_selected": bool(decision.should_save_checkpoint),
                    "checkpoint_reason": str(decision.checkpoint_reason or ""),
                    "best_epoch": int(best_epoch),
                    "best_bundle_relpath": store.best_bundle_relpath,
                },
            }
            if bool(args.use_intensity):
                epoch_state["train"].update(
                    {
                        "mse": float(train_stats["mse"]),
                        "mae": float(train_stats["mae"]),
                        "ccc": float(train_stats["ccc"]),
                        "intensity_n": float(train_stats["intensity_n"]),
                    }
                )
                epoch_state["val"].update(
                    {
                        "mse": float(val_stats["mse"]),
                        "mae": float(val_stats["mae"]),
                        "ccc": float(val_stats["ccc"]),
                        "intensity_n": float(val_stats["intensity_n"]),
                    }
                )

            if scheduler is not None:
                lr_before_step = _current_lr(opt)
                scheduler.step(float(val_stats["loss"]))
                lr_after_step = _current_lr(opt)
                if lr_after_step + 1.0e-12 < lr_before_step:
                    early_stop.register_lr_drop(epoch)
                    print(
                        f"LR reduced at epoch {epoch}: {lr_before_step:.2e} -> {lr_after_step:.2e}",
                        flush=True,
                    )
                    epoch_state["scheduler"] = {
                        "lr_before_step": float(lr_before_step),
                        "lr_after_step": float(lr_after_step),
                        "lr_drop_registered": True,
                    }

            store.write_epoch_state(
                epoch=epoch,
                epoch_payload=epoch_state,
                best_epoch=best_epoch,
                best_bundle_relpath=store.best_bundle_relpath,
            )

            should_stop, stop_reason_candidate = early_stop.evaluate_stop(epoch)
            if should_stop:
                stop_reason = str(stop_reason_candidate or "")
                print(f"Early stopping at epoch {epoch}: {stop_reason}", flush=True)
                break
            if signal_capture.state.requested:
                signal_name = str(signal_capture.state.signal_name or "SIGTERM")
                store.note_signal(signal_name)
                run_status = "interrupted"
                stop_reason = f"interrupted:{signal_name}"
                print(f"Stopping training after epoch {epoch} due to {signal_name}", flush=True)
                break

        if not stop_reason:
            stop_reason = "max_epochs_reached"

        paper_grade = build_paper_grade(validity_summary=validity_summary, ineligibility_reasons=paper_grade_reasons)

        if run_status not in {"failed_numeric", "failed_determinism"}:
            last_ckpt = {
                "model": model.state_dict(),
                "epoch": int(final_epoch),
                "best_acc": float(best_acc),
                "best_f1": float(best_f1),
                "best_monitor_name": str(args.monitor),
                "best_monitor_value": float(best_monitor_value),
                "significant_best_monitor_value": float(
                    significant_best_monitor_value if significant_best_monitor_value != float("-inf") else 0.0
                ),
                "best_val_loss": float(best_val_loss),
                "checkpoint_reason": str(best_checkpoint_reason),
                "args": to_jsonable(vars(args)),
                "label_names": list(label_names),
                "task_mode": task_mode,
                "speaker_id": task_speaker_id,
                "text_policy": str(args.text_policy),
                "validity": validity_summary,
                "resolved_runtime": {
                    "device": str(device),
                    "profile": profile.to_jsonable(),
                    "batch_size": int(resolved_batch_size),
                    "num_workers": int(resolved_num_workers),
                    "prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
                    "amp_mode": resolved_amp_mode,
                    "pipeline": "gpu_stream",
                    "input_cache": str(input_cache_dir) if input_cache_dir is not None else None,
                    "input_cache_in_memory": bool(getattr(train_ds, "in_memory", False) and getattr(val_ds, "in_memory", False)),
                },
                "stop": {
                    "epoch": int(final_epoch),
                    "reason": str(stop_reason),
                    "epochs_without_improvement": int(early_stop.epochs_without_improvement),
                    "lr_drop_epochs": list(early_stop.lr_drop_epochs),
                },
                "paper_contract": run_contract,
                "input_cache_contract": input_cache_contract,
                "provenance": run_provenance,
                "paper_grade": paper_grade,
                "run_status": str(run_status),
            }
            store.publish_last_checkpoint(torch_mod=torch, checkpoint_payload=last_ckpt)

        metrics_payload: dict[str, Any] = dict(history)
        metrics_payload["meta"] = {
            "benchmark_tag": str(args.benchmark_tag),
            "split_manifest": str(split_manifest_path),
            "manifest_sha256": manifest_hash,
            "dataset_kind": dataset_kind,
            "task_mode": task_mode,
            "speaker_id": task_speaker_id,
            "text_policy": str(args.text_policy),
            "label_names": list(label_names),
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "ablation": str(args.ablation),
            "zero_video": bool(args.zero_video),
            "zero_audio": bool(args.zero_audio),
            "zero_text": bool(args.zero_text),
            "args": to_jsonable(vars(args)),
            "device": str(device),
            "runtime_profile": profile.to_jsonable(),
            "resolved_batch_size": int(resolved_batch_size),
            "resolved_num_workers": int(resolved_num_workers),
            "resolved_prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
            "amp_mode": resolved_amp_mode,
            "lr_scheduler": str(args.lr_scheduler),
            "pipeline": "gpu_stream",
            "input_cache": str(input_cache_dir) if input_cache_dir is not None else None,
            "input_cache_contract": input_cache_contract,
            "input_cache_in_memory": bool(getattr(train_ds, "in_memory", False) and getattr(val_ds, "in_memory", False)),
            "manifest_summary": manifest_summary,
            "paper_protocol_version": str(run_contract.get("protocol_version")),
            "deterministic_policy": deterministic_policy,
            "paper_contract": run_contract,
            "run_dir": str(out_dir),
            "attempt_dir": str(store.attempt_dir),
            "best_bundle_relpath": store.best_bundle_relpath,
        }
        metrics_payload["best"] = {
            "epoch": int(best_epoch),
            "best_acc": float(best_acc),
            "best_f1": float(best_f1),
            "best_val_loss": float(best_val_loss),
            "best_monitor_name": str(args.monitor),
            "best_monitor_value": float(best_monitor_value),
            "significant_best_monitor_value": float(
                significant_best_monitor_value if significant_best_monitor_value != float("-inf") else 0.0
            ),
            "checkpoint_reason": str(best_checkpoint_reason),
            "best_train_summary": best_train_summary,
            "best_val_summary": best_val_summary,
        }
        metrics_payload["stop"] = {
            "epoch": int(final_epoch),
            "reason": str(stop_reason),
            "epochs_without_improvement": int(early_stop.epochs_without_improvement),
            "lr_drop_epochs": list(early_stop.lr_drop_epochs),
        }
        metrics_payload["validity"] = validity_summary
        metrics_payload["provenance"] = run_provenance
        metrics_payload["paper_grade"] = paper_grade
        metrics_payload["run_status"] = str(run_status)
        if speaker_baseline is not None:
            metrics_payload["speaker_majority_baseline"] = speaker_baseline
        if speaker_only is not None:
            metrics_payload["speaker_only_baseline"] = speaker_only

        save_metrics_and_plots(metrics_dir, metrics_payload)
        write_results_summary(metrics_dir, metrics_payload)
        store.publish_reports(metrics_dir=metrics_dir)

        terminal_status = {
            "completed": "completed",
            "failed_numeric": "failed_numeric",
            "failed_determinism": "failed_determinism",
            "interrupted": "interrupted",
        }.get(str(run_status), "completed")
        publish_attempt = terminal_status == "completed"
        if terminal_status in {"failed_numeric", "failed_determinism"}:
            store.record_failure(status=terminal_status, message=stop_reason, exc_type=terminal_status)
        store.finalize(
            status=terminal_status,
            run_status=str(run_status),
            stop_reason=str(stop_reason),
            publish_attempt=publish_attempt,
            failure=None,
        )
        print(
            f"Best val macro_f1={best_f1:.4f}, acc={best_acc:.4f}, epoch={best_epoch}, stop_reason={stop_reason}",
            flush=True,
        )
    except KeyboardInterrupt:
        run_status = "interrupted"
        stop_reason = str(stop_reason or "interrupted:KeyboardInterrupt")
        failure = {
            "type": "KeyboardInterrupt",
            "message": str(stop_reason),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        store.finalize(
            status="interrupted",
            run_status=run_status,
            stop_reason=stop_reason,
            publish_attempt=False,
            failure=failure,
        )
        raise
    except Exception as exc:
        run_status = "failed_exception"
        stop_reason = str(stop_reason or f"failed_exception: {type(exc).__name__}: {exc}")
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        store.finalize(
            status="failed_exception",
            run_status=run_status,
            stop_reason=stop_reason,
            publish_attempt=False,
            failure=failure,
        )
        raise
    finally:
        signal_capture.restore()


if __name__ == "__main__":
    main()
