"""Manifest 驱动的主线批量推理入口。

这个文件负责读取训练得到的 `best.pt`，校验 checkpoint 与当前 manifest/
命令行配置是否兼容，然后批量生成逐样本预测 JSONL 与汇总 metrics。
如果训练负责回答“模型怎么学出来”，这里负责回答“这个模型如何被规范地评估”。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

torch: Any = None
DataLoader: Any = None
tqdm: Any = None
ManifestIngressConfig: Any = None
StreamingManifestDataset: Any = None
CachedManifestDataset: Any = None
cache_manifest_text_tokens: Any = None
collate_manifest_items: Any = None
GpuStreamConfig: Any = None
GpuStreamPreprocessor: Any = None
autocast_context: Any = None
build_paper_grade: Any = None
build_run_contract: Any = None
build_run_provenance: Any = None
ccc: Any = None
FLOW_VIDEO_ENCODER_VARIANT: Any = None
LEGACY_FLOW_VIDEO_ENCODER_VARIANT: Any = None
make_dataloader_worker_init_fn: Any = None
make_torch_generator: Any = None
prepare_manifest_items_for_task: Any = None
resolve_ablation_flags: Any = None
set_seed: Any = None
EMOTIONS: Any = None
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
    """确保仓库根目录在 `sys.path` 中。"""

    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()

from hf_compat import ensure_transformers_torch_compat
from hf_loading import resolve_hf_pretrained_source
from run_store import (
    register_published_inference_output,
    resolve_attempt_dir,
    resolve_best_bundle,
)


def _lazy_runtime_imports() -> None:
    """按需导入推理真正需要的运行时模块。

    这样可以让 `--help` 等轻量命令避免提前加载整套训练/推理依赖。
    """

    global torch, DataLoader, tqdm
    global ManifestIngressConfig, StreamingManifestDataset, CachedManifestDataset, cache_manifest_text_tokens, collate_manifest_items
    global GpuStreamConfig, GpuStreamPreprocessor
    global autocast_context, build_paper_grade, build_run_contract, build_run_provenance, ccc
    global FLOW_VIDEO_ENCODER_VARIANT, LEGACY_FLOW_VIDEO_ENCODER_VARIANT
    global make_dataloader_worker_init_fn, make_torch_generator, prepare_manifest_items_for_task, resolve_ablation_flags, set_seed
    global EMOTIONS, build_validity_summary, filter_manifest_items_for_task, load_split_manifest
    global manifest_sha256, map_label_to_task_index, resolve_task_label_names, resolve_task_mode, select_manifest_items
    global classification_summary, speaker_majority_baseline, speaker_only_baseline
    global FusionClassifier
    global detect_runtime, resolve_amp_mode, resolve_batch_size, resolve_prefetch_factor, resolve_worker_count, select_device
    global resolve_text_policy
    global build_input_cache_contract, load_input_cache_meta, validate_input_cache_contract
    if torch is not None:
        return

    import torch as _torch
    from torch.utils.data import DataLoader as _DataLoader
    from tqdm import tqdm as _tqdm

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
        LEGACY_FLOW_VIDEO_ENCODER_VARIANT as _LEGACY_FLOW_VIDEO_ENCODER_VARIANT,
        make_dataloader_worker_init_fn as _make_dataloader_worker_init_fn,
        make_torch_generator as _make_torch_generator,
        prepare_manifest_items_for_task as _prepare_manifest_items_for_task,
        resolve_ablation_flags as _resolve_ablation_flags,
        set_seed as _set_seed,
    )
    from manifest_utils import (
        EMOTIONS as _EMOTIONS,
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
    ccc = _ccc
    FLOW_VIDEO_ENCODER_VARIANT = _FLOW_VIDEO_ENCODER_VARIANT
    LEGACY_FLOW_VIDEO_ENCODER_VARIANT = _LEGACY_FLOW_VIDEO_ENCODER_VARIANT
    make_dataloader_worker_init_fn = _make_dataloader_worker_init_fn
    make_torch_generator = _make_torch_generator
    prepare_manifest_items_for_task = _prepare_manifest_items_for_task
    resolve_ablation_flags = _resolve_ablation_flags
    set_seed = _set_seed
    EMOTIONS = _EMOTIONS
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
    """定义并解析批量推理 CLI 参数。

    这里最关键的参数是：
    - `--split-manifest`: 当前要评估的数据清单
    - `--checkpoint`: 必须指向训练阶段选出的 `best.pt`
    - `--subset`: 指定评估 train/val/test 中哪一部分
    """

    p = argparse.ArgumentParser(description="Batch inference for the manifest-driven gpu_stream mainline")
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--subset", type=str, default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--task-mode", type=str, default="")
    p.add_argument("--speaker-id", type=str, default="")
    p.add_argument("--benchmark-tag", type=str, default="default")
    p.add_argument("--text-policy", type=str, default="")
    p.add_argument("--run-dir", type=Path, default=None, help="Run directory created by train.py with run_manifest.json")
    p.add_argument("--attempt-id", type=str, default="", help="Optional attempt id under --run-dir")
    p.add_argument("--checkpoint", type=Path, default=None, help="Low-level checkpoint path override for direct/debug inference")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--input-cache", type=Path, default=None, help="Optional mainline input cache directory built by build_mainline_input_cache.py")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--batch-size", type=str, default="auto")
    p.add_argument("--num-workers", type=str, default="auto")
    p.add_argument(
        "--ablation",
        type=str,
        default="",
        choices=["full", "text-only", "audio-only", "video-only", "no-text", "no-audio", "no-video"],
    )
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--flow-size", type=int, default=112, help=argparse.SUPPRESS)
    p.add_argument("--rgb-size", type=int, default=224, help=argparse.SUPPRESS)
    p.add_argument("--sample-rate", type=int, default=None)
    p.add_argument("--max-audio-sec", type=float, default=None)
    p.add_argument("--audio-backend", type=str, default="auto", choices=["auto", "torchaudio", "soundfile"], help=argparse.SUPPRESS)
    p.add_argument("--video-decode-backend", type=str, default="auto", choices=["auto", "decord", "cpu"], help=argparse.SUPPRESS)
    p.add_argument("--flow-backend", type=str, default="torch_motion", choices=["torch_motion", "legacy"], help=argparse.SUPPRESS)
    p.add_argument("--prosody-no-pitch", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--amp-mode", type=str, default="auto", choices=["auto", "off", "fp16", "bf16"])
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--print-json", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--print-result", type=str, default="none", choices=["none", "ok", "error", "all"], help=argparse.SUPPRESS)
    p.add_argument("--print-per-sample", action="store_true")
    p.add_argument("--print-text", action="store_true")
    p.add_argument("--text-max-len", type=int, default=0)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--text-model", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--audio-model", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--audio-model-revision", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--video-model", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument(
        "--allow-incompatible-checkpoint",
        action="store_true",
        help="Allow evaluation with checkpoint/manifest/config mismatches. Outputs are marked paper-grade ineligible.",
    )

    # Hidden compatibility args.
    p.add_argument("--zero-video", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--zero-audio", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--zero-text", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--skip-video-encoder", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-amp", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cached-dataset", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--feature-cache", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--pipeline", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--cache-mode", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--ram-cache-size", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def _validate_compat_args(args: argparse.Namespace) -> None:
    """清理旧版兼容参数，并把少量历史别名映射到新行为。"""

    if args.cached_dataset is not None or args.feature_cache is not None:
        raise RuntimeError(
            "Mainline inference no longer supports --cached-dataset or --feature-cache. "
            "Use --split-manifest with the gpu_stream path."
        )
    pipeline = str(args.pipeline or "").strip().lower()
    if pipeline and pipeline not in {"auto", "gpu_stream"}:
        raise RuntimeError(f"Unsupported retired --pipeline value: {args.pipeline}")
    cache_mode = str(args.cache_mode or "").strip().lower()
    if cache_mode and cache_mode not in {"none"}:
        raise RuntimeError("--cache-mode is retired; gpu_stream-only inference accepts at most --cache-mode none.")
    if args.ram_cache_size not in {None, 0}:
        raise RuntimeError("--ram-cache-size is retired in gpu_stream-only inference.")
    if bool(args.no_amp):
        args.amp_mode = "off"
    if bool(args.skip_video_encoder):
        args.zero_video = True
    if bool(args.print_per_sample):
        args.print_result = "all"
        args.no_progress = True


def _normalize_optional_text(value: Any) -> str:
    """把可选文本配置规范化为去首尾空白的字符串。"""

    return str(value or "").strip()


def _normalize_optional_upper(value: Any) -> str | None:
    """把可选字符串标准化为大写形式，常用于 speaker_id。"""

    text = str(value or "").strip().upper()
    return text or None


def _raise_or_record_mismatch(
    *,
    field: str,
    expected: Any,
    actual: Any,
    reasons: list[str],
    allow_mismatch: bool,
) -> None:
    """比较一项训练契约字段与当前推理字段是否一致。

    默认策略是“不一致就失败”，因为论文级评测不能静默混用错配置。
    只有用户显式打开 `--allow-incompatible-checkpoint` 时，才记录警告并把
    当前评测结果标记为 paper-grade ineligible。
    """

    if expected == actual:
        return
    message = f"{field} mismatch: expected {expected!r}, got {actual!r}"
    if not allow_mismatch:
        raise RuntimeError(message)
    print(f"WARNING: {message}", flush=True)
    reasons.append(f"{field}_mismatch")


def _checkpoint_run_contract(
    ckpt: dict[str, Any],
    *,
    ckpt_args: dict[str, Any],
    label_names: list[str],
    validity: dict[str, Any] | None,
) -> dict[str, Any]:
    """从 checkpoint 恢复训练时的 run contract。

    新版 checkpoint 会直接存 `paper_contract`。如果读到的是旧 checkpoint，
    就用它保存的 args 和元数据尽量补建一份契约，方便后面统一走兼容校验。
    """

    contract = ckpt.get("paper_contract", {})
    if isinstance(contract, dict) and contract:
        normalized = dict(contract)
        normalized.setdefault("flow_encoder_variant", LEGACY_FLOW_VIDEO_ENCODER_VARIANT)
        return normalized
    split_manifest_raw = ckpt.get("split_manifest", "")
    return build_run_contract(
        split_manifest=Path(str(split_manifest_raw or ".")),
        manifest_sha256=str(ckpt.get("manifest_sha256", "") or ""),
        dataset_kind=str(ckpt.get("dataset_kind", "") or ""),
        task_mode=str(ckpt.get("task_mode", ckpt_args.get("task_mode", "confounded_7way")) or "confounded_7way"),
        speaker_id=_normalize_optional_upper(ckpt.get("speaker_id", ckpt_args.get("speaker_id", ""))),
        text_policy=str(ckpt.get("text_policy", ckpt_args.get("text_policy", "full")) or "full"),
        label_names=list(label_names),
        validity_summary=validity or {},
        ablation=str(ckpt.get("ablation", ckpt_args.get("ablation", "full")) or "full"),
        zero_video=bool(ckpt.get("zero_video", ckpt_args.get("zero_video", False))),
        zero_audio=bool(ckpt.get("zero_audio", ckpt_args.get("zero_audio", False))),
        zero_text=bool(ckpt.get("zero_text", ckpt_args.get("zero_text", False))),
        use_intensity=bool(ckpt_args.get("use_intensity", False)),
        video_backbone=str(ckpt_args.get("video_backbone", "dual") or "dual"),
        flow_encoder_variant=LEGACY_FLOW_VIDEO_ENCODER_VARIANT,
        sample_rate=int(ckpt_args.get("sample_rate", 24000) or 24000),
        max_audio_sec=float(ckpt_args.get("max_audio_sec", 6.0) or 6.0),
        num_frames=int(ckpt_args.get("num_frames", 64) or 64),
    )


def _checkpoint_provenance(ckpt: dict[str, Any], *, deterministic_policy: dict[str, Any], profile: Any) -> dict[str, Any]:
    """从 checkpoint 恢复 provenance；缺失时生成最小兜底记录。"""

    provenance = ckpt.get("provenance", {})
    if isinstance(provenance, dict) and provenance:
        return dict(provenance)
    runtime_profile = profile.to_jsonable() if hasattr(profile, "to_jsonable") else dict(profile or {})
    return build_run_provenance(
        runtime_profile=runtime_profile,
        deterministic_policy=deterministic_policy,
        repo_root=Path(__file__).resolve().parent,
    )


def _assert_finite_tensor(name: str, value: Any) -> None:
    """在推理时拦截 NaN/Inf，避免把异常输出写进正式结果。"""

    if value is None:
        return
    if isinstance(value, torch.Tensor):
        if bool(torch.isfinite(value).all().item()):
            return
        raise RuntimeError(f"non-finite tensor detected in {name}")
    scalar = float(value)
    if scalar == scalar and scalar not in {float("inf"), float("-inf")}:
        return
    raise RuntimeError(f"non-finite scalar detected in {name}: {scalar}")


def _load_tokenizer(model_name: str) -> Any:
    """加载推理阶段使用的 tokenizer。"""

    try:
        ensure_transformers_torch_compat()
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required for text-enabled inference.") from e
    source, load_kwargs = resolve_hf_pretrained_source(str(model_name))
    return AutoTokenizer.from_pretrained(source, **load_kwargs)


def _load_model(
    ckpt_path: Path,
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[
    FusionClassifier,
    bool,
    list[str],
    str,
    str | None,
    str,
    int,
    dict[str, Any] | None,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    list[str],
]:
    """加载 checkpoint，并恢复推理所需的模型与实验契约信息。

    这个函数除了“把权重 load 进模型”，还承担两个论文级职责：
    1. 恢复训练时到底用了什么模型结构、任务设置、文本策略
    2. 在用户尝试覆盖这些关键配置时，决定是直接报错还是降级为不合格评测
    """

    _lazy_runtime_imports()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    label_names = [str(x) for x in ckpt.get("label_names", [])] if isinstance(ckpt, dict) and isinstance(ckpt.get("label_names"), list) else list(EMOTIONS)
    validity = ckpt.get("validity", None) if isinstance(ckpt, dict) and isinstance(ckpt.get("validity"), dict) else None
    compatibility_reasons: list[str] = []

    # 这些值默认优先使用 checkpoint 内保存的训练配置，
    # 因为论文级推理必须“跟训练时选出来的模型保持同一个定义”。
    text_model = str(ckpt_args.get("text_model", "") or "xlm-roberta-large")
    audio_model = str(ckpt_args.get("audio_model", "") or "microsoft/wavlm-large")
    audio_model_revision = str(ckpt_args.get("audio_model_revision", "") or "")
    video_model = str(ckpt_args.get("video_model", "") or "MCG-NJU/videomae-large")
    video_backbone = str(ckpt_args.get("video_backbone", "") or "dual")
    fusion_mode = str(ckpt_args.get("fusion_mode", "") or "gated_text")
    gate_temperature = float(ckpt_args.get("gate_temperature", 1.0) or 1.0)
    gate_scale = float(ckpt_args.get("gate_scale", 1.0) or 1.0)
    delta_scale = float(ckpt_args.get("delta_scale", 1.0) or 1.0)
    modality_dropout = float(ckpt_args.get("modality_dropout", 0.0) or 0.0)
    use_intensity = bool(ckpt_args.get("use_intensity", False))
    task_mode = str(ckpt_args.get("task_mode", "") or "confounded_7way")
    task_speaker_id = _normalize_optional_upper(ckpt_args.get("speaker_id", ""))
    text_policy = str(ckpt_args.get("text_policy", "") or "full")
    checkpoint_max_text_len = int(ckpt_args.get("max_text_len", 0) or 0)

    _raise_or_record_mismatch(
        field="text_model",
        expected=text_model,
        actual=_normalize_optional_text(args.text_model) or text_model,
        reasons=compatibility_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="audio_model",
        expected=audio_model,
        actual=_normalize_optional_text(args.audio_model) or audio_model,
        reasons=compatibility_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="audio_model_revision",
        expected=audio_model_revision,
        actual=_normalize_optional_text(args.audio_model_revision) or audio_model_revision,
        reasons=compatibility_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="video_model",
        expected=video_model,
        actual=_normalize_optional_text(args.video_model) or video_model,
        reasons=compatibility_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="task_mode",
        expected=task_mode,
        actual=_normalize_optional_text(args.task_mode) or task_mode,
        reasons=compatibility_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="speaker_id",
        expected=task_speaker_id,
        actual=_normalize_optional_upper(args.speaker_id) or task_speaker_id,
        reasons=compatibility_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="text_policy",
        expected=text_policy,
        actual=_normalize_optional_text(args.text_policy) or text_policy,
        reasons=compatibility_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )

    # 只有在用户显式传入覆盖参数时，才尝试替换本地变量；
    # 替换前已经由 `_raise_or_record_mismatch()` 做过硬校验或降级记录。
    if _normalize_optional_text(args.text_model):
        text_model = _normalize_optional_text(args.text_model)
    if _normalize_optional_text(args.audio_model):
        audio_model = _normalize_optional_text(args.audio_model)
    if _normalize_optional_text(args.audio_model_revision):
        audio_model_revision = _normalize_optional_text(args.audio_model_revision)
    if _normalize_optional_text(args.video_model):
        video_model = _normalize_optional_text(args.video_model)
    if _normalize_optional_text(args.task_mode):
        task_mode = _normalize_optional_text(args.task_mode)
    if _normalize_optional_upper(args.speaker_id) is not None:
        task_speaker_id = _normalize_optional_upper(args.speaker_id)
    if _normalize_optional_text(args.text_policy):
        text_policy = _normalize_optional_text(args.text_policy)

    task_mode = resolve_task_mode(task_mode)
    text_policy = resolve_text_policy(text_policy)
    if label_names == list(EMOTIONS):
        label_names = resolve_task_label_names(task_mode, task_speaker_id)
    if validity is None or str(validity.get("task_mode", "")).strip().lower() != task_mode:
        validity = build_validity_summary({}, task_mode, task_speaker_id)
    checkpoint_contract = _checkpoint_run_contract(
        ckpt if isinstance(ckpt, dict) else {},
        ckpt_args=ckpt_args if isinstance(ckpt_args, dict) else {},
        label_names=label_names,
        validity=validity,
    )
    checkpoint_paper_grade = ckpt.get("paper_grade", {}) if isinstance(ckpt, dict) and isinstance(ckpt.get("paper_grade"), dict) else {}
    if not isinstance(checkpoint_paper_grade, dict) or not checkpoint_paper_grade:
        checkpoint_paper_grade = build_paper_grade(
            validity_summary=validity or {},
            ineligibility_reasons=["missing_checkpoint_paper_grade"],
        )
    if not bool(checkpoint_paper_grade.get("eligible", False)):
        _raise_or_record_mismatch(
            field="checkpoint_paper_grade",
            expected=True,
            actual=bool(checkpoint_paper_grade.get("eligible", False)),
            reasons=compatibility_reasons,
            allow_mismatch=bool(args.allow_incompatible_checkpoint),
        )

    # 推理阶段统一把各编码器 freeze 掉，因为这里只需要前向。
    model = FusionClassifier(
        num_classes=len(label_names),
        freeze_audio=True,
        freeze_video=True,
        freeze_flow=True,
        freeze_rgb=True,
        freeze_prosody=True,
        text_model=text_model,
        freeze_text=True,
        audio_model=audio_model,
        audio_model_revision=(audio_model_revision or None),
        video_backbone=video_backbone,
        video_model=video_model,
        fusion_mode=fusion_mode,
        gate_temperature=gate_temperature,
        gate_scale=gate_scale,
        delta_scale=delta_scale,
        modality_dropout=modality_dropout,
        intensity_head=bool(use_intensity),
    )
    if bool(args.allow_incompatible_checkpoint):
        # 只有在显式“允许不兼容”的情况下，才允许 strict=False。
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"WARNING: missing checkpoint keys: {missing[:6]}" + (" ..." if len(missing) > 6 else ""), flush=True)
            compatibility_reasons.append("checkpoint_missing_keys")
        if unexpected:
            print(f"WARNING: unexpected checkpoint keys: {unexpected[:6]}" + (" ..." if len(unexpected) > 6 else ""), flush=True)
            compatibility_reasons.append("checkpoint_unexpected_keys")
    else:
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    setattr(model, "_label_names", list(label_names))
    checkpoint_provenance = _checkpoint_provenance(
        ckpt if isinstance(ckpt, dict) else {},
        deterministic_policy={"seed": 0, "deterministic_requested": True},
        profile={},
    )
    checkpoint_input_cache_contract = (
        dict(ckpt.get("input_cache_contract"))
        if isinstance(ckpt, dict) and isinstance(ckpt.get("input_cache_contract"), dict)
        else None
    )
    return (
        model,
        bool(use_intensity),
        label_names,
        task_mode,
        task_speaker_id,
        text_policy,
        int(checkpoint_max_text_len),
        validity,
        checkpoint_contract,
        checkpoint_paper_grade,
        checkpoint_provenance,
        checkpoint_input_cache_contract,
        compatibility_reasons,
    )


def _truncate_text(text: str, max_len: int) -> str:
    """按需要截断长文本，避免日志/JSON 中出现过长样本内容。"""

    if int(max_len) <= 0 or len(text) <= int(max_len):
        return text
    return text[: int(max_len)] + "..."


def _format_error_record(item: dict[str, Any], err: Exception, *, text_max_len: int) -> dict[str, Any]:
    """把单个失败样本格式化成标准错误记录。"""

    return {
        "stem": str(item.get("stem", item.get("seq", ""))),
        "status": "error",
        "error": f"{type(err).__name__}: {err}",
        "speaker_id": str(item.get("speaker_id", "UNKNOWN")),
        "label": str(item.get("label_en", "")),
        "text": _truncate_text(str(item.get("mn", item.get("text", ""))), text_max_len),
    }


def _infer_batch(
    *,
    items: list[dict[str, Any]],
    preprocessor: GpuStreamPreprocessor,
    tokenizer: Any | None,
    text_policy: str,
    max_text_len: int,
    model: FusionClassifier,
    use_intensity: bool,
    amp_mode: str,
) -> list[dict[str, Any]]:
    """对一个 batch 执行前向推理，并返回逐样本记录。"""

    _lazy_runtime_imports()
    # 复用训练时同一套 preprocessor，保证推理前的张量构造方式与训练一致。
    batch = preprocessor.prepare_batch(items, tokenizer=tokenizer, text_policy=text_policy, max_text_len=max_text_len)
    labels = batch["labels"]
    device = next(model.parameters()).device
    # 推理始终关闭梯度，但仍可根据设备和 AMP 配置使用 autocast。
    with torch.no_grad(), autocast_context(device, amp_mode):
        if model.video_backbone == "videomae":
            flow = None
        else:
            flow = batch.get("flow", None)
        logits_or_pair = model(
            flow,
            batch["audio"],
            batch["prosody"],
            text_inputs=batch.get("text_inputs", None),
            return_intensity=bool(use_intensity),
            rgb=batch.get("rgb", None),
            audio_lens=batch.get("audio_lens", None),
        )
        if bool(use_intensity):
            logits, pred_intensity = logits_or_pair
        else:
            logits = logits_or_pair
            pred_intensity = None
    _assert_finite_tensor("logits", logits)
    _assert_finite_tensor("pred_intensity", pred_intensity)
    probs = torch.softmax(logits, dim=1)
    _assert_finite_tensor("probabilities", probs)
    pred = logits.argmax(dim=1)
    pred_probs = probs.gather(1, pred.view(-1, 1)).squeeze(1).detach().cpu().tolist()
    labels_cpu = labels.detach().cpu().tolist()
    pred_cpu = pred.detach().cpu().tolist()
    intensity = batch["intensity"].detach().cpu().to(torch.float32)
    intensity_mask = batch["intensity_mask"].detach().cpu()
    pred_intensity_cpu = pred_intensity.detach().cpu().to(torch.float32) if pred_intensity is not None else None
    label_names = list(getattr(model, "_label_names", []))

    # 每个样本都会写成一条独立记录，方便后续做误差分析和重放。
    records: list[dict[str, Any]] = []
    for idx, stem in enumerate(batch["stems"]):
        rec = {
            "stem": str(stem),
            "status": "ok",
            "label": label_names[int(labels_cpu[idx])],
            "pred": label_names[int(pred_cpu[idx])],
            "match": bool(int(labels_cpu[idx]) == int(pred_cpu[idx])),
            "probability": float(pred_probs[idx]),
            "speaker_id": str(batch["speaker_id"][idx]),
            "text": str(batch["mn"][idx]),
        }
        if bool(use_intensity):
            if bool(intensity_mask[idx].item()):
                rec["intensity_gt"] = float(intensity[idx].item())
                rec["pred_intensity"] = float(pred_intensity_cpu[idx].item()) if pred_intensity_cpu is not None else None
            else:
                rec["intensity_gt"] = None
                rec["pred_intensity"] = None
        records.append(rec)
    return records


def main() -> None:
    """主推理入口。

    整体流程：
    1. 解析参数并加载 checkpoint
    2. 校验 checkpoint 契约与当前 manifest/CLI 是否一致
    3. 构建 dataset、DataLoader、preprocessor
    4. 逐 batch 推理并写出 JSONL
    5. 从逐样本结果汇总出 metrics.json
    """

    args = parse_args()
    _validate_compat_args(args)
    _lazy_runtime_imports()
    # 推理本身不做优化，但仍固定种子，确保 DataLoader 与随机路径可复现。
    deterministic_policy = set_seed(0)

    split_manifest_path = args.split_manifest.expanduser()
    attempt_dir: Path | None = None
    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser()
        attempt_dir = resolve_attempt_dir(run_dir, attempt_id=str(args.attempt_id or ""), prefer_published=True)
        bundle = resolve_best_bundle(attempt_dir)
        ckpt_path = Path(bundle["checkpoint_path"]).expanduser()
        if args.output is None:
            out_path = attempt_dir / "published" / f"inference_{str(args.subset)}.jsonl"
        else:
            out_path = args.output.expanduser()
    else:
        if args.checkpoint is None:
            raise RuntimeError("Either --run-dir or --checkpoint must be provided.")
        ckpt_path = args.checkpoint.expanduser()
        out_path = args.output.expanduser() if args.output is not None else Path("outputs/motion_prosody/inference_results.jsonl")
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if attempt_dir is None and ckpt_path.name != "best.pt" and not bool(args.allow_incompatible_checkpoint):
        raise RuntimeError(f"checkpoint must point to best.pt for paper-grade evaluation, got: {ckpt_path.name}")

    profile = detect_runtime(str(args.device))
    device = select_device(str(args.device))
    resolved_amp_mode = resolve_amp_mode(str(args.amp_mode), profile)
    print(f"Using device: {device}", flush=True)

    (
        model,
        use_intensity,
        label_names,
        task_mode,
        task_speaker_id,
        text_policy,
        checkpoint_max_text_len,
        validity,
        checkpoint_contract,
        checkpoint_paper_grade,
        checkpoint_provenance,
        checkpoint_input_cache_contract,
        compatibility_reasons,
    ) = _load_model(
        ckpt_path,
        device=device,
        args=args,
    )
    resolved_text_max_len = int(args.text_max_len)
    if resolved_text_max_len <= 0:
        resolved_text_max_len = int(checkpoint_max_text_len)
    if resolved_text_max_len <= 0:
        resolved_text_max_len = 128
    # 后续所有“为什么这次评测不再算 paper-grade”的原因都汇总到这里。
    paper_grade_reasons = list(compatibility_reasons)
    if bool(args.allow_incompatible_checkpoint):
        paper_grade_reasons.append("allow_incompatible_checkpoint_enabled")
    if ckpt_path.name != "best.pt":
        paper_grade_reasons.append("checkpoint_name_mismatch")
    # `flow_encoder_variant` 是这条主线里很关键的结构语义字段，
    # 目的是防止不同版本的 flow 分支 checkpoint 被静默混评。
    _raise_or_record_mismatch(
        field="flow_encoder_variant",
        expected=str(checkpoint_contract.get("flow_encoder_variant", LEGACY_FLOW_VIDEO_ENCODER_VARIANT)),
        actual=str(FLOW_VIDEO_ENCODER_VARIANT),
        reasons=paper_grade_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    args.ablation = _normalize_optional_text(args.ablation) or str(checkpoint_contract.get("ablation", "full") or "full")
    args.sample_rate = int(args.sample_rate if args.sample_rate is not None else checkpoint_contract.get("sample_rate", 24000))
    args.max_audio_sec = float(
        args.max_audio_sec if args.max_audio_sec is not None else checkpoint_contract.get("max_audio_sec", 6.0)
    )
    args.num_frames = int(args.num_frames if args.num_frames is not None else checkpoint_contract.get("num_frames", 64))
    _raise_or_record_mismatch(
        field="ablation",
        expected=checkpoint_contract.get("ablation"),
        actual=str(args.ablation),
        reasons=paper_grade_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="sample_rate",
        expected=int(checkpoint_contract.get("sample_rate", args.sample_rate)),
        actual=int(args.sample_rate),
        reasons=paper_grade_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="max_audio_sec",
        expected=float(checkpoint_contract.get("max_audio_sec", args.max_audio_sec)),
        actual=float(args.max_audio_sec),
        reasons=paper_grade_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="num_frames",
        expected=int(checkpoint_contract.get("num_frames", args.num_frames)),
        actual=int(args.num_frames),
        reasons=paper_grade_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    zero_video, zero_audio, zero_text = resolve_ablation_flags(
        ablation=str(args.ablation),
        zero_video=bool(args.zero_video),
        zero_audio=bool(args.zero_audio),
        zero_text=bool(args.zero_text),
    )
    _raise_or_record_mismatch(
        field="zero_video",
        expected=bool(checkpoint_contract.get("zero_video", zero_video)),
        actual=bool(zero_video),
        reasons=paper_grade_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="zero_audio",
        expected=bool(checkpoint_contract.get("zero_audio", zero_audio)),
        actual=bool(zero_audio),
        reasons=paper_grade_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="zero_text",
        expected=bool(checkpoint_contract.get("zero_text", zero_text)),
        actual=bool(zero_text),
        reasons=paper_grade_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    args.zero_video = bool(zero_video)
    args.zero_audio = bool(zero_audio)
    args.zero_text = bool(zero_text)
    if str(text_policy) == "drop":
        args.zero_text = True
    inference_provenance = build_run_provenance(
        runtime_profile=profile.to_jsonable(),
        deterministic_policy=deterministic_policy,
        repo_root=Path(__file__).resolve().parent,
    )

    # 推理使用的是“当前磁盘上的 manifest”，因此需要重新计算 validity，
    # 再和 checkpoint 当时的契约逐项比较，防止 stale metadata。
    manifest = load_split_manifest(split_manifest_path)
    manifest_hash = manifest_sha256(split_manifest_path)
    manifest_summary = manifest.get("summary", {})
    dataset_kind = str(manifest.get("dataset_kind", "") or "")
    validity = build_validity_summary(manifest_summary, task_mode, task_speaker_id)
    _raise_or_record_mismatch(
        field="dataset_kind",
        expected=str(checkpoint_contract.get("dataset_kind", dataset_kind)),
        actual=dataset_kind,
        reasons=paper_grade_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="manifest_sha256",
        expected=str(checkpoint_contract.get("manifest_sha256", manifest_hash)),
        actual=manifest_hash,
        reasons=paper_grade_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="label_names",
        expected=list(checkpoint_contract.get("label_names", label_names)),
        actual=list(label_names),
        reasons=paper_grade_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="claim_scope",
        expected=str(checkpoint_contract.get("claim_scope", validity.get("claim_scope", ""))),
        actual=str(validity.get("claim_scope", "")),
        reasons=paper_grade_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )
    _raise_or_record_mismatch(
        field="scientific_validity",
        expected=bool(checkpoint_contract.get("scientific_validity", validity.get("scientific_validity", False))),
        actual=bool(validity.get("scientific_validity", False)),
        reasons=paper_grade_reasons,
        allow_mismatch=bool(args.allow_incompatible_checkpoint),
    )

    items = prepare_manifest_items_for_task(
        filter_manifest_items_for_task(select_manifest_items(manifest, args.subset), task_mode, task_speaker_id),
        task_mode=task_mode,
        speaker_id=task_speaker_id,
        map_label_to_task_index=map_label_to_task_index,
    )
    if int(args.limit) > 0:
        items = items[: int(args.limit)]
    if not items:
        raise RuntimeError("No usable manifest items remain after task/subset filtering.")

    ingress_cfg = ManifestIngressConfig(
        sample_rate=int(args.sample_rate),
        max_audio_sec=float(args.max_audio_sec),
        audio_backend_mode=str(args.audio_backend),
        video_decode_backend=str(args.video_decode_backend),
        num_frames=int(args.num_frames),
        zero_audio=bool(args.zero_audio),
        zero_video=bool(args.zero_video),
        video_backbone=str(model.video_backbone),
    )
    need_audio = not bool(args.zero_audio)
    need_video = (not bool(args.zero_video)) and str(model.video_backbone) in {"flow", "videomae", "dual"}
    need_text = not bool(args.zero_text)
    input_cache_contract = None
    input_cache_dir = args.input_cache.expanduser() if args.input_cache is not None else None
    if input_cache_dir is not None:
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
            text_model=str(args.text_model).strip() or str(model.text.model_name),
            max_text_len=int(resolved_text_max_len),
            need_audio=bool(need_audio),
            need_video=bool(need_video),
            need_text=bool(need_text),
            text_policy=str(text_policy),
        )
        if cache_mismatch_reasons:
            raise RuntimeError(
                "Input cache contract mismatch: " + ", ".join(str(reason) for reason in cache_mismatch_reasons)
            )
        if isinstance(checkpoint_input_cache_contract, dict) and checkpoint_input_cache_contract:
            _raise_or_record_mismatch(
                field="input_cache_contract",
                expected=checkpoint_input_cache_contract,
                actual=input_cache_contract,
                reasons=paper_grade_reasons,
                allow_mismatch=bool(args.allow_incompatible_checkpoint),
            )
        ds = CachedManifestDataset(
            items,
            ingress=ingress_cfg,
            cache_dir=input_cache_dir,
            text_policy=str(text_policy),
            runtime_profile=profile,
        )
    else:
        ds = StreamingManifestDataset(items, ingress=ingress_cfg)
    tokenizer = None
    if not bool(args.zero_text) and input_cache_contract is None:
        # 推理文本 token 同样会先缓存到 manifest item 中，
        # 避免每个 batch 反复调用 tokenizer。
        tokenizer = _load_tokenizer(str(args.text_model).strip() or str(model.text.model_name))
        cache_manifest_text_tokens(ds.items, tokenizer, max_text_len=int(resolved_text_max_len), text_policy=str(text_policy))

    resolved_batch_size = resolve_batch_size(
        args.batch_size,
        phase="inference",
        profile=profile,
        feature_cache=False,
        video_backbone=str(model.video_backbone),
        freeze_audio=True,
        freeze_text=True,
        freeze_flow=True,
        freeze_rgb=True,
    )
    resolved_num_workers = resolve_worker_count(
        args.num_workers,
        phase="inference",
        profile=profile,
        dataset_in_memory=bool(getattr(ds, "in_memory", False)),
        cache_backed=bool(input_cache_contract is not None),
        total_items=len(items),
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
    loader = DataLoader(
        ds,
        batch_size=int(resolved_batch_size),
        shuffle=False,
        worker_init_fn=make_dataloader_worker_init_fn(0),
        generator=make_torch_generator(0),
        **dl_kwargs,
    )

    preprocessor = GpuStreamPreprocessor(
        GpuStreamConfig(
            device=device,
            video_backbone=str(model.video_backbone),
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

    attempted = 0
    error_count = 0
    print_every = max(0, int(args.print_every))
    error_types: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    backend_logged = False
    run_status = "completed"

    iterator = loader
    if not bool(args.no_progress):
        iterator = tqdm(loader, desc="Batch inference [gpu_stream]", unit="batch", dynamic_ncols=True)

    for batch_items in iterator:
        attempted += len(batch_items)
        try:
            batch_records = _infer_batch(
                items=batch_items,
                preprocessor=preprocessor,
                tokenizer=tokenizer,
                text_policy=str(text_policy),
                max_text_len=int(resolved_text_max_len),
                model=model,
                use_intensity=bool(use_intensity),
                amp_mode=resolved_amp_mode,
            )
            if not backend_logged:
                print(
                    json.dumps(
                        {
                            "gpu_stream_backends": preprocessor.backend_summary(),
                            "gpu_stream_prepare_stats": preprocessor.consume_prepare_stats(),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                backend_logged = True
            records.extend(batch_records)
        except Exception as batch_err:
            if bool(args.fail_fast):
                raise
            # 非 fail-fast 模式下，先尝试把整批拆成逐样本推理，
            # 尽量保住其它样本的结果，只把真正失败的样本记成 error record。
            if len(batch_items) > 1:
                for item in batch_items:
                    try:
                        records.extend(
                            _infer_batch(
                                items=[item],
                                preprocessor=preprocessor,
                                tokenizer=tokenizer,
                                text_policy=str(text_policy),
                                max_text_len=int(resolved_text_max_len),
                                model=model,
                                use_intensity=bool(use_intensity),
                                amp_mode=resolved_amp_mode,
                            )
                        )
                    except Exception as item_err:
                        error_count += 1
                        error_types[type(item_err).__name__] += 1
                        records.append(_format_error_record(item, item_err, text_max_len=int(resolved_text_max_len)))
            else:
                error_count += 1
                error_types[type(batch_err).__name__] += 1
                records.append(_format_error_record(batch_items[0], batch_err, text_max_len=int(resolved_text_max_len)))

        if print_every > 0 and len(records) > 0 and (len(records) % print_every) == 0:
            ok_records = [rec for rec in records if rec.get("status") == "ok"]
            correct = sum(1 for rec in ok_records if bool(rec.get("match")))
            acc = float(correct / max(1, len(ok_records)))
            print(f"[{len(records)}/{len(items)}] ok={len(ok_records)} err={error_count} acc={acc:.4f}", flush=True)

    # 逐样本记录是最原始的正式产物；后面的 metrics 汇总都会从这里再聚合。
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if str(args.print_result) in {"all", str(rec.get("status", "none"))}:
                line = json.dumps(rec, ensure_ascii=False) if bool(args.print_json) else (
                    f"{str(rec.get('status', 'ok')).upper()} stem={rec.get('stem')} "
                    f"label={rec.get('label')} pred={rec.get('pred')} p={rec.get('probability')}"
                )
                print(line, flush=True)
                if bool(args.print_text) and str(rec.get("text", "")):
                    print(f"    text={_truncate_text(str(rec.get('text', '')), int(resolved_text_max_len))}", flush=True)

    ok_records = [rec for rec in records if rec.get("status") == "ok"]
    if error_count > 0:
        run_status = "completed_with_item_errors"
        paper_grade_reasons.append("inference_item_errors")
    y_true = [label_names.index(str(rec["label"])) for rec in ok_records]
    y_pred = [label_names.index(str(rec["pred"])) for rec in ok_records]
    class_summary = classification_summary(y_true, y_pred, label_names)

    reg_true = [float(rec["intensity_gt"]) for rec in ok_records if rec.get("intensity_gt") is not None and rec.get("pred_intensity") is not None]
    reg_pred = [float(rec["pred_intensity"]) for rec in ok_records if rec.get("intensity_gt") is not None and rec.get("pred_intensity") is not None]
    mse = None
    mae = None
    ccc_value = None
    if reg_true and reg_pred:
        rt = torch.tensor(reg_true, dtype=torch.float32)
        rp = torch.tensor(reg_pred, dtype=torch.float32)
        diff = (rp - rt).to(torch.float64)
        mse = float((diff * diff).mean().item())
        mae = float(diff.abs().mean().item())
        ccc_value = ccc(rp, rt)

    speaker_baseline = None
    speaker_only = None
    if args.subset in {"train", "val"} and task_mode == "confounded_7way":
        train_items = filter_manifest_items_for_task(select_manifest_items(manifest, "train"), task_mode, task_speaker_id)
        eval_items = filter_manifest_items_for_task(select_manifest_items(manifest, args.subset), task_mode, task_speaker_id)
        speaker_baseline = speaker_majority_baseline(train_items, eval_items, label_names)
        speaker_only = speaker_only_baseline(train_items, eval_items, label_names)

    paper_grade = build_paper_grade(validity_summary=validity, ineligibility_reasons=paper_grade_reasons)

    # metrics_summary 是推理阶段的正式摘要文件，
    # 聚合脚本会直接消费这里的字段做多 seed 统计。
    metrics_summary = {
        "split_manifest": str(split_manifest_path),
        "subset": str(args.subset),
        "benchmark_tag": str(args.benchmark_tag),
        "dataset_kind": dataset_kind,
        "ablation": str(args.ablation),
        "task_mode": task_mode,
        "speaker_id": task_speaker_id,
        "text_policy": str(text_policy),
        "pipeline": "gpu_stream",
        "label_names": list(label_names),
        "zero_video": bool(args.zero_video),
        "zero_audio": bool(args.zero_audio),
        "zero_text": bool(args.zero_text),
        "checkpoint": str(ckpt_path),
        "output": str(out_path),
        "attempted": int(attempted),
        "ok": int(len(ok_records)),
        "errors": int(error_count),
        "skipped": 0,
        "accuracy_on_ok": float(class_summary.get("accuracy", 0.0)),
        "macro_f1_on_ok": float(class_summary.get("macro_f1", 0.0)),
        "per_class_recall": class_summary.get("per_class_recall", {}),
        "confusion_matrix": class_summary.get("confusion_matrix", []),
        "support": class_summary.get("support", {}),
        "pred_counts": class_summary.get("pred_counts", {}),
        "intensity_enabled": bool(use_intensity),
        "intensity_n": int(len(reg_true)),
        "intensity_mse": mse,
        "intensity_mae": mae,
        "intensity_ccc": ccc_value,
        "device": str(device),
        "runtime_profile": profile.to_jsonable(),
        "amp_mode": resolved_amp_mode,
        "resolved_batch_size": int(resolved_batch_size),
        "resolved_num_workers": int(resolved_num_workers),
        "resolved_prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
        "input_cache": str(input_cache_dir) if input_cache_dir is not None else None,
        "input_cache_contract": input_cache_contract,
        "input_cache_in_memory": bool(getattr(ds, "in_memory", False)),
        "manifest_sha256": manifest_hash,
        "manifest_summary": manifest_summary,
        "validity": validity,
        "error_types": dict(sorted(error_types.items())),
        "paper_contract": checkpoint_contract,
        "provenance": inference_provenance,
        "checkpoint_paper_grade": checkpoint_paper_grade,
        "checkpoint_provenance": checkpoint_provenance,
        "checkpoint_input_cache_contract": checkpoint_input_cache_contract,
        "paper_grade": paper_grade,
        "run_status": str(run_status),
    }
    if speaker_baseline is not None:
        metrics_summary["speaker_majority_baseline"] = speaker_baseline
    if speaker_only is not None:
        metrics_summary["speaker_only_baseline"] = speaker_only

    metrics_path = out_path.parent / f"{out_path.stem}.metrics.json"
    metrics_path.write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if attempt_dir is not None:
        register_published_inference_output(attempt_dir, subset=str(args.subset), output_path=out_path)
    print(json.dumps(metrics_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
