"""主线输入缓存协议与辅助工具。

这个模块只服务当前 manifest-driven `gpu_stream` 主线。它定义了一种
“论文级等价的输入缓存”：
- 音频缓存的是 `load_audio()` 的输出波形
- 视频缓存的是已经均匀采样好的原始 `uint8` 帧
- 文本缓存的是 tokenizer 输出的 token

它故意不缓存以下内容：
- prosody 向量
- flow / optical-flow 特征
- 音频或文本 encoder embedding

原因是这些内容会改变当前主线的计算语义，不能再被当作“仅仅是更快的输入路径”。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

INPUT_CACHE_PROTOCOL_VERSION = "mainline_input_cache_v1"


def manifest_item_cache_key(item: dict[str, Any]) -> str:
    """为 manifest item 生成稳定的缓存 key。"""

    split = str(item.get("split", "") or "").strip().lower()
    sample_id = str(item.get("seq") or item.get("sample_id") or item.get("stem") or "").strip()
    if not sample_id:
        raise RuntimeError("Cannot derive input-cache key because manifest item has no seq/sample_id/stem")
    return f"{split}:{sample_id}" if split else sample_id


def sample_relpath_for_key(cache_key: str) -> Path:
    """把 cache key 映射成两级目录下的样本文件路径。"""

    digest = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()
    return Path("samples") / digest[:2] / f"{digest}.pt"


def cache_meta_path(cache_dir: Path) -> Path:
    """返回缓存元数据文件路径。"""

    return cache_dir / "cache_meta.json"


def cache_index_path(cache_dir: Path) -> Path:
    """返回缓存索引文件路径。"""

    return cache_dir / "index.jsonl"


def load_input_cache_meta(cache_dir: Path) -> dict[str, Any]:
    """读取输入缓存元数据。"""

    path = cache_meta_path(cache_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Input cache metadata not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Input cache metadata must be a JSON object: {path}")
    return data


def load_input_cache_index(cache_dir: Path) -> list[dict[str, Any]]:
    """读取输入缓存索引。"""

    path = cache_index_path(cache_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Input cache index not found: {path}")
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = str(line).strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise RuntimeError(f"Input cache index row must be a JSON object: {path}:{line_no}")
            entries.append(row)
    return entries


def index_entries_by_key(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """把索引列表转换成以 cache key 为键的查找表。"""

    mapping: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = str(entry.get("cache_key", "") or "")
        if not key:
            raise RuntimeError("Input cache index entry missing cache_key")
        if key in mapping:
            raise RuntimeError(f"Duplicate cache_key in input cache index: {key}")
        mapping[key] = dict(entry)
    return mapping


def build_input_cache_contract(meta: dict[str, Any]) -> dict[str, Any]:
    """从元数据提取出需要进入训练/推理记录的缓存契约。"""

    return {
        "protocol_version": str(meta.get("protocol_version", INPUT_CACHE_PROTOCOL_VERSION)),
        "manifest_sha256": str(meta.get("manifest_sha256", "")),
        "dataset_kind": str(meta.get("dataset_kind", "")),
        "sample_rate": int(meta.get("sample_rate", 0) or 0),
        "max_audio_sec": float(meta.get("max_audio_sec", 0.0) or 0.0),
        "num_frames": int(meta.get("num_frames", 0) or 0),
        "text_model": str(meta.get("text_model", "")),
        "max_text_len": int(meta.get("max_text_len", 0) or 0),
        "has_audio": bool(meta.get("has_audio", False)),
        "has_video": bool(meta.get("has_video", False)),
        "has_text_full_tokens": bool(meta.get("has_text_full_tokens", False)),
        "has_text_masked_tokens": bool(meta.get("has_text_masked_tokens", False)),
        "subset": str(meta.get("subset", "all")),
    }


def _hf_model_name_matches(left: str, right: str) -> bool:
    """尽量温和地比较两个 Hugging Face 模型名是否指向同一模型。"""

    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if left_text == right_text:
        return True
    if not left_text or not right_text:
        return False
    return left_text.split("/")[-1] == right_text.split("/")[-1]


def validate_input_cache_contract(
    contract: dict[str, Any],
    *,
    manifest_sha256: str,
    dataset_kind: str,
    sample_rate: int,
    max_audio_sec: float,
    num_frames: int,
    text_model: str,
    max_text_len: int,
    need_audio: bool,
    need_video: bool,
    need_text: bool,
    text_policy: str,
) -> list[str]:
    """校验当前运行参数是否与输入缓存兼容。"""

    reasons: list[str] = []
    if str(contract.get("protocol_version", "")) != INPUT_CACHE_PROTOCOL_VERSION:
        reasons.append("protocol_version_mismatch")
    if str(contract.get("manifest_sha256", "")) != str(manifest_sha256):
        reasons.append("manifest_sha256_mismatch")
    if str(contract.get("dataset_kind", "")) != str(dataset_kind):
        reasons.append("dataset_kind_mismatch")

    if bool(need_audio):
        if not bool(contract.get("has_audio", False)):
            reasons.append("missing_cached_audio")
        if int(contract.get("sample_rate", 0) or 0) != int(sample_rate):
            reasons.append("sample_rate_mismatch")
        if abs(float(contract.get("max_audio_sec", 0.0) or 0.0) - float(max_audio_sec)) > 1.0e-9:
            reasons.append("max_audio_sec_mismatch")

    if bool(need_video):
        if not bool(contract.get("has_video", False)):
            reasons.append("missing_cached_video")
        if int(contract.get("num_frames", 0) or 0) != int(num_frames):
            reasons.append("num_frames_mismatch")

    if bool(need_text):
        if not _hf_model_name_matches(str(contract.get("text_model", "")), str(text_model)):
            reasons.append("text_model_mismatch")
        if int(contract.get("max_text_len", 0) or 0) != int(max_text_len):
            reasons.append("max_text_len_mismatch")
        if str(text_policy) == "full" and not bool(contract.get("has_text_full_tokens", False)):
            reasons.append("missing_full_text_tokens")
        if str(text_policy) == "mask_emotion_cues" and not bool(contract.get("has_text_masked_tokens", False)):
            reasons.append("missing_masked_text_tokens")

    return reasons


def count_selected_cache_bytes(
    *,
    entries_by_key: dict[str, dict[str, Any]],
    keys: list[str],
) -> int:
    """统计当前要使用的缓存样本预计总字节数。"""

    total = 0
    for key in keys:
        entry = entries_by_key.get(str(key))
        if entry is None:
            raise RuntimeError(f"Input cache index missing selected key: {key}")
        total += int(entry.get("sample_bytes", 0) or 0)
    return int(total)


def save_input_cache_meta(cache_dir: Path, meta: dict[str, Any]) -> None:
    """写出缓存元数据。"""

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_meta_path(cache_dir).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def save_input_cache_index(cache_dir: Path, entries: list[dict[str, Any]]) -> None:
    """写出缓存索引。"""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_index_path(cache_dir)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


_WORKER_THREADS_LIMITED = False


def _limit_worker_threads() -> None:
    """限制缓存构建 worker 的线程数，避免多进程时线程过度膨胀。"""

    global _WORKER_THREADS_LIMITED
    if _WORKER_THREADS_LIMITED:
        return
    _WORKER_THREADS_LIMITED = True
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass


def build_cached_media_payload(task: dict[str, Any]) -> dict[str, Any]:
    """在构建阶段处理单个样本的媒体缓存。"""

    _limit_worker_threads()

    from data import _load_sampled_video_frames_cpu
    from predecode_motion_audio import load_audio
    from runtime_adapt import estimate_tensor_bytes
    import torch

    key = str(task["cache_key"])
    payload: dict[str, Any] = {
        "meta": {
            "cache_key": key,
            "split": str(task.get("split", "")),
            "seq": str(task.get("seq", "")),
            "sample_id": str(task.get("sample_id", "")),
            "speaker_id": str(task.get("speaker_id", "")),
            "label_en": str(task.get("label_en", "")),
            "dataset_kind": str(task.get("dataset_kind", "")),
        }
    }

    if bool(task.get("need_audio", False)):
        wav_f32, audio_backend = load_audio(
            Path(str(task["audio_path"])).expanduser(),
            sample_rate=int(task["sample_rate"]),
            max_sec=float(task["max_audio_sec"]),
            backend_mode=str(task.get("audio_backend_mode", "auto")),
        )
        payload["audio"] = wav_f32.to(torch.float32)
        payload["_audio_backend"] = str(audio_backend)

    if bool(task.get("need_video", False)):
        frames_cpu, video_backend = _load_sampled_video_frames_cpu(
            Path(str(task["video_path"])).expanduser(),
            num_frames=int(task["num_frames"]),
            backend=str(task.get("video_decode_backend", "auto")),
        )
        payload["video_frames"] = frames_cpu.contiguous()
        payload["_video_backend"] = str(video_backend)

    full_tokens = task.get("text_full", None)
    if isinstance(full_tokens, dict) and full_tokens:
        payload["text_full"] = full_tokens
    masked_tokens = task.get("text_masked", None)
    if isinstance(masked_tokens, dict) and masked_tokens:
        payload["text_masked"] = masked_tokens

    return {
        "cache_key": key,
        "payload": payload,
        "sample_bytes": int(estimate_tensor_bytes(payload)),
    }
