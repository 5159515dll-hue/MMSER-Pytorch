"""Manifest-driven dataset ingress helpers for the active gpu_stream mainline."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Optional

import torch
from torch.utils.data import Dataset

from input_cache import (
    build_input_cache_contract,
    count_selected_cache_bytes,
    index_entries_by_key,
    load_input_cache_entry_payload,
    load_input_cache_index,
    load_input_cache_meta,
    manifest_item_cache_key,
)
from predecode_motion_audio import load_audio
from runtime_adapt import RuntimeProfile, should_keep_dataset_in_memory
from text_policy_utils import select_text_for_policy
from video_motion import _read_video_frames_cv2

_DECORD_CPU_READY = False
_DECORD_CPU = None
ProgressLogger = Callable[[str], None]


def _emit_progress(progress_logger: Optional[ProgressLogger], message: str) -> None:
    """向可选的进度记录器发送一条消息。"""

    if progress_logger is not None:
        progress_logger(str(message))


def _maybe_import_decord_cpu() -> Any | None:
    """按需导入 CPU 版 decord，并把结果缓存到模块级变量。

    DataLoader worker 会频繁读取视频。这里用懒加载方式避免在没有 decord
    或根本不需要视频解码时，让整个模块导入直接失败。
    """

    global _DECORD_CPU_READY, _DECORD_CPU
    if _DECORD_CPU_READY:
        return _DECORD_CPU
    _DECORD_CPU_READY = True
    try:
        import decord  # type: ignore

        decord.bridge.set_bridge("torch")
        _DECORD_CPU = decord
    except Exception:
        _DECORD_CPU = None
    return _DECORD_CPU


def _select_indices(total: int, num_frames: int) -> list[int]:
    """把任意长度的视频映射成固定数量的采样帧下标。"""

    if total <= 0:
        return [0] * max(1, int(num_frames))
    if total >= int(num_frames):
        return torch.linspace(0, total - 1, int(num_frames)).round().to(torch.long).tolist()
    return list(range(total)) + [total - 1] * (int(num_frames) - total)


def _load_sampled_video_frames_cpu(path: Path, *, num_frames: int, backend: str) -> tuple[torch.Tensor, str]:
    """在 CPU 侧读取并采样视频帧。

    返回值除了帧张量，还会携带实际命中的后端名字，方便训练/推理日志统计
    当前到底是走了 `decord_cpu` 还是回退到 `cv2`。
    """

    backend = str(backend or "auto").strip().lower()
    decord = _maybe_import_decord_cpu() if backend in {"auto", "decord"} else None
    if decord is not None:
        try:
            vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
            idxs = _select_indices(len(vr), int(num_frames))
            frames = vr.get_batch(idxs)
            if not isinstance(frames, torch.Tensor):
                frames = torch.as_tensor(frames)
            return frames.to(torch.uint8).contiguous(), "decord_cpu"
        except Exception:
            pass

    frames_np = _read_video_frames_cv2(path)
    if not frames_np:
        return torch.zeros((int(num_frames), 224, 224, 3), dtype=torch.uint8), "cv2_empty"
    idxs = _select_indices(len(frames_np), int(num_frames))
    frames = torch.stack([torch.from_numpy(frames_np[i][..., ::-1].copy()) for i in idxs], dim=0)
    return frames.to(torch.uint8).contiguous(), "cv2"


@dataclass
class ManifestIngressConfig:
    """Worker-side media ingress configuration for manifest streaming."""

    sample_rate: int = 16000
    max_audio_sec: float = 6.0
    audio_backend_mode: str = "auto"
    video_decode_backend: str = "auto"
    num_frames: int = 64
    zero_audio: bool = False
    zero_video: bool = False
    video_backbone: str = "dual"


def _manifest_item_to_base_output(item: dict[str, Any], idx: int) -> dict[str, Any]:
    """把原始 manifest item 规范化成主线数据入口统一使用的样本头。"""

    label_idx = item.get("label_idx", None)
    if label_idx is None:
        raise RuntimeError(f"Manifest item missing label_idx: {item.get('seq') or item.get('sample_id') or idx}")

    intensity = item.get("intensity", None)
    intensity_t = torch.tensor(float("nan"), dtype=torch.float32) if intensity is None else torch.tensor(float(intensity), dtype=torch.float32)
    return {
        "label": torch.tensor(int(label_idx), dtype=torch.long),
        "stem": str(item.get("seq") or item.get("sample_id") or idx),
        "text": str(item.get("text", item.get("mn", ""))),
        "mn": str(item.get("mn", item.get("text", ""))),
        "masked_text": str(item.get("masked_text", item.get("masked_mn", ""))),
        "masked_mn": str(item.get("masked_mn", item.get("masked_text", ""))),
        "speaker_id": str(item.get("speaker_id", "UNKNOWN")),
        "text_cue_flag": bool(item.get("text_cue_flag", False)),
        "cue_severity": str(item.get("cue_severity", "none")),
        "prompt_group_id": str(item.get("prompt_group_id", "")),
        "_global_label_en": str(item.get("label_en", item.get("_global_label_en", ""))),
        "dataset_kind": str(item.get("dataset_kind", "")),
        "intensity": intensity_t,
        "video_path": str(item.get("video_path") or ""),
        "audio_path": str(item.get("audio_path") or ""),
    }


def _attach_text_tokens(out: dict[str, Any], token_bundle: dict[str, Any]) -> None:
    """把缓存好的 token 写回主线统一使用的 `_text_*` 字段。"""

    out["_text_input_ids"] = token_bundle["input_ids"].to(torch.long)
    out["_text_attention_mask"] = token_bundle["attention_mask"].to(torch.long)
    if "token_type_ids" in token_bundle:
        out["_text_token_type_ids"] = token_bundle["token_type_ids"].to(torch.long)


class StreamingManifestDataset(Dataset):
    """Load audio sidecars and sampled video frames inside DataLoader workers."""

    def __init__(self, items: list[dict[str, Any]], *, ingress: ManifestIngressConfig):
        """保存 manifest items 与 worker 侧媒体读取配置。"""

        super().__init__()
        self.items = [dict(item) for item in items]
        self.ingress = ingress

    def __len__(self) -> int:
        """返回数据集大小。"""

        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """读取单个 manifest item，并在 worker 内补齐媒体张量。

        这里返回的仍然是“半成品样本”：
        - 文本还是字符串或预分词后的 token
        - 音频是 1D 波形
        - 视频是采样后的原始帧
        真正拼成模型 batch 的工作会在 `gpu_stream.py` 的 preprocessor 中完成。
        """

        item = dict(self.items[idx])
        out = _manifest_item_to_base_output(item, idx)
        # 如果训练/推理前已经提前缓存过 token，这里直接把缓存带上，
        # 避免 DataLoader worker 再调用一次 tokenizer。
        if "_text_input_ids" in item:
            _attach_text_tokens(
                out,
                {
                    "input_ids": item["_text_input_ids"],
                    "attention_mask": item["_text_attention_mask"],
                    **({"token_type_ids": item["_text_token_type_ids"]} if "_text_token_type_ids" in item else {}),
                },
            )

        need_video = (not bool(self.ingress.zero_video)) and str(self.ingress.video_backbone) in {"flow", "videomae", "dual"}
        need_audio = not bool(self.ingress.zero_audio)

        if need_audio:
            audio_path = Path(str(out["audio_path"] or "")).expanduser()
            if not audio_path.exists():
                raise FileNotFoundError(f"Manifest audio_path not found: {audio_path}")
            wav_f32, audio_backend = load_audio(
                audio_path,
                sample_rate=int(self.ingress.sample_rate),
                max_sec=float(self.ingress.max_audio_sec),
                backend_mode=str(self.ingress.audio_backend_mode),
            )
            out["audio"] = wav_f32.to(torch.float32)
            out["_audio_backend"] = str(audio_backend)

        if need_video:
            video_path = Path(str(out["video_path"] or "")).expanduser()
            if not video_path.exists():
                raise FileNotFoundError(f"Manifest video_path not found: {video_path}")
            frames_cpu, video_backend = _load_sampled_video_frames_cpu(
                video_path,
                num_frames=int(self.ingress.num_frames),
                backend=str(self.ingress.video_decode_backend),
            )
            out["video_frames"] = frames_cpu
            out["_video_backend"] = str(video_backend)

        return out


class CachedManifestDataset(Dataset):
    """从主线输入缓存读取样本，尽量复用流式数据集的输出结构。"""

    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        ingress: ManifestIngressConfig,
        cache_dir: Path,
        text_policy: str,
        runtime_profile: RuntimeProfile | None = None,
        keep_in_memory: bool | None = None,
        progress_logger: Optional[ProgressLogger] = None,
        progress_interval: int = 512,
    ):
        """保存 manifest items，并建立它们与缓存索引的对应关系。"""

        super().__init__()
        self.items = [dict(item) for item in items]
        self.ingress = ingress
        self.cache_dir = cache_dir.expanduser()
        self.text_policy = str(text_policy)
        self.meta = load_input_cache_meta(self.cache_dir)
        self.cache_contract = build_input_cache_contract(self.meta)
        self.entries = load_input_cache_index(self.cache_dir)
        self.entries_by_key = index_entries_by_key(self.entries)
        self.cache_keys = [manifest_item_cache_key(item) for item in self.items]
        missing = [key for key in self.cache_keys if key not in self.entries_by_key]
        if missing:
            preview = ", ".join(missing[:3])
            raise RuntimeError(f"Input cache is missing {len(missing)} selected sample(s). Example(s): {preview}")

        self.selected_cache_bytes = count_selected_cache_bytes(entries_by_key=self.entries_by_key, keys=self.cache_keys)
        if keep_in_memory is None:
            sample_count = len(self.cache_keys)
            avg_sample_bytes = int(self.selected_cache_bytes / max(1, sample_count))
            self.in_memory = bool(
                runtime_profile is not None
                and should_keep_dataset_in_memory(
                    sample_count=sample_count,
                    sample_bytes=avg_sample_bytes,
                    profile=runtime_profile,
                )
            )
        else:
            self.in_memory = bool(keep_in_memory)

        self._payload_cache: dict[str, dict[str, Any]] = {}
        self._shard_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._shard_cache_limit = None if self.in_memory else 2
        selected_shards = {
            str(self.entries_by_key[key].get("shard_relpath", "") or "")
            for key in self.cache_keys
            if str(self.entries_by_key[key].get("shard_relpath", "") or "")
        }
        total = len(self.cache_keys)
        total_gib = float(self.selected_cache_bytes) / float(1024**3)
        _emit_progress(
            progress_logger,
            (
                f"input cache residency decided: in_memory={self.in_memory}, samples={total}, "
                f"estimated_size_gib={total_gib:.2f}, selected_shards={len(selected_shards)}"
            ),
        )
        if self.in_memory:
            _emit_progress(
                progress_logger,
                f"preloading input cache into RAM: samples={total}, estimated_size_gib={total_gib:.2f}",
            )
            preload_started = time.perf_counter()
            every = max(1, int(progress_interval))
            for idx, key in enumerate(self.cache_keys, start=1):
                self._payload_cache[key] = self._load_cached_payload(key)
                if idx == total or idx % every == 0:
                    elapsed = time.perf_counter() - preload_started
                    pct = 100.0 * float(idx / max(1, total))
                    _emit_progress(
                        progress_logger,
                        f"input cache preload progress: {idx}/{total} ({pct:.1f}%), elapsed_sec={elapsed:.1f}",
                    )
            _emit_progress(
                progress_logger,
                f"finished preloading input cache: samples={total}, elapsed_sec={time.perf_counter() - preload_started:.1f}",
            )

    def __len__(self) -> int:
        """返回缓存数据集大小。"""

        return len(self.items)

    def _load_cached_payload(self, cache_key: str) -> dict[str, Any]:
        """从磁盘读取一个缓存样本。"""

        entry = self.entries_by_key[str(cache_key)]
        shard_cache: dict[str, dict[str, Any]] | None = self._shard_cache
        if not self.in_memory and "shard_relpath" not in entry:
            shard_cache = None
        payload = load_input_cache_entry_payload(self.cache_dir, entry, shard_cache=shard_cache)
        shard_relpath = str(entry.get("shard_relpath", "") or "")
        if shard_relpath and shard_cache is self._shard_cache:
            self._shard_cache.move_to_end(shard_relpath)
            if self._shard_cache_limit is not None:
                while len(self._shard_cache) > int(self._shard_cache_limit):
                    self._shard_cache.popitem(last=False)
        return payload

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """读取缓存样本，并恢复成主线 preprocessor 认识的字段。"""

        item = dict(self.items[idx])
        out = _manifest_item_to_base_output(item, idx)
        cache_key = self.cache_keys[idx]
        payload = self._payload_cache.get(cache_key)
        if payload is None:
            payload = self._load_cached_payload(cache_key)

        need_video = (not bool(self.ingress.zero_video)) and str(self.ingress.video_backbone) in {"flow", "videomae", "dual"}
        need_audio = not bool(self.ingress.zero_audio)

        if need_audio:
            if "audio" not in payload:
                raise RuntimeError(f"Cached sample missing audio payload: {cache_key}")
            out["audio"] = payload["audio"].to(torch.float32)
            out["_audio_backend"] = str(payload.get("_audio_backend", "input_cache"))

        if need_video:
            if "cached_rgb" in payload:
                # 新版主线缓存直接保存已经按训练口径 crop/resize 完的 RGB clip。
                # 这样 CPU 服务器只做一次重活，GPU 服务器训练时不再重复做同样的
                # 视频预处理，也显著降低单样本缓存体积。
                out["cached_rgb"] = payload["cached_rgb"].to(torch.float16)
            elif "video_frames" in payload:
                # 保留对旧缓存格式的兼容：如果缓存里还是原始采样帧，就继续走老字段。
                out["video_frames"] = payload["video_frames"].to(torch.uint8)
            else:
                raise RuntimeError(f"Cached sample missing video payload: {cache_key}")
            out["_video_backend"] = str(payload.get("_video_backend", "input_cache"))

        if str(self.text_policy) == "full":
            token_bundle = payload.get("text_full", None)
        elif str(self.text_policy) == "mask_emotion_cues":
            token_bundle = payload.get("text_masked", None)
        else:
            token_bundle = None
        if isinstance(token_bundle, dict):
            _attach_text_tokens(out, token_bundle)

        return out


def cache_manifest_text_tokens(
    items: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_text_len: int,
    text_policy: str,
) -> None:
    """Pre-tokenize manifest items once so each batch does not hit the tokenizer again."""

    texts = [
        select_text_for_policy(
            full_text=str(item.get("mn", item.get("text", ""))),
            masked_text=str(item.get("masked_mn", item.get("masked_text", ""))),
            label_en=str(item.get("label_en", item.get("_global_label_en", ""))) or None,
            policy=str(text_policy),
        )
        for item in items
    ]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_text_len),
        return_tensors="pt",
    )
    token_type_ids = enc.get("token_type_ids", None)
    for idx, item in enumerate(items):
        item["_text_input_ids"] = enc["input_ids"][idx].to(torch.long).clone()
        item["_text_attention_mask"] = enc["attention_mask"][idx].to(torch.long).clone()
        if token_type_ids is not None:
            item["_text_token_type_ids"] = token_type_ids[idx].to(torch.long).clone()


def collate_manifest_items(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the raw manifest items so media preprocessing stays in gpu_stream.py."""

    return list(batch)
