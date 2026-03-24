"""缓存/manifest 数据集读取与 batch 拼接。

`predecode_motion_audio.py` 会把预处理好的多模态样本写成 `.pt` 分片。
这个模块负责：
- 读取 legacy 缓存；
- 在 gpu_stream 路线下把 manifest 条目包装成 worker 可并行 ingest 的 dataset；
- 提供统一的 collate，尽量把重型 ingress 从训练主循环挪到 DataLoader worker。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from predecode_motion_audio import load_audio
from text_policy_utils import select_text_for_policy
from video_motion import _read_video_frames_cv2

_DECORD_CPU_READY = False
_DECORD_CPU = None


def _maybe_import_decord_cpu() -> Any | None:
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
    if total <= 0:
        return [0] * max(1, int(num_frames))
    if total >= int(num_frames):
        return torch.linspace(0, total - 1, int(num_frames)).round().to(torch.long).tolist()
    return list(range(total)) + [total - 1] * (int(num_frames) - total)


def _load_sampled_video_frames_cpu(path: Path, *, num_frames: int, backend: str) -> tuple[torch.Tensor, str]:
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
class CacheConfig:
    """缓存数据集的轻量配置占位符。"""

    sample_rate: int = 24000


@dataclass
class ManifestIngressConfig:
    """gpu_stream 路线下 worker 侧 ingress 配置。"""

    sample_rate: int = 16000
    max_audio_sec: float = 6.0
    audio_backend_mode: str = "auto"
    video_decode_backend: str = "auto"
    num_frames: int = 64
    zero_audio: bool = False
    zero_video: bool = False
    video_backbone: str = "dual"


class CachedMotionAudioDataset(Dataset):
    """从预解码缓存中读取样本。

    设计重点：
    - 支持目录或单个 shard 文件两种输入；
    - 自动跳过损坏 shard，但会把问题打印出来；
    - `__getitem__` 里统一做 dtype 和缺失值整理，减轻训练入口负担。
    """

    def __init__(self, cached_dir: Path):
        """加载缓存目录中的所有 shard。"""
        super().__init__()
        cached_dir = cached_dir.expanduser()
        if cached_dir.is_file():
            paths = [cached_dir]
        else:
            paths = sorted(cached_dir.glob("*.pt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        if not paths:
            raise FileNotFoundError(f"No .pt shards found under {cached_dir}")

        self.samples: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {}
        bad_shards: List[Tuple[Path, str]] = []
        for i, p in enumerate(paths):
            try:
                obj = torch.load(p, map_location="cpu", weights_only=False)
            except Exception as e:
                bad_shards.append((p, f"{type(e).__name__}: {e}"))
                continue
            part = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj
            self.samples.extend(part)
            if i == 0 and isinstance(obj, dict):
                self.config = obj.get("config", {})

        if bad_shards:
            preview = "\n".join([f"  - {p} | {err}" for p, err in bad_shards[:10]])
            print(
                "WARNING: Skipped corrupt cache shard(s) while loading cached dataset:\n"
                f"{preview}\n"
                f"(total corrupt shards skipped: {len(bad_shards)})",
                flush=True,
            )

        if not self.samples:
            raise RuntimeError(
                f"No samples could be loaded from cache under {cached_dir}. "
                "All shards are missing/corrupt, or contain no samples."
            )

    def __len__(self) -> int:
        """返回缓存样本总数。"""

        return len(self.samples)

    def estimate_sample_bytes(self, idx: int = 0) -> int:
        """粗略估算单个样本的 tensor 体积。"""

        if not self.samples:
            return 0
        sample = self.samples[min(max(0, int(idx)), len(self.samples) - 1)]
        total = 0
        for value in sample.values():
            if isinstance(value, torch.Tensor):
                total += int(value.numel() * value.element_size())
        return total

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """取出单个样本，并把字段整理成训练阶段统一约定的格式。"""

        s = self.samples[idx]
        intensity = s.get("intensity", None)
        if isinstance(intensity, torch.Tensor):
            intensity_t = intensity.to(torch.float32).reshape(())
        elif intensity is None:
            intensity_t = torch.tensor(float("nan"), dtype=torch.float32)
        else:
            try:
                intensity_t = torch.tensor(float(intensity), dtype=torch.float32)
            except Exception:
                intensity_t = torch.tensor(float("nan"), dtype=torch.float32)
        item = {
            "prosody": s["prosody"].to(torch.float32),
            "label": s["label"].to(torch.long) if isinstance(s["label"], torch.Tensor) else torch.tensor(int(s["label"]), dtype=torch.long),
            "stem": s.get("stem", str(idx)),
            "text": str(s.get("text", s.get("mn", ""))),
            "mn": str(s.get("mn", "")),
            "masked_text": str(s.get("masked_text", s.get("masked_mn", ""))),
            "masked_mn": str(s.get("masked_mn", "")),
            "speaker_id": str(s.get("speaker_id", "UNKNOWN")),
            "text_cue_flag": bool(s.get("text_cue_flag", False)),
            "cue_severity": str(s.get("cue_severity", "none")),
            "prompt_group_id": str(s.get("prompt_group_id", "")),
            "_global_label_en": str(s.get("_global_label_en", "")),
            "dataset_kind": str(s.get("dataset_kind", "")),
            "intensity": intensity_t,
        }
        if "audio" in s:
            item["audio"] = s["audio"].to(torch.float32)
        if "audio_emb" in s:
            item["audio_emb"] = s["audio_emb"].to(torch.float32)
        if "flow" in s:
            item["flow"] = s["flow"].to(torch.float32)
        if "rgb" in s:
            item["rgb"] = s["rgb"].to(torch.float32)
        if "flow_emb" in s:
            item["flow_emb"] = s["flow_emb"].to(torch.float32)
        if "rgb_emb" in s:
            item["rgb_emb"] = s["rgb_emb"].to(torch.float32)
        if "text_emb" in s:
            item["text_emb"] = s["text_emb"].to(torch.float32)
        if "_text_input_ids" in s:
            item["_text_input_ids"] = s["_text_input_ids"].to(torch.long)
        if "_text_attention_mask" in s:
            item["_text_attention_mask"] = s["_text_attention_mask"].to(torch.long)
        if "_text_token_type_ids" in s:
            item["_text_token_type_ids"] = s["_text_token_type_ids"].to(torch.long)
        return item


class ManifestItemDataset(Dataset):
    """直接从 split manifest 条目构造轻量 streaming dataset。

    这个 dataset 不读取媒体内容，只负责把 manifest 里的元数据规范化成
    训练/推理可消费的样本描述；真正的媒体读取与 CUDA 侧预处理交给
    `gpu_stream.py`。
    """

    def __init__(self, items: list[dict[str, Any]]):
        super().__init__()
        self.items = [dict(item) for item in items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.items[idx])
        label_idx = item.get("label_idx", None)
        if label_idx is None:
            raise RuntimeError(f"Manifest item missing label_idx: {item.get('seq') or item.get('sample_id') or idx}")
        intensity = item.get("intensity", None)
        if intensity is None:
            intensity_t = torch.tensor(float("nan"), dtype=torch.float32)
        else:
            intensity_t = torch.tensor(float(intensity), dtype=torch.float32)
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


class StreamingManifestDataset(Dataset):
    """在 DataLoader worker 中完成媒体 ingress 的 manifest dataset。

    设计重点
    - worker 进程负责读取音频 sidecar、采样视频帧；
    - 主训练循环不再逐样本读文件；
    - 只做 CPU ingress，不在 worker 中触碰 CUDA。
    """

    def __init__(self, items: list[dict[str, Any]], *, ingress: ManifestIngressConfig):
        super().__init__()
        self.items = [dict(item) for item in items]
        self.ingress = ingress

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.items[idx])
        label_idx = item.get("label_idx", None)
        if label_idx is None:
            raise RuntimeError(f"Manifest item missing label_idx: {item.get('seq') or item.get('sample_id') or idx}")

        intensity = item.get("intensity", None)
        if intensity is None:
            intensity_t = torch.tensor(float("nan"), dtype=torch.float32)
        else:
            intensity_t = torch.tensor(float(intensity), dtype=torch.float32)

        out: dict[str, Any] = {
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
        if "_text_input_ids" in item:
            out["_text_input_ids"] = item["_text_input_ids"].to(torch.long)
            out["_text_attention_mask"] = item["_text_attention_mask"].to(torch.long)
            if "_text_token_type_ids" in item:
                out["_text_token_type_ids"] = item["_text_token_type_ids"].to(torch.long)

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


def cache_manifest_text_tokens(
    items: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_text_len: int,
    text_policy: str,
) -> None:
    """一次性为 manifest 条目预分词，避免每个 batch 重复 tokenizer。"""

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


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把若干缓存样本拼成一个 batch。

    这里会做三类事情：
    1. 检查某个模态是否在 batch 内“有的样本有、有的样本没有”；
    2. 对变长音频做 padding，并记录 `audio_lens`；
    3. 为强度回归生成 `intensity_mask`，把 NaN 标注屏蔽掉。
    """

    labels = torch.stack([b["label"] for b in batch], dim=0)

    rgb_list = [b.get("rgb", None) for b in batch]
    has_rgb = any(x is not None for x in rgb_list)
    if has_rgb and any(x is None for x in rgb_list):
        raise RuntimeError("Mixed rgb presence in batch; re-generate cached dataset consistently.")

    flow_list = [b.get("flow", None) for b in batch]
    has_flow = any(x is not None for x in flow_list)
    if has_flow and any(x is None for x in flow_list):
        raise RuntimeError("Mixed flow presence in batch; re-generate cached dataset consistently.")

    audio_list = [b.get("audio", None) for b in batch]
    has_audio = any(x is not None for x in audio_list)
    if has_audio and any(x is None for x in audio_list):
        raise RuntimeError("Mixed audio presence in batch; re-generate cached dataset consistently.")

    if has_audio:
        audios = [b["audio"] for b in batch]
        lens = torch.tensor([a.numel() for a in audios], dtype=torch.long)
        max_len = int(lens.max().item())
        padded = torch.zeros((len(batch), max_len), dtype=torch.float32)
        for i, a in enumerate(audios):
            padded[i, : a.numel()] = a

    audio_emb_list = [b.get("audio_emb", None) for b in batch]
    has_audio_emb = any(x is not None for x in audio_emb_list)
    if has_audio_emb and any(x is None for x in audio_emb_list):
        raise RuntimeError("Mixed audio_emb presence in batch; re-generate cached dataset consistently.")

    flow_emb_list = [b.get("flow_emb", None) for b in batch]
    has_flow_emb = any(x is not None for x in flow_emb_list)
    if has_flow_emb and any(x is None for x in flow_emb_list):
        raise RuntimeError("Mixed flow_emb presence in batch; re-generate cached dataset consistently.")

    rgb_emb_list = [b.get("rgb_emb", None) for b in batch]
    has_rgb_emb = any(x is not None for x in rgb_emb_list)
    if has_rgb_emb and any(x is None for x in rgb_emb_list):
        raise RuntimeError("Mixed rgb_emb presence in batch; re-generate cached dataset consistently.")

    text_emb_list = [b.get("text_emb", None) for b in batch]
    has_text_emb = any(x is not None for x in text_emb_list)
    if has_text_emb and any(x is None for x in text_emb_list):
        raise RuntimeError("Mixed text_emb presence in batch; re-generate cached dataset consistently.")

    prosody = torch.stack([b["prosody"] for b in batch], dim=0)

    text_list = [str(b.get("text", b.get("mn", ""))) for b in batch]
    mn_list = [str(b.get("mn", b.get("text", ""))) for b in batch]
    masked_text_list = [str(b.get("masked_text", b.get("masked_mn", ""))) for b in batch]
    masked_mn_list = [str(b.get("masked_mn", b.get("masked_text", ""))) for b in batch]
    stems = [str(b.get("stem", "")) for b in batch]
    speaker_ids = [str(b.get("speaker_id", "UNKNOWN")) for b in batch]
    cue_severities = [str(b.get("cue_severity", "none")) for b in batch]
    prompt_group_ids = [str(b.get("prompt_group_id", "")) for b in batch]
    global_labels = [str(b.get("_global_label_en", "")) for b in batch]
    dataset_kinds = [str(b.get("dataset_kind", "")) for b in batch]
    text_cue_flags = torch.tensor([1 if bool(b.get("text_cue_flag", False)) else 0 for b in batch], dtype=torch.bool)

    intensity = torch.stack(
        [b.get("intensity", torch.tensor(float("nan"), dtype=torch.float32)) for b in batch],
        dim=0,
    ).to(torch.float32)
    intensity_mask = torch.isfinite(intensity)

    batch_out = {
        "prosody": prosody,
        "labels": labels,
        "text": text_list,
        "mn": mn_list,
        "masked_text": masked_text_list,
        "masked_mn": masked_mn_list,
        "stems": stems,
        "speaker_id": speaker_ids,
        "cue_severity": cue_severities,
        "prompt_group_id": prompt_group_ids,
        "_global_label_en": global_labels,
        "dataset_kind": dataset_kinds,
        "text_cue_flag": text_cue_flags,
        "intensity": intensity,
        "intensity_mask": intensity_mask,
    }
    if has_audio:
        batch_out["audio"] = padded
        batch_out["audio_lens"] = lens
    if has_audio_emb:
        batch_out["audio_emb"] = torch.stack([b["audio_emb"] for b in batch], dim=0)
    if has_flow:
        batch_out["flow"] = torch.stack([b["flow"] for b in batch], dim=0)
    if has_rgb:
        batch_out["rgb"] = torch.stack([b["rgb"] for b in batch], dim=0)
    if has_flow_emb:
        batch_out["flow_emb"] = torch.stack([b["flow_emb"] for b in batch], dim=0)
    if has_rgb_emb:
        batch_out["rgb_emb"] = torch.stack([b["rgb_emb"] for b in batch], dim=0)
    if has_text_emb:
        batch_out["text_emb"] = torch.stack([b["text_emb"] for b in batch], dim=0)
    if all("_text_input_ids" in b for b in batch):
        text_inputs = {
            "input_ids": torch.stack([b["_text_input_ids"] for b in batch], dim=0),
            "attention_mask": torch.stack([b["_text_attention_mask"] for b in batch], dim=0),
        }
        if all("_text_token_type_ids" in b for b in batch):
            text_inputs["token_type_ids"] = torch.stack([b["_text_token_type_ids"] for b in batch], dim=0)
        batch_out["text_inputs"] = text_inputs
    return batch_out


def collate_manifest_items(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """manifest streaming 模式下保持原始 item 列表。"""

    return list(batch)
