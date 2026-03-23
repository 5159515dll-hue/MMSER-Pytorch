"""缓存数据集读取与 batch 拼接。

`predecode_motion_audio.py` 会把预处理好的多模态样本写成 `.pt` 分片。
这个模块负责把这些分片重新读回来，并在 DataLoader 层面统一成训练可用的 batch。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class CacheConfig:
    """缓存数据集的轻量配置占位符。"""

    sample_rate: int = 24000


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
            "mn": str(s.get("mn", "")),
            "speaker_id": str(s.get("speaker_id", "UNKNOWN")),
            "text_cue_flag": bool(s.get("text_cue_flag", False)),
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

    mn_list = [str(b.get("mn", "")) for b in batch]
    stems = [str(b.get("stem", "")) for b in batch]
    speaker_ids = [str(b.get("speaker_id", "UNKNOWN")) for b in batch]
    text_cue_flags = torch.tensor([1 if bool(b.get("text_cue_flag", False)) else 0 for b in batch], dtype=torch.bool)

    intensity = torch.stack(
        [b.get("intensity", torch.tensor(float("nan"), dtype=torch.float32)) for b in batch],
        dim=0,
    ).to(torch.float32)
    intensity_mask = torch.isfinite(intensity)

    batch_out = {
        "prosody": prosody,
        "labels": labels,
        "mn": mn_list,
        "stems": stems,
        "speaker_id": speaker_ids,
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
