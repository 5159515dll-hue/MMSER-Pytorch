from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class CacheConfig:
    sample_rate: int = 24000


class CachedMotionAudioDataset(Dataset):
    def __init__(self, cached_dir: Path):
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
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
        return item


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    prosody = torch.stack([b["prosody"] for b in batch], dim=0)

    mn_list = [str(b.get("mn", "")) for b in batch]

    intensity = torch.stack([b.get("intensity", torch.tensor(float("nan"), dtype=torch.float32)) for b in batch], dim=0).to(torch.float32)
    intensity_mask = torch.isfinite(intensity)

    batch_out = {
        "prosody": prosody,
        "labels": labels,
        "mn": mn_list,
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
    return batch_out
