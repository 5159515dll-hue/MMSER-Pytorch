"""从 manifest + 原始媒体直接构造训练/特征缓存样本。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from audio_aug import normalize_wav
from predecode_motion_audio import load_audio
from prosody import ProsodyConfig, extract_prosody_features
from video_motion import (
    MotionConfig,
    RgbConfig,
    compute_face_flow_and_rgb_tensors,
    compute_face_flow_tensor,
    compute_face_rgb_tensor,
)


class ManifestMediaDataset(Dataset):
    """直接从 manifest 条目读取原始媒体并产出统一样本字典。

    这个 dataset 的目标不是取代小数据集上的 raw predecode，而是为大 benchmark
    提供一条“媒体 -> feature cache”的直通路径，避免先落一层超大的 raw 张量缓存。
    """

    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        need_audio: bool,
        need_flow: bool,
        need_rgb: bool,
        sample_rate: int,
        max_audio_sec: float,
        audio_backend_mode: str,
        motion_cfg: MotionConfig,
        rgb_cfg: RgbConfig,
        prosody_cfg: ProsodyConfig,
    ) -> None:
        super().__init__()
        self.items = list(items)
        self.need_audio = bool(need_audio)
        self.need_flow = bool(need_flow)
        self.need_rgb = bool(need_rgb)
        self.sample_rate = int(sample_rate)
        self.max_audio_sec = float(max_audio_sec)
        self.audio_backend_mode = str(audio_backend_mode)
        self.motion_cfg = motion_cfg
        self.rgb_cfg = rgb_cfg
        self.prosody_cfg = prosody_cfg

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.items[idx])
        label_idx = item.get("label_idx", None)
        if label_idx is None:
            raise RuntimeError(f"Manifest item missing label_idx: {item.get('seq')}")

        video_path = Path(str(item.get("video_path") or "")).expanduser()
        if not video_path.exists():
            raise FileNotFoundError(f"Manifest video_path not found: {video_path}")

        audio = None
        prosody = torch.zeros((10,), dtype=torch.float32)
        if self.need_audio:
            audio_path = Path(str(item.get("audio_path") or "")).expanduser()
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Manifest audio_path not found: {audio_path}. "
                    "Run prepare_dataset_media.py first for datasets that ship without wav sidecars."
                )
            wav_f32, _backend = load_audio(
                audio_path,
                sample_rate=self.sample_rate,
                max_sec=self.max_audio_sec,
                backend_mode=self.audio_backend_mode,
            )
            wav_f32 = normalize_wav(wav_f32, target_rms=0.1)
            audio = wav_f32.to(torch.float32)
            prosody = extract_prosody_features(audio, self.prosody_cfg).to(torch.float32)

        flow = None
        rgb = None
        if self.need_flow and self.need_rgb:
            flow_np, rgb_np = compute_face_flow_and_rgb_tensors(video_path, self.motion_cfg, self.rgb_cfg)
            flow = torch.from_numpy(flow_np).to(torch.float32)
            rgb = torch.from_numpy(rgb_np).to(torch.float32)
        elif self.need_flow:
            flow = torch.from_numpy(compute_face_flow_tensor(video_path, self.motion_cfg)).to(torch.float32)
        elif self.need_rgb:
            rgb = torch.from_numpy(compute_face_rgb_tensor(video_path, self.rgb_cfg)).to(torch.float32)

        text = str(item.get("text") or item.get("mn") or "")
        masked_text = str(item.get("masked_text") or item.get("masked_mn") or "")
        intensity = item.get("intensity", None)
        if intensity is None:
            intensity_t = torch.tensor(float("nan"), dtype=torch.float32)
        else:
            intensity_t = torch.tensor(float(intensity), dtype=torch.float32)

        out: dict[str, Any] = {
            "prosody": prosody,
            "label": torch.tensor(int(label_idx), dtype=torch.long),
            "stem": str(item.get("seq") or item.get("sample_id") or idx),
            "text": text,
            "mn": text,
            "masked_text": masked_text,
            "masked_mn": masked_text,
            "speaker_id": str(item.get("speaker_id", "UNKNOWN")),
            "text_cue_flag": bool(item.get("text_cue_flag", False)),
            "cue_severity": str(item.get("cue_severity", "none")),
            "prompt_group_id": str(item.get("prompt_group_id", "")),
            "_global_label_en": str(item.get("label_en", "")),
            "intensity": intensity_t,
            "dataset_kind": str(item.get("dataset_kind", "")),
        }
        if audio is not None:
            out["audio"] = audio
        if flow is not None:
            out["flow"] = flow
        if rgb is not None:
            out["rgb"] = rgb
        return out
