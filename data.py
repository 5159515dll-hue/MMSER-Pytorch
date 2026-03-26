"""Manifest-driven dataset ingress helpers for the active gpu_stream mainline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


class StreamingManifestDataset(Dataset):
    """Load audio sidecars and sampled video frames inside DataLoader workers."""

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
        intensity_t = torch.tensor(float("nan"), dtype=torch.float32) if intensity is None else torch.tensor(float(intensity), dtype=torch.float32)
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
