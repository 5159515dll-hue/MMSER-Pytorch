"""Frozen-backbone embedding cache helpers for exploratory mainline runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from input_cache import manifest_item_cache_key

EMBEDDING_CACHE_PROTOCOL_VERSION = "mainline_embedding_cache_v1"
EMBEDDING_CACHE_INDEX_VERSION = "mainline_embedding_cache_index_v1"
EMBEDDING_CACHE_STORAGE_PER_SAMPLE = "per_sample_pt_v1"


def embedding_cache_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "cache_meta.json"


def embedding_cache_index_path(cache_dir: Path) -> Path:
    return cache_dir / "index.jsonl"


def embedding_sample_relpath_for_key(cache_key: str) -> Path:
    digest = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()
    return Path("samples") / digest[:2] / f"{digest}.pt"


def load_embedding_cache_meta(cache_dir: Path) -> dict[str, Any]:
    path = embedding_cache_meta_path(cache_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Embedding cache metadata not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Embedding cache metadata must be a JSON object: {path}")
    return data


def load_embedding_cache_index(cache_dir: Path) -> list[dict[str, Any]]:
    path = embedding_cache_index_path(cache_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Embedding cache index not found: {path}")
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise RuntimeError(f"Embedding cache index row must be an object: {path}:{line_no}")
            entries.append(row)
    return entries


def index_embedding_entries_by_key(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = str(entry.get("cache_key", "") or "")
        if not key:
            raise RuntimeError("Embedding cache index entry missing cache_key")
        if key in mapping:
            raise RuntimeError(f"Duplicate cache_key in embedding cache index: {key}")
        mapping[key] = dict(entry)
    return mapping


def build_embedding_cache_contract(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": str(meta.get("protocol_version", EMBEDDING_CACHE_PROTOCOL_VERSION)),
        "manifest_sha256": str(meta.get("manifest_sha256", "")),
        "dataset_kind": str(meta.get("dataset_kind", "")),
        "text_model": str(meta.get("text_model", "")),
        "audio_model": str(meta.get("audio_model", "")),
        "audio_model_revision": str(meta.get("audio_model_revision", "")),
        "video_model": str(meta.get("video_model", "")),
        "max_text_len": int(meta.get("max_text_len", 0) or 0),
        "sample_rate": int(meta.get("sample_rate", 0) or 0),
        "max_audio_sec": float(meta.get("max_audio_sec", 0.0) or 0.0),
        "num_frames": int(meta.get("num_frames", 0) or 0),
        "rgb_size": int(meta.get("rgb_size", 0) or 0),
        "text_pooling": str(meta.get("text_pooling", "")),
        "audio_pooling": str(meta.get("audio_pooling", "")),
        "rgb_pooling": str(meta.get("rgb_pooling", "")),
        "pooling_version": str(meta.get("pooling_version", "")),
        "embedding_dtype": str(meta.get("embedding_dtype", "")),
        "text_dim": int(meta.get("text_dim", 0) or 0),
        "audio_dim": int(meta.get("audio_dim", 0) or 0),
        "rgb_dim": int(meta.get("rgb_dim", 0) or 0),
        "has_text_emb": bool(meta.get("has_text_emb", False)),
        "has_audio_emb": bool(meta.get("has_audio_emb", False)),
        "has_rgb_emb": bool(meta.get("has_rgb_emb", False)),
        "freeze_text_required": bool(meta.get("freeze_text_required", False)),
        "freeze_audio_required": bool(meta.get("freeze_audio_required", False)),
        "freeze_rgb_required": bool(meta.get("freeze_rgb_required", False)),
        "audio_aug_allowed": bool(meta.get("audio_aug_allowed", True)),
        "storage_format": str(meta.get("storage_format", "")),
        "index_version": str(meta.get("index_version", "")),
    }


def validate_embedding_cache_runtime_allowed(
    *,
    freeze_text: bool,
    freeze_audio: bool,
    freeze_rgb: bool,
    audio_aug: bool,
) -> list[str]:
    reasons: list[str] = []
    if not bool(freeze_text):
        reasons.append("freeze_text_required")
    if not bool(freeze_audio):
        reasons.append("freeze_audio_required")
    if not bool(freeze_rgb):
        reasons.append("freeze_rgb_required")
    if bool(audio_aug):
        reasons.append("audio_aug_not_allowed")
    return reasons


def validate_embedding_cache_contract(
    contract: dict[str, Any],
    *,
    manifest_sha256: str,
    dataset_kind: str,
    text_model: str,
    audio_model: str,
    audio_model_revision: str | None,
    video_model: str,
    max_text_len: int,
    sample_rate: int,
    max_audio_sec: float,
    num_frames: int,
    rgb_size: int,
    text_dim: int,
    audio_dim: int,
    rgb_dim: int,
    need_text: bool,
    need_audio: bool,
    need_rgb: bool,
) -> list[str]:
    reasons: list[str] = []
    expected_revision = str(audio_model_revision or "")
    checks: list[tuple[str, Any, Any]] = [
        ("protocol_version", EMBEDDING_CACHE_PROTOCOL_VERSION, contract.get("protocol_version")),
        ("manifest_sha256", str(manifest_sha256), contract.get("manifest_sha256")),
        ("dataset_kind", str(dataset_kind), contract.get("dataset_kind")),
        ("text_model", str(text_model), contract.get("text_model")),
        ("audio_model", str(audio_model), contract.get("audio_model")),
        ("audio_model_revision", expected_revision, contract.get("audio_model_revision")),
        ("video_model", str(video_model), contract.get("video_model")),
        ("max_text_len", int(max_text_len), int(contract.get("max_text_len", 0) or 0)),
        ("sample_rate", int(sample_rate), int(contract.get("sample_rate", 0) or 0)),
        ("num_frames", int(num_frames), int(contract.get("num_frames", 0) or 0)),
        ("rgb_size", int(rgb_size), int(contract.get("rgb_size", 0) or 0)),
        ("text_pooling", "cls", contract.get("text_pooling")),
        ("audio_pooling", "mean_std_masked_v1", contract.get("audio_pooling")),
        ("rgb_pooling", "videomae_pooler_or_cls_v1", contract.get("rgb_pooling")),
        ("pooling_version", "mainline_embedding_pooling_v1", contract.get("pooling_version")),
        ("embedding_dtype", "float32", contract.get("embedding_dtype")),
        ("text_dim", int(text_dim), int(contract.get("text_dim", 0) or 0)),
        ("audio_dim", int(audio_dim), int(contract.get("audio_dim", 0) or 0)),
        ("rgb_dim", int(rgb_dim), int(contract.get("rgb_dim", 0) or 0)),
        ("storage_format", EMBEDDING_CACHE_STORAGE_PER_SAMPLE, contract.get("storage_format")),
        ("index_version", EMBEDDING_CACHE_INDEX_VERSION, contract.get("index_version")),
    ]
    for field, expected, actual in checks:
        if expected != actual:
            reasons.append(f"{field}_mismatch")
    if abs(float(contract.get("max_audio_sec", 0.0) or 0.0) - float(max_audio_sec)) > 1.0e-9:
        reasons.append("max_audio_sec_mismatch")
    if bool(need_text) and not bool(contract.get("has_text_emb", False)):
        reasons.append("missing_text_emb")
    if bool(need_audio) and not bool(contract.get("has_audio_emb", False)):
        reasons.append("missing_audio_emb")
    if bool(need_rgb) and not bool(contract.get("has_rgb_emb", False)):
        reasons.append("missing_rgb_emb")
    if not bool(contract.get("freeze_text_required", False)):
        reasons.append("freeze_text_required_contract_missing")
    if not bool(contract.get("freeze_audio_required", False)):
        reasons.append("freeze_audio_required_contract_missing")
    if not bool(contract.get("freeze_rgb_required", False)):
        reasons.append("freeze_rgb_required_contract_missing")
    if bool(contract.get("audio_aug_allowed", True)):
        reasons.append("audio_aug_allowed_mismatch")
    return reasons


def save_embedding_cache_meta(cache_dir: Path, meta: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache_meta_path(cache_dir).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def save_embedding_cache_index(cache_dir: Path, entries: list[dict[str, Any]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with embedding_cache_index_path(cache_dir).open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_embedding_payload(cache_dir: Path, cache_key: str, payload: dict[str, torch.Tensor]) -> dict[str, Any]:
    relpath = embedding_sample_relpath_for_key(cache_key)
    path = cache_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "cache_key": str(cache_key),
        "relpath": str(relpath),
        "sample_bytes": int(path.stat().st_size),
        "has_text_emb": "text_emb" in payload,
        "has_audio_emb": "audio_emb" in payload,
        "has_rgb_emb": "rgb_emb" in payload,
    }


class EmbeddingCacheReader:
    def __init__(self, cache_dir: Path, *, need_text: bool = True, need_audio: bool = True, need_rgb: bool = True):
        self.cache_dir = cache_dir.expanduser()
        self.meta = load_embedding_cache_meta(self.cache_dir)
        self.contract = build_embedding_cache_contract(self.meta)
        self.entries = load_embedding_cache_index(self.cache_dir)
        self.entries_by_key = index_embedding_entries_by_key(self.entries)
        self.need_text = bool(need_text)
        self.need_audio = bool(need_audio)
        self.need_rgb = bool(need_rgb)

    def _load_payload(self, item: dict[str, Any]) -> dict[str, torch.Tensor]:
        key = manifest_item_cache_key(item)
        entry = self.entries_by_key.get(key)
        if entry is None:
            raise RuntimeError(f"Embedding cache missing selected sample: {key}")
        relpath = str(entry.get("relpath", "") or "")
        if not relpath:
            raise RuntimeError(f"Embedding cache entry missing relpath: {key}")
        payload = torch.load(self.cache_dir / relpath, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Embedding cache payload must be a dict: {key}")
        return payload

    def load_batch(self, items: list[dict[str, Any]], *, device: torch.device) -> dict[str, torch.Tensor]:
        payloads = [self._load_payload(item) for item in items]
        out: dict[str, torch.Tensor] = {}
        for field, need in (("text_emb", self.need_text), ("audio_emb", self.need_audio), ("rgb_emb", self.need_rgb)):
            if not bool(need):
                continue
            values: list[torch.Tensor] = []
            for payload in payloads:
                value = payload.get(field)
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError(f"Embedding cache payload missing {field}")
                values.append(value.to(device=device, dtype=torch.float32))
            out[field] = torch.stack(values, dim=0).detach()
        return out
