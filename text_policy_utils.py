"""Shared text-policy and prompt-group helpers.

This module centralizes two pieces of logic that need to stay aligned across
manifest building, feature-cache construction, training, inference, and
scientific audits:

1. How text is masked/dropped under different evaluation policies.
2. How prompt groups are derived so grouped CV can prevent prompt leakage.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


TEXT_POLICIES = ("full", "mask_emotion_cues", "drop")
MASK_TOKEN = "[EMOTION]"

EMOTION_WORDS_EN = [
    "happy",
    "angry",
    "sad",
    "fear",
    "disgust",
    "disgusted",
    "surprise",
    "neutral",
]
EMOTION_WORDS_CN = [
    "高兴",
    "开心",
    "快乐",
    "愤怒",
    "生气",
    "悲伤",
    "伤心",
    "恐惧",
    "害怕",
    "厌恶",
    "惊讶",
    "中性",
]
EMOTION_DESC_CN = [
    "很开心",
    "非常开心",
    "特别开心",
    "很愤怒",
    "非常愤怒",
    "特别愤怒",
    "很生气",
    "非常生气",
    "特别生气",
    "很悲伤",
    "非常悲伤",
    "很害怕",
    "非常害怕",
    "很恐惧",
    "非常恐惧",
    "很惊讶",
    "非常惊讶",
    "很厌恶",
    "非常厌恶",
]


def resolve_text_policy(policy: str | None) -> str:
    """Normalize and validate text policy names."""

    value = str(policy or "full").strip().lower()
    if value not in TEXT_POLICIES:
        raise RuntimeError(f"Unsupported text policy: {policy}")
    return value


def derive_cue_severity(cue_details: dict[str, Any] | None) -> str:
    """Collapse detailed cue flags into one severity bucket."""

    details = cue_details or {}
    if bool(details.get("label_in_text", False)):
        return "label_in_text"
    if bool(details.get("emotion_desc_cn", False)):
        return "emotion_phrase"
    if bool(details.get("emotion_words_en", False) or details.get("emotion_words_cn", False)):
        return "emotion_word"
    return "none"


def mask_emotion_cues(
    text: str,
    *,
    label_raw: str | None = None,
    label_en: str | None = None,
    mask_token: str = MASK_TOKEN,
) -> str:
    """Mask explicit emotion cues from text.

    The masking is intentionally simple and deterministic. It is not meant to
    fully solve semantic leakage; it removes direct lexical cues so the
    remaining text dependence can be measured more honestly.
    """

    raw = str(text or "")
    if not raw:
        return ""

    out = raw
    cn_patterns = list(EMOTION_DESC_CN) + list(EMOTION_WORDS_CN)
    extra_cn = str(label_raw or "").strip()
    if extra_cn:
        cn_patterns.append(extra_cn)
    for token in sorted({x for x in cn_patterns if x}, key=len, reverse=True):
        out = out.replace(token, mask_token)

    en_patterns = list(EMOTION_WORDS_EN)
    extra_en = str(label_en or "").strip().lower()
    if extra_en:
        en_patterns.append(extra_en)
    for token in sorted({x for x in en_patterns if x}, key=len, reverse=True):
        out = re.sub(re.escape(token), mask_token, out, flags=re.IGNORECASE)

    out = re.sub(r"(?:\s*" + re.escape(mask_token) + r"\s*){2,}", f" {mask_token} ", out)
    return re.sub(r"\s+", " ", out).strip()


def normalize_text_for_grouping(
    text: str,
    *,
    lowercase: bool = True,
    strip_punct: bool = True,
) -> str:
    """Normalize text for prompt grouping and duplicate detection."""

    value = str(text or "").strip()
    if lowercase:
        value = value.lower()
    if strip_punct:
        value = re.sub(r"[^\w\s\u4e00-\u9fff\u0400-\u04ff\u1800-\u18af]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def build_prompt_group_text(
    mn_text: str,
    zh_text: str,
    *,
    label_raw: str | None = None,
    label_en: str | None = None,
) -> str:
    """Build a cue-masked canonical text used for prompt grouping."""

    masked_mn = mask_emotion_cues(mn_text, label_raw=label_raw, label_en=label_en)
    masked_zh = mask_emotion_cues(zh_text, label_raw=label_raw, label_en=label_en)
    chunks = [normalize_text_for_grouping(x) for x in (masked_mn, masked_zh) if str(x or "").strip()]
    return " || ".join(chunks)


def build_prompt_group_id(
    mn_text: str,
    zh_text: str,
    *,
    label_raw: str | None = None,
    label_en: str | None = None,
) -> str:
    """Create a stable prompt group id from masked normalized text."""

    canonical = build_prompt_group_text(mn_text, zh_text, label_raw=label_raw, label_en=label_en)
    if not canonical:
        canonical = "__empty_prompt__"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def select_text_for_policy(
    *,
    full_text: str,
    masked_text: str | None = None,
    label_raw: str | None = None,
    label_en: str | None = None,
    policy: str | None = "full",
) -> str:
    """Select the effective text content under the requested text policy."""

    resolved = resolve_text_policy(policy)
    if resolved == "drop":
        return ""
    if resolved == "full":
        return str(full_text or "")
    if masked_text is not None and str(masked_text).strip():
        return str(masked_text)
    return mask_emotion_cues(str(full_text or ""), label_raw=label_raw, label_en=label_en)
