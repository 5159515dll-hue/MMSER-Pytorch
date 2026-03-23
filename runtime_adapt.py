from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import platform
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class RuntimeProfile:
    """当前机器的轻量运行时画像。"""

    platform: str
    cpu_count: int
    available_mem_bytes: Optional[int]
    device_type: str
    device_index: Optional[int]
    gpu_name: Optional[str]
    total_vram_bytes: Optional[int]
    bf16_supported: bool

    def to_jsonable(self) -> dict[str, Any]:
        """转换成便于写入 metrics / config 的字典。"""

        return asdict(self)


def _available_mem_bytes() -> Optional[int]:
    """尽量无依赖地探测可用内存。"""

    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            if page_size > 0 and avail_pages > 0:
                return page_size * avail_pages
        except Exception:
            pass
    return None


def select_device(preference: str = "auto") -> torch.device:
    """按用户偏好选择 torch device。"""

    pref = str(preference or "auto").strip().lower()
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if getattr(torch.backends, "mps", None) is not None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def detect_runtime(preference: str = "auto") -> RuntimeProfile:
    """汇总 CPU / 内存 / GPU 的关键运行时信息。"""

    device = select_device(preference)
    gpu_name = None
    total_vram_bytes = None
    bf16_supported = False
    if device.type == "cuda":
        try:
            idx = int(device.index or 0)
            props = torch.cuda.get_device_properties(idx)
            gpu_name = str(props.name)
            total_vram_bytes = int(props.total_memory)
        except Exception:
            idx = 0
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False
        device_index = idx
    else:
        device_index = None

    return RuntimeProfile(
        platform=platform.platform(),
        cpu_count=max(1, int(os.cpu_count() or 1)),
        available_mem_bytes=_available_mem_bytes(),
        device_type=str(device.type),
        device_index=device_index,
        gpu_name=gpu_name,
        total_vram_bytes=total_vram_bytes,
        bf16_supported=bf16_supported,
    )


def parse_auto_or_int(raw: Any, *, allow_zero: bool = True) -> Optional[int]:
    """把 'auto' / None / 整数字符串解析成 int 或 None。"""

    if raw is None:
        return None
    if isinstance(raw, int):
        if raw < 0 or (raw == 0 and not allow_zero):
            raise ValueError(f"Expected positive integer, got {raw}")
        return int(raw)
    text = str(raw).strip().lower()
    if not text or text == "auto":
        return None
    value = int(text)
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"Expected positive integer, got {value}")
    return value


def resolve_amp_mode(requested: str, profile: RuntimeProfile) -> str:
    """选择最终 AMP 模式。"""

    mode = str(requested or "auto").strip().lower()
    if mode not in {"auto", "off", "fp16", "bf16"}:
        raise ValueError(f"Unsupported amp mode: {requested}")
    if mode == "off":
        return "off"
    if profile.device_type != "cuda":
        return "off"
    if mode == "auto":
        if profile.bf16_supported:
            return "bf16"
        return "fp16"
    if mode == "bf16" and not profile.bf16_supported:
        return "fp16"
    return mode


def resolve_worker_count(
    raw_value: Any,
    *,
    phase: str,
    profile: RuntimeProfile,
    dataset_in_memory: bool = False,
    total_items: Optional[int] = None,
) -> int:
    """根据阶段和机器资源解析 worker 数。"""

    parsed = parse_auto_or_int(raw_value)
    if parsed is not None:
        return int(parsed)

    cpu_count = max(1, int(profile.cpu_count))
    reserve = 1 if cpu_count > 2 else 0
    usable = max(1, cpu_count - reserve)
    if phase == "predecode":
        if total_items is not None and total_items <= 16:
            return min(2, usable)
        return max(1, min(usable, 8))
    if phase == "validate":
        return max(1, min(usable, 8))
    if phase in {"train", "inference", "feature_cache"}:
        if dataset_in_memory:
            return 0
        return max(1, min(usable // 2 if usable > 2 else usable, 4))
    return max(1, min(usable, 4))


def resolve_prefetch_factor(raw_value: Any, *, num_workers: int) -> Optional[int]:
    """DataLoader prefetch_factor 的 auto 解析。"""

    if num_workers <= 0:
        return None
    parsed = parse_auto_or_int(raw_value, allow_zero=False)
    if parsed is not None:
        return int(parsed)
    return 2 if num_workers <= 2 else 4


def resolve_batch_size(
    raw_value: Any,
    *,
    phase: str,
    profile: RuntimeProfile,
    feature_cache: bool,
    video_backbone: str,
    freeze_audio: bool,
    freeze_text: bool,
    freeze_flow: bool,
    freeze_rgb: bool,
) -> int:
    """按设备能力和模型重度给出保守但较快的 batch size。"""

    parsed = parse_auto_or_int(raw_value, allow_zero=False)
    if parsed is not None:
        return int(parsed)

    if profile.device_type != "cuda":
        if phase == "train":
            return 2
        return 4

    vram_gb = float((profile.total_vram_bytes or 0) / (1024**3))
    backbone = str(video_backbone).strip().lower()
    frozen_modalities = sum([bool(freeze_audio), bool(freeze_text), bool(freeze_flow), bool(freeze_rgb)])

    if phase == "feature_cache":
        if vram_gb >= 48:
            return 32
        if vram_gb >= 24:
            return 16
        return 8

    if phase == "inference":
        if feature_cache:
            if vram_gb >= 48:
                return 64
            if vram_gb >= 24:
                return 32
            return 16
        if backbone == "dual":
            if vram_gb >= 48:
                return 8
            if vram_gb >= 24:
                return 4
            return 2
        if backbone == "videomae":
            if vram_gb >= 48:
                return 12
            if vram_gb >= 24:
                return 6
            return 2
        if vram_gb >= 48:
            return 32
        if vram_gb >= 24:
            return 16
        return 8

    if feature_cache:
        if vram_gb >= 48:
            return 64
        if vram_gb >= 24:
            return 32
        return 16

    if backbone == "dual":
        if frozen_modalities >= 3:
            if vram_gb >= 48:
                return 8
            if vram_gb >= 24:
                return 4
            return 2
        if vram_gb >= 48:
            return 4
        if vram_gb >= 24:
            return 2
        return 1

    if backbone == "videomae":
        if vram_gb >= 48:
            return 6
        if vram_gb >= 24:
            return 3
        return 1

    if vram_gb >= 48:
        return 24
    if vram_gb >= 24:
        return 12
    return 6


def resolve_mp_start_method(raw_value: Any) -> str:
    """预处理多进程启动方式的 auto 解析。"""

    text = str(raw_value or "auto").strip().lower()
    if text and text != "auto":
        return text
    if sys_platform_is_linux():
        return "forkserver"
    return "spawn"


def resolve_mp_chunksize(raw_value: Any, *, workers: int, total_items: int) -> int:
    """预处理 map chunksize 的 auto 解析。"""

    parsed = parse_auto_or_int(raw_value, allow_zero=False)
    if parsed is not None:
        return int(parsed)
    if workers <= 1:
        return 1
    return max(1, min(16, int(math.ceil(total_items / max(1, workers * 8)))))


def choose_scratch_dir(raw_value: Any, *, output_dir: Path) -> Path:
    """选择中间文件临时目录。"""

    if raw_value is not None:
        text = str(raw_value).strip()
        if text and text.lower() != "auto":
            return Path(text).expanduser()
    if sys_platform_is_linux():
        return Path("/tmp") / "motion_prosody_cache"
    return output_dir.expanduser()


def estimate_tensor_bytes(obj: Any) -> int:
    """粗略估算一个样本/批次中 tensor 占用的字节数。"""

    if isinstance(obj, torch.Tensor):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return sum(estimate_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(estimate_tensor_bytes(v) for v in obj)
    return 0


def should_keep_dataset_in_memory(
    *,
    sample_count: int,
    sample_bytes: int,
    profile: RuntimeProfile,
    ratio_limit: float = 0.25,
) -> bool:
    """判断当前数据规模是否适合整集常驻内存。"""

    if sample_count <= 0 or sample_bytes <= 0:
        return True
    if profile.available_mem_bytes is None:
        return sample_count <= 10000
    total_estimated = int(sample_count) * int(sample_bytes)
    return total_estimated <= int(profile.available_mem_bytes * float(ratio_limit))


def sys_platform_is_linux() -> bool:
    """判断当前系统是否为 Linux。"""

    return platform.system().strip().lower() == "linux"
