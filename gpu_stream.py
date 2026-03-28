"""CUDA-first streaming preprocessors for manifest-driven training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import torch
import torch.nn.functional as F

from audio_aug import normalize_wav
from predecode_motion_audio import load_audio
from prosody import ProsodyConfig, extract_prosody_features_gpu
from text_policy_utils import select_text_for_policy
from video_motion import _read_video_frames_cv2


def _select_indices(total: int, num_frames: int) -> list[int]:
    """把任意帧数的视频均匀映射到固定数量的采样位置。"""

    if total <= 0:
        return [0] * num_frames
    if total >= num_frames:
        return torch.linspace(0, total - 1, num_frames).round().to(torch.long).tolist()
    return list(range(total)) + [total - 1] * (num_frames - total)

@dataclass
class GpuStreamConfig:
    """gpu_stream 预处理阶段的静态配置。

    这份配置描述“一个 batch 在进入模型前应该被处理成什么样”，例如：
    - 使用哪个视频分支
    - 音频重采样到多少 Hz
    - 采多少帧
    - 是否把某个模态整体置零
    """

    device: torch.device
    video_backbone: str = "dual"
    sample_rate: int = 24000
    max_audio_sec: float = 6.0
    audio_backend_mode: str = "auto"
    num_frames: int = 64
    flow_size: int = 112
    rgb_size: int = 224
    zero_video: bool = False
    zero_audio: bool = False
    zero_text: bool = False
    prosody_use_pitch: bool = True
    video_decode_backend: str = "auto"
    flow_backend: str = "torch_motion"


class GpuStreamPreprocessor:
    """把 manifest item 批次转成模型可直接消费的 batch。"""

    def __init__(self, cfg: GpuStreamConfig):
        """初始化预处理器并准备后端统计桶。"""

        self.cfg = cfg
        self._decord_ready = False
        self._decord = None
        self._audio_backend_counts: dict[str, int] = {}
        self._video_backend_counts: dict[str, int] = {}
        self._last_prepare_stats: dict[str, float | int] = {}

    @staticmethod
    def _bump_counter(bucket: dict[str, int], key: str) -> None:
        """给某个后端命中计数加一。"""

        name = str(key or "unknown")
        bucket[name] = int(bucket.get(name, 0)) + 1

    def backend_summary(self) -> dict[str, dict[str, int]]:
        """返回当前累计的音频/视频后端命中统计。"""

        return {
            "audio": dict(sorted(self._audio_backend_counts.items())),
            "video": dict(sorted(self._video_backend_counts.items())),
        }

    def consume_prepare_stats(self) -> dict[str, float | int]:
        """取出最近一次 `prepare_batch()` 记录的摘要统计。"""

        stats = dict(self._last_prepare_stats)
        self._last_prepare_stats = {}
        return stats

    def _maybe_import_decord(self) -> Any | None:
        """按需导入 decord，并缓存结果。"""

        if self._decord_ready:
            return self._decord
        self._decord_ready = True
        try:
            import decord  # type: ignore

            decord.bridge.set_bridge("torch")
            self._decord = decord
        except Exception:
            self._decord = None
        return self._decord

    def _read_video_frames(self, video_path: Path) -> torch.Tensor:
        """读取一个视频并采样成固定数量的帧。

        优先尝试 decord；失败时回退到 OpenCV 读取。这样既能在有 decord 的
        环境里获得更高效率，也不会因为 decord 缺失而让主线完全不可用。
        """

        backend = str(self.cfg.video_decode_backend).strip().lower()
        decord = self._maybe_import_decord() if backend in {"auto", "decord"} else None
        if decord is not None:
            ctx = None
            try:
                if self.cfg.device.type == "cuda":
                    ctx = decord.gpu(int(self.cfg.device.index or 0))
                else:
                    ctx = decord.cpu(0)
                vr = decord.VideoReader(str(video_path), ctx=ctx)
                idxs = _select_indices(len(vr), int(self.cfg.num_frames))
                frames = vr.get_batch(idxs)
                if not isinstance(frames, torch.Tensor):
                    frames = torch.as_tensor(frames)
                if frames.device != self.cfg.device:
                    frames = frames.to(self.cfg.device)
                return frames
            except Exception:
                pass

        frames_np = _read_video_frames_cv2(video_path)
        if not frames_np:
            return torch.zeros(
                (int(self.cfg.num_frames), int(self.cfg.rgb_size), int(self.cfg.rgb_size), 3),
                device=self.cfg.device,
                dtype=torch.uint8,
            )
        idxs = _select_indices(len(frames_np), int(self.cfg.num_frames))
        frames = torch.stack([torch.from_numpy(frames_np[i][..., ::-1].copy()) for i in idxs], dim=0)
        return frames.to(self.cfg.device)

    @staticmethod
    def _center_crop_square(frames: torch.Tensor) -> torch.Tensor:
        """把矩形帧裁成中心正方形，便于后续统一 resize。"""

        # frames: (T, H, W, 3) or (T, 3, H, W)
        if frames.ndim != 4:
            raise ValueError(f"Expected 4D frames tensor, got {tuple(frames.shape)}")
        if frames.shape[-1] == 3:
            h = int(frames.shape[1])
            w = int(frames.shape[2])
            side = max(1, min(h, w))
            top = max(0, (h - side) // 2)
            left = max(0, (w - side) // 2)
            return frames[:, top : top + side, left : left + side, :]
        h = int(frames.shape[2])
        w = int(frames.shape[3])
        side = max(1, min(h, w))
        top = max(0, (h - side) // 2)
        left = max(0, (w - side) // 2)
        return frames[:, :, top : top + side, left : left + side]

    def _prepare_rgb(self, frames: torch.Tensor) -> torch.Tensor:
        """把原始帧转换成 RGB 分支需要的 `(T, 3, H, W)` float 张量。"""

        x = frames
        if x.ndim != 4:
            raise ValueError(f"Expected video frames to have 4 dims, got {tuple(x.shape)}")
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self._center_crop_square(x)
        x = x.to(torch.float32) / 255.0
        x = F.interpolate(x, size=(int(self.cfg.rgb_size), int(self.cfg.rgb_size)), mode="bilinear", align_corners=False)
        return x

    def _prepare_motion(self, frames: torch.Tensor) -> torch.Tensor:
        """从 RGB 帧近似构造 motion/flow 表征。

        这里不是传统光流求解器，而是用相邻帧差分 + Sobel 梯度近似运动方向
        与幅值，再把它们堆成 3 通道张量供 flow 分支使用。
        """

        if str(self.cfg.flow_backend) not in {"torch_motion", "legacy"}:
            raise RuntimeError(f"Unsupported flow backend: {self.cfg.flow_backend}")
        rgb = self._prepare_rgb(frames)
        return self._prepare_motion_from_rgb(rgb)

    def _prepare_motion_from_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """从已经预处理好的 RGB clip 构造 motion/flow 表征。

        这个辅助函数专门服务新的 `input cache`：
        - 流式路径会先读原始帧，再调用 `_prepare_rgb()`
        - 缓存路径会直接提供已经处理好的 `cached_rgb`
        两条路径最终都在这里汇合，确保 flow 分支看到的是同一套数学定义。
        """

        # 先把三通道 RGB 压成灰度，减少后续差分和梯度计算开销。
        gray = (
            0.2989 * rgb[:, 0]
            + 0.5870 * rgb[:, 1]
            + 0.1140 * rgb[:, 2]
        )  # (T, H, W)
        gray = F.interpolate(
            gray.unsqueeze(1),
            size=(int(self.cfg.flow_size), int(self.cfg.flow_size)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        if gray.shape[0] <= 1:
            return torch.zeros(
                (3, max(1, int(self.cfg.num_frames) - 1), int(self.cfg.flow_size), int(self.cfg.flow_size)),
                device=self.cfg.device,
                dtype=torch.float32,
            )
        # `dt` 表示相邻帧亮度变化，后续会在它上面再估计空间梯度。
        dt = gray[1:] - gray[:-1]  # (T-1, H, W)
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=dt.device,
            dtype=dt.dtype,
        ).unsqueeze(1)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            device=dt.device,
            dtype=dt.dtype,
        ).unsqueeze(1)
        dt_4d = dt.unsqueeze(1)
        dx = F.conv2d(dt_4d, sobel_x, padding=1).squeeze(1)
        dy = F.conv2d(dt_4d, sobel_y, padding=1).squeeze(1)
        mag = torch.sqrt(dx * dx + dy * dy + 1e-6)
        flow = torch.stack([dx, dy, mag], dim=0).to(torch.float32)
        # 用 95 分位做缩放，比直接用最大值更不容易被极端运动噪声支配。
        scale = torch.quantile(mag.flatten(), q=0.95)
        if bool(torch.isfinite(scale).item()) and float(scale.item()) > 1e-6:
            flow = flow / scale
        return flow.clamp(-5.0, 5.0)

    def _process_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """把单个 manifest item 变成模型前处理前的样本字典。"""

        stem = str(item.get("stem", ""))
        sample: dict[str, Any] = {
            "label": item["label"].to(torch.long),
            "stem": stem,
            "text": str(item.get("text", item.get("mn", ""))),
            "mn": str(item.get("mn", item.get("text", ""))),
            "masked_text": str(item.get("masked_text", item.get("masked_mn", ""))),
            "masked_mn": str(item.get("masked_mn", item.get("masked_text", ""))),
            "speaker_id": str(item.get("speaker_id", "UNKNOWN")),
            "text_cue_flag": bool(item.get("text_cue_flag", False)),
            "cue_severity": str(item.get("cue_severity", "none")),
            "prompt_group_id": str(item.get("prompt_group_id", "")),
            "_global_label_en": str(item.get("_global_label_en", "")),
            "dataset_kind": str(item.get("dataset_kind", "")),
            "intensity": item.get("intensity", torch.tensor(float("nan"), dtype=torch.float32)).to(torch.float32),
        }

        if not bool(self.cfg.zero_audio):
            if "audio" in item and isinstance(item["audio"], torch.Tensor):
                sample["audio"] = item["audio"].to(torch.float32)
                self._bump_counter(self._audio_backend_counts, str(item.get("_audio_backend", "prefetched")))
            else:
                audio_path = Path(str(item.get("audio_path", ""))).expanduser()
                if not audio_path.exists():
                    raise FileNotFoundError(f"Manifest audio_path not found: {audio_path}")
                wav_f32, backend = load_audio(
                    audio_path,
                    sample_rate=int(self.cfg.sample_rate),
                    max_sec=float(self.cfg.max_audio_sec),
                    backend_mode=str(self.cfg.audio_backend_mode),
                )
                sample["audio"] = wav_f32.to(torch.float32)
                self._bump_counter(self._audio_backend_counts, str(backend))

        need_rgb = (not bool(self.cfg.zero_video)) and str(self.cfg.video_backbone) in {"videomae", "dual"}
        need_flow = (not bool(self.cfg.zero_video)) and str(self.cfg.video_backbone) in {"flow", "dual"}
        if need_rgb or need_flow:
            if "cached_rgb" in item and isinstance(item["cached_rgb"], torch.Tensor):
                rgb = item["cached_rgb"].to(self.cfg.device, dtype=torch.float32)
                self._bump_counter(self._video_backend_counts, str(item.get("_video_backend", "input_cache_rgb")))
                if need_rgb:
                    sample["rgb"] = rgb
                if need_flow:
                    sample["flow"] = self._prepare_motion_from_rgb(rgb)
            elif "video_frames" in item and isinstance(item["video_frames"], torch.Tensor):
                frames = item["video_frames"].to(self.cfg.device)
                self._bump_counter(self._video_backend_counts, str(item.get("_video_backend", "prefetched")))
                if need_rgb:
                    sample["rgb"] = self._prepare_rgb(frames)
                if need_flow:
                    sample["flow"] = self._prepare_motion(frames)
            else:
                video_path = Path(str(item.get("video_path", ""))).expanduser()
                if not video_path.exists():
                    raise FileNotFoundError(f"Manifest video_path not found: {video_path}")
                frames = self._read_video_frames(video_path)
                self._bump_counter(self._video_backend_counts, str(self.cfg.video_decode_backend))
                if need_rgb:
                    sample["rgb"] = self._prepare_rgb(frames)
                if need_flow:
                    sample["flow"] = self._prepare_motion(frames)
        return sample

    def prepare_batch(
        self,
        items: list[dict[str, Any]],
        *,
        tokenizer: Any | None,
        text_policy: str,
        max_text_len: int,
    ) -> dict[str, Any]:
        """把一组样本组装成模型可直接消费的 batch。

        这是 gpu_stream 的核心入口。它会同时处理：
        - 标签与元信息
        - 波形 padding 与长度张量
        - prosody 提取
        - flow/rgb 堆叠
        - 文本 token 或缓存 token 的打包
        """

        t0 = time.perf_counter()
        processed = [self._process_item(item) for item in items]

        # 这些字段虽然不直接参与前向，但会进入验证/推理逐样本记录，
        # 因此 batch 级别也要一起带着走。
        labels = torch.stack([sample["label"] for sample in processed], dim=0).to(self.cfg.device)
        stems = [str(sample.get("stem", "")) for sample in processed]
        speaker_ids = [str(sample.get("speaker_id", "UNKNOWN")) for sample in processed]
        cue_severities = [str(sample.get("cue_severity", "none")) for sample in processed]
        prompt_group_ids = [str(sample.get("prompt_group_id", "")) for sample in processed]
        global_labels = [str(sample.get("_global_label_en", "")) for sample in processed]
        dataset_kinds = [str(sample.get("dataset_kind", "")) for sample in processed]
        text_cue_flags = torch.tensor(
            [1 if bool(sample.get("text_cue_flag", False)) else 0 for sample in processed],
            dtype=torch.bool,
            device=self.cfg.device,
        )
        intensity = torch.stack(
            [sample.get("intensity", torch.tensor(float("nan"), dtype=torch.float32)) for sample in processed],
            dim=0,
        ).to(device=self.cfg.device, dtype=torch.float32)
        intensity_mask = torch.isfinite(intensity)

        batch_out: dict[str, Any] = {
            "labels": labels,
            "text": [str(sample.get("text", sample.get("mn", ""))) for sample in processed],
            "mn": [str(sample.get("mn", sample.get("text", ""))) for sample in processed],
            "masked_text": [str(sample.get("masked_text", sample.get("masked_mn", ""))) for sample in processed],
            "masked_mn": [str(sample.get("masked_mn", sample.get("masked_text", ""))) for sample in processed],
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

        if bool(self.cfg.zero_audio):
            padded = torch.zeros((len(processed), 1), device=self.cfg.device, dtype=torch.float32)
            lens = torch.ones((len(processed),), device=self.cfg.device, dtype=torch.long)
            prosody = torch.zeros((len(processed), 10), device=self.cfg.device, dtype=torch.float32)
        else:
            audios = [sample["audio"] for sample in processed]
            lens_cpu = torch.tensor([audio.numel() for audio in audios], dtype=torch.long)
            max_len = int(lens_cpu.max().item())
            padded_cpu = torch.zeros((len(audios), max_len), dtype=torch.float32)
            for idx, audio in enumerate(audios):
                padded_cpu[idx, : audio.numel()] = audio
            # 先统一 padding，再归一化 RMS，最后提 prosody，确保 batch 内口径一致。
            padded = normalize_wav(padded_cpu.to(self.cfg.device), target_rms=0.1)
            lens = lens_cpu.to(self.cfg.device)
            prosody = extract_prosody_features_gpu(
                padded,
                ProsodyConfig(sample_rate=int(self.cfg.sample_rate), use_pitch=bool(self.cfg.prosody_use_pitch)),
                lengths=lens,
            )
        batch_out["audio"] = padded
        batch_out["audio_lens"] = lens
        batch_out["prosody"] = prosody.to(torch.float32)

        if str(self.cfg.video_backbone) in {"flow", "dual"}:
            if bool(self.cfg.zero_video):
                flow = torch.zeros(
                    (len(processed), 3, max(1, int(self.cfg.num_frames) - 1), int(self.cfg.flow_size), int(self.cfg.flow_size)),
                    device=self.cfg.device,
                    dtype=torch.float32,
                )
            else:
                flow = torch.stack([sample["flow"] for sample in processed], dim=0).to(self.cfg.device)
            batch_out["flow"] = flow
        if str(self.cfg.video_backbone) in {"videomae", "dual"}:
            if bool(self.cfg.zero_video):
                rgb = torch.zeros(
                    (len(processed), int(self.cfg.num_frames), 3, int(self.cfg.rgb_size), int(self.cfg.rgb_size)),
                    device=self.cfg.device,
                    dtype=torch.float32,
                )
            else:
                rgb = torch.stack([sample["rgb"] for sample in processed], dim=0).to(self.cfg.device)
            batch_out["rgb"] = rgb

        if (not bool(self.cfg.zero_text)) and tokenizer is not None:
            if all("_text_input_ids" in item and "_text_attention_mask" in item for item in items):
                # 训练/推理前如果已经缓存过 token，就直接堆叠缓存。
                text_inputs = {
                    "input_ids": torch.stack([item["_text_input_ids"].to(torch.long) for item in items], dim=0).to(self.cfg.device),
                    "attention_mask": torch.stack([item["_text_attention_mask"].to(torch.long) for item in items], dim=0).to(self.cfg.device),
                }
                if all("_text_token_type_ids" in item for item in items):
                    text_inputs["token_type_ids"] = torch.stack(
                        [item["_text_token_type_ids"].to(torch.long) for item in items],
                        dim=0,
                    ).to(self.cfg.device)
                batch_out["text_inputs"] = text_inputs
            else:
                # 否则按当前 text policy 动态选择 full/masked/drop 文本，再现场分词。
                texts = [
                    select_text_for_policy(
                        full_text=str(sample.get("mn", "")),
                        masked_text=str(sample.get("masked_mn", "")),
                        label_en=str(sample.get("_global_label_en", "")) or None,
                        policy=str(text_policy),
                    )
                    for sample in processed
                ]
                enc = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=int(max_text_len),
                    return_tensors="pt",
                )
                batch_out["text_inputs"] = {k: v.to(self.cfg.device) for k, v in enc.items()}

        # 这些统计不进入模型，只用于日志解释“当前 batch 的准备阶段花了多久”。
        self._last_prepare_stats = {
            "prepare_batch_sec": float(time.perf_counter() - t0),
            "batch_items": int(len(items)),
            "audio_backend_kinds": int(len(self._audio_backend_counts)),
            "video_backend_kinds": int(len(self._video_backend_counts)),
        }

        return batch_out
