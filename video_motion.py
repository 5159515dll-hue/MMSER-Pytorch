"""视频预处理与面部运动表征提取。

当前主线里，视频模态分成两条路：
1. 光流分支：显式建模相邻帧间的运动。
2. RGB 分支：保留外观信息，供 VideoMAE 这类预训练模型编码。

这两个函数输出的都是“已经对齐好形状”的 numpy 张量，方便后续缓存。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MotionConfig:
    """光流分支的采样与输出尺寸配置。"""

    num_frames: int = 64
    flow_size: int = 112


@dataclass
class RgbConfig:
    """RGB 分支的采样与输出尺寸配置。"""

    num_frames: int = 64
    rgb_size: int = 224


def _resolve_haar_cascade(cascade: Optional[cv2.CascadeClassifier] = None) -> cv2.CascadeClassifier:
    """构造 Haar 人脸检测器。"""

    if cascade is not None:
        return cascade
    haar_root = getattr(getattr(cv2, "data", None), "haarcascades", "")
    if not haar_root:
        haar_root = str(Path(cv2.__file__).resolve().parent / "data" / "haarcascades")
    return cv2.CascadeClassifier(str(Path(haar_root) / "haarcascade_frontalface_default.xml"))


def _fallback_face_box(frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """在人脸检测失败时退回到中心裁剪框。"""

    h, w = frame_bgr.shape[:2]
    side = int(min(h, w) * 0.8)
    x = (w - side) // 2
    y = (h - side) // 2
    return (x, y, side, side)


def _read_video_frames_cv2(path: Path) -> List[np.ndarray]:
    """用 OpenCV 读取整段视频帧。

    返回 BGR 帧列表；如果视频打不开，返回空列表而不是抛错，
    这样上层可以统一决定是跳过样本还是回退成全零张量。
    """

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # BGR
        frames.append(frame)
    cap.release()
    return frames


def _select_indices(total: int, num_frames: int) -> List[int]:
    """生成均匀采样索引，并在短视频场景下用最后一帧补齐。"""

    if total <= 0:
        return [0] * num_frames
    if total >= num_frames:
        # uniform
        return np.linspace(0, total - 1, num_frames).astype(np.int64).tolist()
    # pad last
    return list(range(total)) + [total - 1] * (num_frames - total)


def _detect_face_box_haar(frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier) -> Optional[Tuple[int, int, int, int]]:
    """在单帧里用 Haar 分类器找最大的人脸框。"""

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
    return int(x), int(y), int(w), int(h)


def _crop_and_resize(frame_bgr: np.ndarray, box: Tuple[int, int, int, int], size: int) -> np.ndarray:
    """按人脸框裁剪，并统一缩放到固定尺寸。"""

    x, y, w, h = box
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_bgr.shape[1], x + w)
    y2 = min(frame_bgr.shape[0], y + h)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame_bgr
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return crop


def compute_face_flow_tensor(
    video_path: Path,
    cfg: MotionConfig,
    cascade: Optional[cv2.CascadeClassifier] = None,
) -> np.ndarray:
    """Return optical-flow tensor shaped (3, T-1, H, W) in float32.

    Channels: u, v, magnitude. Flow is computed on grayscale cropped face.
    """
    cascade = _resolve_haar_cascade(cascade)
    frames = _read_video_frames_cv2(video_path)
    if not frames:
        # return zeros
        return np.zeros((3, cfg.num_frames - 1, cfg.flow_size, cfg.flow_size), dtype=np.float32)

    idxs = _select_indices(len(frames), cfg.num_frames)
    sampled = [frames[i] for i in idxs]

    first_box = _detect_face_box_haar(sampled[0], cascade)
    if first_box is None:
        first_box = _fallback_face_box(sampled[0])

    # 这里只在首帧做人脸检测，后续沿用同一个框。
    # 这样虽然不如逐帧跟踪精细，但计算稳定、成本低，也避免人脸框抖动反而污染光流。
    face_imgs = [_crop_and_resize(f, first_box, cfg.flow_size) for f in sampled]
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in face_imgs]

    flows = []
    for t in range(1, len(grays)):
        prev = grays[t - 1]
        nxt = grays[t]
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )  # (H,W,2)
        u = flow[..., 0]
        v = flow[..., 1]
        mag = np.sqrt(u * u + v * v)
        flows.append(np.stack([u, v, mag], axis=0))  # (3,H,W)

    out = np.stack(flows, axis=1).astype(np.float32)  # (3,T-1,H,W)

    # 用每个样本 magnitude 的 95 分位数做鲁棒缩放：
    # - 如果直接按最大值缩放，少量异常光流点会把整体动态范围拉坏；
    # - 用 95 分位数更稳，同时仍保持 u/v/magnitude 三个通道的相对比例一致。
    mag = out[2]
    scale = np.percentile(np.abs(mag), 95)
    if scale and scale > 1e-6:
        out = out / float(scale)
    out = np.clip(out, -5.0, 5.0)
    return out


def compute_face_rgb_tensor(
    video_path: Path,
    cfg: RgbConfig,
    cascade: Optional[cv2.CascadeClassifier] = None,
) -> np.ndarray:
    """Return RGB clip tensor shaped (T, 3, H, W) in float32.

    - Frames are face-cropped using Haar cascade, then resized to (rgb_size, rgb_size).
    - Pixel values are scaled to [0, 1]. Normalization is handled in the model.
    """
    cascade = _resolve_haar_cascade(cascade)

    frames = _read_video_frames_cv2(video_path)
    if not frames:
        return np.zeros((cfg.num_frames, 3, cfg.rgb_size, cfg.rgb_size), dtype=np.float32)

    idxs = _select_indices(len(frames), cfg.num_frames)
    sampled = [frames[i] for i in idxs]

    first_box = _detect_face_box_haar(sampled[0], cascade)
    if first_box is None:
        first_box = _fallback_face_box(sampled[0])

    # 与光流分支保持同样的人脸框来源，减少两路视频表征的空间错位。
    face_imgs = [_crop_and_resize(f, first_box, cfg.rgb_size) for f in sampled]
    rgb_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in face_imgs]
    rgb = np.stack(rgb_frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,3)
    rgb = np.transpose(rgb, (0, 3, 1, 2))  # (T,3,H,W)
    return rgb


def compute_face_flow_and_rgb_tensors(
    video_path: Path,
    motion_cfg: MotionConfig,
    rgb_cfg: RgbConfig,
    cascade: Optional[cv2.CascadeClassifier] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """一次解码同时生成 flow 与 RGB，避免 `both` 模式重复读视频。"""

    cascade = _resolve_haar_cascade(cascade)
    frames = _read_video_frames_cv2(video_path)
    if not frames:
        flow = np.zeros((3, motion_cfg.num_frames - 1, motion_cfg.flow_size, motion_cfg.flow_size), dtype=np.float32)
        rgb = np.zeros((rgb_cfg.num_frames, 3, rgb_cfg.rgb_size, rgb_cfg.rgb_size), dtype=np.float32)
        return flow, rgb

    first_box = _detect_face_box_haar(frames[0], cascade)
    if first_box is None:
        first_box = _fallback_face_box(frames[0])

    flow_idxs = _select_indices(len(frames), motion_cfg.num_frames)
    rgb_idxs = flow_idxs if motion_cfg.num_frames == rgb_cfg.num_frames else _select_indices(len(frames), rgb_cfg.num_frames)

    flow_sampled = [frames[i] for i in flow_idxs]
    rgb_sampled = flow_sampled if flow_idxs == rgb_idxs else [frames[i] for i in rgb_idxs]

    flow_faces = [_crop_and_resize(f, first_box, motion_cfg.flow_size) for f in flow_sampled]
    rgb_faces = [_crop_and_resize(f, first_box, rgb_cfg.rgb_size) for f in rgb_sampled]

    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in flow_faces]
    flows = []
    for t in range(1, len(grays)):
        prev = grays[t - 1]
        nxt = grays[t]
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        u = flow[..., 0]
        v = flow[..., 1]
        mag = np.sqrt(u * u + v * v)
        flows.append(np.stack([u, v, mag], axis=0))

    flow_out = np.stack(flows, axis=1).astype(np.float32)
    mag = flow_out[2]
    scale = np.percentile(np.abs(mag), 95)
    if scale and scale > 1e-6:
        flow_out = flow_out / float(scale)
    flow_out = np.clip(flow_out, -5.0, 5.0)

    rgb_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in rgb_faces]
    rgb_out = np.stack(rgb_frames, axis=0).astype(np.float32) / 255.0
    rgb_out = np.transpose(rgb_out, (0, 3, 1, 2))
    return flow_out, rgb_out
