from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MotionConfig:
    num_frames: int = 64
    flow_size: int = 112


@dataclass
class RgbConfig:
    num_frames: int = 64
    rgb_size: int = 224


def _read_video_frames_cv2(path: Path) -> List[np.ndarray]:
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
    if total <= 0:
        return [0] * num_frames
    if total >= num_frames:
        # uniform
        return np.linspace(0, total - 1, num_frames).astype(np.int64).tolist()
    # pad last
    return list(range(total)) + [total - 1] * (num_frames - total)


def _detect_face_box_haar(frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
    return int(x), int(y), int(w), int(h)


def _crop_and_resize(frame_bgr: np.ndarray, box: Tuple[int, int, int, int], size: int) -> np.ndarray:
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
    # cv2.data is available at runtime, but some type checkers don't know it.
    haar_root = getattr(getattr(cv2, "data", None), "haarcascades", "")
    if not haar_root:
        # Best-effort fallback; may be empty depending on OpenCV build.
        haar_root = str(Path(cv2.__file__).resolve().parent / "data" / "haarcascades")
    cascade = cascade or cv2.CascadeClassifier(str(Path(haar_root) / "haarcascade_frontalface_default.xml"))
    frames = _read_video_frames_cv2(video_path)
    if not frames:
        # return zeros
        return np.zeros((3, cfg.num_frames - 1, cfg.flow_size, cfg.flow_size), dtype=np.float32)

    idxs = _select_indices(len(frames), cfg.num_frames)
    sampled = [frames[i] for i in idxs]

    first_box = _detect_face_box_haar(sampled[0], cascade)
    if first_box is None:
        # fall back to center crop
        h, w = sampled[0].shape[:2]
        side = int(min(h, w) * 0.8)
        x = (w - side) // 2
        y = (h - side) // 2
        first_box = (x, y, side, side)

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

    # Per-sample normalization to reduce identity/lighting cues: robust scale on magnitude.
    # Keep u/v scale consistent by same factor.
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
    haar_root = getattr(getattr(cv2, "data", None), "haarcascades", "")
    if not haar_root:
        haar_root = str(Path(cv2.__file__).resolve().parent / "data" / "haarcascades")
    cascade = cascade or cv2.CascadeClassifier(str(Path(haar_root) / "haarcascade_frontalface_default.xml"))

    frames = _read_video_frames_cv2(video_path)
    if not frames:
        return np.zeros((cfg.num_frames, 3, cfg.rgb_size, cfg.rgb_size), dtype=np.float32)

    idxs = _select_indices(len(frames), cfg.num_frames)
    sampled = [frames[i] for i in idxs]

    first_box = _detect_face_box_haar(sampled[0], cascade)
    if first_box is None:
        h, w = sampled[0].shape[:2]
        side = int(min(h, w) * 0.8)
        x = (w - side) // 2
        y = (h - side) // 2
        first_box = (x, y, side, side)

    face_imgs = [_crop_and_resize(f, first_box, cfg.rgb_size) for f in sampled]
    rgb_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in face_imgs]
    rgb = np.stack(rgb_frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,3)
    rgb = np.transpose(rgb, (0, 3, 1, 2))  # (T,3,H,W)
    return rgb
