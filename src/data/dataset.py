import json
import os
import random
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchaudio
import soundfile as sf
import cv2
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from facenet_pytorch import MTCNN

try:
    from decord import VideoReader, cpu as decord_cpu, gpu as decord_gpu
except Exception:
    VideoReader = None
    decord_cpu = None
    decord_gpu = None

# Disable OpenCL in OpenCV to avoid OpenCL queue errors on some platforms.
try:
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from src.config import EMOTIONS


ImageNetStats = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def default_video_transform() -> Callable:
    return transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.Normalize(ImageNetStats["mean"], ImageNetStats["std"]),
        ]
    )


class MultiModalEmotionDataset(Dataset):
    def __init__(
        self,
        root: str,
        num_frames: int = 16,
        frame_strategy: str = "random",
        sample_rate: int = 16000,
        tokenizer_name: str = "bert-base-multilingual-cased",
        max_text_length: int = 64,
        text_map_path: Optional[str] = None,
        is_train: bool = True,
        video_transform: Optional[Callable] = None,
        scan: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.num_frames = num_frames
        self.frame_strategy = frame_strategy
        self.sample_rate = sample_rate
        self.max_text_length = max_text_length
        self.is_train = is_train
        self.video_transform = video_transform or default_video_transform()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) if tokenizer_name else None
        self.text_map = self._load_text_map(text_map_path)
        # Allow forcing CPU face detection to avoid driver/arch issues (e.g., non-NVIDIA GPUs)
        force_face_cpu = os.environ.get("FORCE_FACE_CPU", "0") == "1"
        if force_face_cpu:
            face_device = torch.device("cpu")
        else:
            face_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_detector = MTCNN(keep_all=True, device=face_device)
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.decord_ctx = None
        self.decord_ctx_kind = None
        self.decord_enabled = os.environ.get("USE_DECORD", "0") == "1"
        if self.decord_enabled and VideoReader is not None:
            if torch.cuda.is_available():
                try:
                    self.decord_ctx = decord_gpu(0)
                    self.decord_ctx_kind = "gpu"
                except Exception:
                    self.decord_ctx = decord_cpu(0)
                    self.decord_ctx_kind = "cpu"
            else:
                self.decord_ctx = decord_cpu(0)
                self.decord_ctx_kind = "cpu"
        self.voice_energy_threshold = 1e-4  # absolute amplitude threshold for non-silence
        self.min_voiced_samples = int(self.sample_rate * 0.1)  # require at least 100 ms of voice
        self.bad_video_log = self.root / "bad_videos.txt"
        self._bad_video_set = set()

        self.label_to_idx: Dict[str, int] = {label: i for i, label in enumerate(EMOTIONS)}
        self.samples = self._scan() if scan else []
        if scan and len(self.samples) == 0:
            raise RuntimeError(f"No samples found under {self.root}")

    def _load_text_map(self, path: Optional[str]) -> Dict[str, str]:
        if path is None:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _scan(self) -> List[Dict]:
        samples: List[Dict] = []
        for label in self.label_to_idx.keys():
            label_dir = self.root / label
            audio_dir = label_dir / f"{label}_audio"
            if not label_dir.exists():
                continue
            for video_path in label_dir.glob("*.mp4"):
                stem = video_path.stem
                if stem.startswith("._"):
                    # Skip macOS resource fork artifacts like ._1.mp4
                    continue
                audio_path = audio_dir / f"{stem}.wav"
                if not audio_path.exists():
                    continue
                samples.append(
                    {
                        "video": video_path,
                        "audio": audio_path,
                        "label": label,
                        "stem": stem,
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _select_frame_indices(self, t: int) -> List[int]:
        if t >= self.num_frames:
            if self.frame_strategy == "random":
                return sorted(random.sample(range(t), self.num_frames))
            return torch.linspace(0, t - 1, steps=self.num_frames).long().tolist()
        return list(range(t)) + [max(0, t - 1)] * (self.num_frames - t)

    def _apply_video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        sampled = torch.stack([self.video_transform(f) for f in frames])
        return sampled.permute(1, 0, 2, 3)  # (C, T, H, W)

    def _load_video(self, path: Path) -> torch.Tensor:
        decoder_used = "opencv"
        frames = None
        if self.decord_enabled and self.decord_ctx is not None:
            frames = self._load_video_decord(path)
            decoder_used = f"decord-{self.decord_ctx_kind}" if self.decord_ctx_kind else "decord"
        if frames is None or frames.shape[0] == 0:
            frames = self._load_video_cv2(path)
            decoder_used = "opencv"
        if frames is None or frames.shape[0] == 0:
            warnings.warn(f"Failed to decode video {path} via OpenCV; using zero frames instead.")
            self._record_bad_video(path, "opencv_decode_failed")
            frames = torch.zeros(self.num_frames, 3, 112, 112)
        else:
            print(f"[decode] {path} decoder={decoder_used}")
        if frames.shape[0] == 0:
            # Corrupted/empty video: fall back to black frames to keep the loader running.
            self._record_bad_video(path, "empty_frames")
            frames = torch.zeros(self.num_frames, 3, 112, 112)
        indices = self._select_frame_indices(frames.shape[0])
        frames = frames[indices]
        face_crops: List[torch.Tensor] = []
        first_frame_box = None
        for idx, f in enumerate(frames):
            if idx == 0 or first_frame_box is None:
                boxes = self._detect_face_box(f)
                if boxes is None:
                    raise RuntimeError(f"No face detected in {path}")
                first_frame_box = boxes
            x1, y1, x2, y2 = first_frame_box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(f.shape[2], x2)
            y2 = min(f.shape[1], y2)
            crop = f[:, y1:y2, x1:x2]
            if crop.numel() == 0:
                raise RuntimeError(f"Empty face crop in {path}")
            # Normalize crop size so stacking succeeds; downstream transform will center-crop to 112.
            crop = F.interpolate(crop.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False).squeeze(0)
            face_crops.append(crop.cpu())
        face_tensor = torch.stack(face_crops)
        return self._apply_video_transform(face_tensor)

    def _load_video_decord(self, path: Path) -> Optional[torch.Tensor]:
        if VideoReader is None or self.decord_ctx is None:
            return None
        try:
            vr = VideoReader(str(path), ctx=self.decord_ctx)
            # Uniformly sample frames first, to align with our frame selection
            total = len(vr)
            indices = self._select_frame_indices(total)
            batch = vr.get_batch(indices)  # (T, H, W, C), NDArray
            # Convert to torch tensor, permute to (T, C, H, W)
            frames = torch.from_numpy(batch.asnumpy()).permute(0, 3, 1, 2)
            return frames
        except Exception:
            return None

    def _load_video_cv2(self, path: Path) -> Optional[torch.Tensor]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        frames_list: List[torch.Tensor] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_t = torch.from_numpy(frame).permute(2, 0, 1)  # (C, H, W), uint8
            frames_list.append(frame_t)
        cap.release()
        if len(frames_list) == 0:
            return None
        return torch.stack(frames_list)

    def _detect_face_box(self, frame: torch.Tensor) -> Optional[Tuple[float, float, float, float]]:
        # Try OpenCV Haar cascade first (robust for frontal, centered faces)
        np_img = frame.permute(1, 2, 0).cpu().numpy().astype("uint8")  # HWC
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(48, 48))
        if len(faces) > 0:
            # Pick the largest face
            x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
            return [float(x), float(y), float(x + w), float(y + h)]

        # Try MTCNN at native resolution
        img = (frame.float() / 255.0).permute(1, 2, 0).to(self.face_detector.device)
        boxes, _ = self.face_detector.detect(img)
        if boxes is not None and len(boxes) > 0:
            return boxes[0].tolist()

        # Retry MTCNN on upscaled frame to help small faces
        h, w = img.shape[0], img.shape[1]
        max_side = max(h, w)
        if max_side > 0:
            scale = 256.0 / max_side
            if scale > 1.0:
                up = F.interpolate(img.permute(2, 0, 1).unsqueeze(0), scale_factor=scale, mode="bilinear", align_corners=False)
                up_img = up.squeeze(0).permute(1, 2, 0)
                boxes, _ = self.face_detector.detect(up_img)
                if boxes is not None and len(boxes) > 0:
                    boxes = boxes[0] / scale
                    return boxes.tolist()

        # Fallback: assume single centered face; crop a central square covering 80% of the shorter side.
        short_side = min(img.shape[0], img.shape[1]) if max_side > 0 else 0
        if short_side == 0:
            return None
        side = int(short_side * 0.8)
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(img.shape[1], x1 + side)
        y2 = min(img.shape[0], y1 + side)
        return [float(x1), float(y1), float(x2), float(y2)]

    def _record_bad_video(self, path: Path, error) -> None:
        key = str(path)
        if key in self._bad_video_set:
            return
        self._bad_video_set.add(key)
        msg = f"{path}\t{error}\n"
        try:
            with open(self.bad_video_log, "a", encoding="utf-8") as f:
                f.write(msg)
        except Exception:
            pass

    def _load_audio(self, path: Path) -> torch.Tensor:
        try:
            wav, sr = torchaudio.load(str(path))  # (channels, n)
        except Exception:
            # Any failure (missing torchcodec/ffmpeg/etc.) falls back to soundfile
            wav, sr = sf.read(str(path), always_2d=True)
            wav = torch.from_numpy(wav.T).float()  # (channels, n)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.squeeze(0).float()  # (n,)
        voiced = self._trim_to_voiced(wav)
        if voiced.numel() < self.min_voiced_samples:
            raise RuntimeError(f"No voiced segment detected in {path}")
        return voiced

    def _trim_to_voiced(self, wav: torch.Tensor) -> torch.Tensor:
        # Keep only the region that contains non-silent samples.
        mask = wav.abs() > self.voice_energy_threshold
        if not mask.any():
            return torch.empty(0, device=wav.device)
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        start = idx[0].item()
        end = idx[-1].item() + 1
        return wav[start:end]

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        video = self._load_video(sample["video"])
        audio = self._load_audio(sample["audio"])
        text = self.text_map.get(sample["stem"], "")
        label = self.label_to_idx[sample["label"]]
        return {
            "video": video,
            "audio": audio,
            "text": text,
            "label": label,
            "stem": sample["stem"],
        }

    def collate_fn(self, batch: List[Dict]) -> Dict:
        # Keep collate_fn picklable for multiprocessing DataLoader workers (spawn on macOS)
        videos = torch.stack([b["video"] for b in batch])  # (B, C, T, H, W)
        audios = [b["audio"] for b in batch]
        audio_lens = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
        audios = pad_sequence(audios, batch_first=True)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        texts = [b["text"] for b in batch]

        text_inputs = None
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )

        return {
            "video": videos,
            "audio": audios,
            "audio_lens": audio_lens,
            "text_inputs": text_inputs,
            "labels": labels,
        }
