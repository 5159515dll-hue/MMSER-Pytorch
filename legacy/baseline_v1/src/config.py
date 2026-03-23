import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


_REPO_ROOT = _ensure_repo_root_on_path()

from path_utils import default_databases_dir


EMOTIONS: List[str] = [
    "angry",
    "disgusted",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


@dataclass
class DataConfig:
    data_root: str = field(default_factory=lambda: str(default_databases_dir(_REPO_ROOT)))
    num_frames: int = 16
    frame_strategy: str = "random"  # random | uniform
    sample_rate: int = 16000
    max_text_length: int = 64
    train_split: float = 0.8
    text_map: Optional[str] = None  # path to stem->text JSON


@dataclass
class ModelConfig:
    video_encoder: str = "r2plus1d_18"
    audio_encoder: str = "wav2vec2_base"
    text_encoder: str = "bert-base-multilingual-cased"
    freeze_video: bool = False
    freeze_audio: bool = True
    freeze_text: bool = True
    fusion_hidden: int = 512
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 2
    num_workers: int = 2
    epochs: int = 5
    lr: float = 3e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    intensity_loss_weight: float = 0.3
    amp: bool = True
    log_every: int = 10
    output_dir: str = "outputs"
    pre_decode: bool = True
    pre_decode_device: str = "auto"  # auto | cpu | cuda


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


DEFAULT_CONFIG = Config()
