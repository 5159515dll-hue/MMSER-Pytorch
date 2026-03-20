from dataclasses import dataclass, field
from typing import List, Optional


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
    data_root: str = "databases"
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
