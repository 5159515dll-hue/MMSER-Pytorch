import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


_ensure_repo_root_on_path()

import torch
from hf_compat import ensure_transformers_torch_compat

ensure_transformers_torch_compat()
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import EMOTIONS, DEFAULT_CONFIG
from src.data.dataset import MultiModalEmotionDataset
from src.models.encoders import EmotionModel
from src.utils.device import autocast_enabled, get_device


def _infer_data_root(video_path: Path) -> Path:
    # Expect databases/<label>/<id>.mp4
    # e.g. databases/angry/1.mp4 -> parents[1] == databases
    try:
        return video_path.resolve().parents[1]
    except Exception:
        return video_path.parent


def predict(
    model: EmotionModel,
    tokenizer: Optional[PreTrainedTokenizerBase],
    processor: MultiModalEmotionDataset,
    video_path: Path,
    audio_path: Path,
    text: str,
    num_frames: int,
    frame_strategy: str,
    sample_rate: int,
    device: torch.device,
    amp: bool,
    debug: bool = False,
):
    # Keep inference preprocessing identical to training/predecode:
    # - video: decode -> face crop -> resize/normalize
    # - audio: resample -> trim to voiced region
    processor.num_frames = num_frames
    processor.frame_strategy = frame_strategy
    processor.sample_rate = sample_rate
    processor.min_voiced_samples = int(sample_rate * 0.1)

    video = processor._load_video(video_path).to(device, non_blocking=True).unsqueeze(0)
    audio = processor._load_audio(audio_path).to(device, non_blocking=True).unsqueeze(0)
    text_inputs = None
    if tokenizer is not None:
        text_inputs = tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=DEFAULT_CONFIG.data.max_text_length,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    if debug:
        text_shapes = None
        if text_inputs is not None:
            text_shapes = {k: list(v.shape) for k, v in text_inputs.items()}
        print(
            json.dumps(
                {
                    "video_shape": list(video.shape),
                    "audio_shape": list(audio.shape),
                    "text_shapes": text_shapes,
                    "tokenizer_enabled": bool(tokenizer is not None),
                    "num_frames": num_frames,
                    "frame_strategy": frame_strategy,
                    "sample_rate": sample_rate,
                    "face_device": str(processor.face_detector.device),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and autocast_enabled(device),
        ):
            logits, intensity = model(video, audio, text_inputs)
        probs = torch.softmax(logits, dim=1)[0]
        intensity = torch.sigmoid(intensity)[0].item()
    pred_idx = int(probs.argmax().item())
    result = {
        "emotion": EMOTIONS[pred_idx],
        "probability": float(probs[pred_idx].item()),
        "intensity": float(intensity),
    }
    return result


def parse_args():
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description="Run inference on a single sample")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root (e.g. databases/). If omitted, inferred from --video-path.",
    )
    parser.add_argument(
        "--video-path", type=Path, required=False, help="Path to mp4 file"
    )
    parser.add_argument(
        "--audio-path", type=Path, required=False, help="Path to wav file"
    )
    parser.add_argument("--text", type=str, default="", help="Optional text input")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/checkpoints/best.pt")
    )
    parser.add_argument("--num-frames", type=int, default=cfg.data.num_frames)
    parser.add_argument(
        "--frame-strategy",
        choices=["random", "uniform"],
        default=cfg.data.frame_strategy,
    )
    parser.add_argument("--sample-rate", type=int, default=cfg.data.sample_rate)
    parser.add_argument("--tokenizer", default=cfg.model.text_encoder)
    parser.add_argument("--demo", action="store_true", help="Interactive CLI demo")
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable AMP during inference"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print decoded tensor shapes and preprocessing settings",
    )
    return parser.parse_args()


def load_model(checkpoint: Path, cfg=DEFAULT_CONFIG, device=None) -> EmotionModel:
    model = EmotionModel(
        num_classes=len(EMOTIONS),
        video_name=cfg.model.video_encoder,
        audio_name=cfg.model.audio_encoder,
        text_name=cfg.model.text_encoder,
        fusion_hidden=cfg.model.fusion_hidden,
        dropout=cfg.model.dropout,
        freeze_video=cfg.model.freeze_video,
        freeze_audio=cfg.model.freeze_audio,
        freeze_text=cfg.model.freeze_text,
    )
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device or "cpu")
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from {checkpoint}")
    else:
        print(f"Checkpoint {checkpoint} not found; using randomly initialized weights")
    if device:
        model.to(device)
    return model


def interactive_demo(
    model: EmotionModel,
    tokenizer: Optional[PreTrainedTokenizerBase],
    processor: MultiModalEmotionDataset,
    cfg,
    device: torch.device,
    amp: bool,
):
    print("Entering interactive mode. Press Ctrl+C to exit.")
    while True:
        video = Path(input("Video path (.mp4): ").strip())
        audio = Path(input("Audio path (.wav): ").strip())
        text = input("Optional text: ").strip()
        if not video.exists() or not audio.exists():
            print("Invalid paths, try again.")
            continue
        result = predict(
            model,
            tokenizer,
            processor,
            video,
            audio,
            text,
            cfg.data.num_frames,
            cfg.data.frame_strategy,
            cfg.data.sample_rate,
            device,
            amp,
            debug=False,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    device = get_device()
    amp = not args.no_amp
    tokenizer = (
        AutoTokenizer.from_pretrained(args.tokenizer) if args.tokenizer else None
    )
    model = load_model(args.checkpoint, cfg=cfg, device=device)

    if args.video_path is not None:
        data_root = args.data_root or _infer_data_root(args.video_path)
    else:
        data_root = args.data_root or Path(cfg.data.data_root)

    processor = MultiModalEmotionDataset(
        root=str(data_root),
        num_frames=args.num_frames,
        frame_strategy=args.frame_strategy,
        sample_rate=args.sample_rate,
        tokenizer_name="",
        max_text_length=cfg.data.max_text_length,
        text_map_path=None,
        is_train=False,
        scan=False,
    )

    if args.demo:
        interactive_demo(model, tokenizer, processor, cfg, device, amp)
        return

    if args.video_path is None or args.audio_path is None:
        raise ValueError(
            "--video-path and --audio-path are required unless --demo is set"
        )

    result = predict(
        model,
        tokenizer,
        processor,
        args.video_path,
        args.audio_path,
        args.text,
        args.num_frames,
        args.frame_strategy,
        args.sample_rate,
        device,
        amp,
        debug=args.debug,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
