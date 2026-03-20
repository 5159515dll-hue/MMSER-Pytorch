import argparse
import warnings
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from src.config import DEFAULT_CONFIG, EMOTIONS
from src.data.dataset import MultiModalEmotionDataset

# Silence noisy deprecation warnings from torchvision/torchaudio backends
warnings.filterwarnings(
    "ignore",
    message=r".*video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torchvision\.io.*",
)


def _single_item_collate(batch):
    return batch[0]


def _tokenize_once(tokenizer, text: str, max_len: int):
    if tokenizer is None:
        return None
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in encoded.items()}


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("CUDA requested but unavailable; falling back to CPU.")
    return torch.device("cpu")


def default_output_path(root: Path, cfg) -> Path:
    name = root.name or "dataset"
    tag = f"{name}_f{cfg.data.num_frames}_{cfg.data.frame_strategy}_sr{cfg.data.sample_rate}"
    out_dir = Path("outputs") / "predecoded"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.pt"
    if path.exists():
        idx = 1
        while (out_dir / f"{tag}_{idx}.pt").exists():
            idx += 1
        path = out_dir / f"{tag}_{idx}.pt"
    return path


def parse_args():
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description="Pre-decode dataset and save to disk")
    parser.add_argument("--data-root", default=cfg.data.data_root)
    parser.add_argument("--xlsx", default="databases/视频数据集对应文档.xlsx", help="Excel with columns 序号/蒙文/中文/情感类别")
    parser.add_argument("--output", default=None, help="Output .pt path; default auto under outputs/predecoded")
    parser.add_argument("--text-map", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=cfg.data.num_frames)
    parser.add_argument("--frame-strategy", choices=["random", "uniform"], default=cfg.data.frame_strategy)
    parser.add_argument("--sample-rate", type=int, default=cfg.data.sample_rate)
    parser.add_argument("--max-text-length", type=int, default=cfg.data.max_text_length)
    parser.add_argument("--tokenizer", default=cfg.model.text_encoder)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to place cached tensors (kept on CPU to save memory)")
    parser.add_argument("--num-workers", type=int, default=cfg.train.num_workers, help="Reserved for future parallel decode; currently unused")
    parser.add_argument("--shard-size", type=int, default=0, help="If >0, write shards of this many samples to reduce RAM usage")
    parser.add_argument("--merge-after", action="store_true", help="After sharded save, merge shards into a single file (needs enough RAM)")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    xlsx_path = Path(args.xlsx).expanduser()
    if not xlsx_path.exists():
        raise FileNotFoundError(f"XLSX not found: {xlsx_path}")

    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "hf"))
    print(f"Using TORCH_HOME={torch_home}")
    print(f"Using HF_HOME={hf_home}")

    cfg = DEFAULT_CONFIG
    cfg.data.data_root = str(data_root)
    cfg.data.num_frames = args.num_frames
    cfg.data.frame_strategy = args.frame_strategy
    cfg.data.sample_rate = args.sample_rate
    cfg.data.max_text_length = args.max_text_length
    cfg.data.text_map = args.text_map
    cfg.model.text_encoder = args.tokenizer

    # Always cache on CPU to avoid exhausting GPU memory when saving tensors to disk.
    if args.device == "cuda":
        if torch.cuda.is_available():
            print("CUDA requested, but caching stays on CPU to avoid GPU OOM during predecode.")
        else:
            print("CUDA requested but unavailable; using CPU.")
    device = torch.device("cpu")

    # Build a helper dataset instance for decoding utilities (video/audio transforms, tokenizer)
    dataset = MultiModalEmotionDataset(
        root=cfg.data.data_root,
        num_frames=cfg.data.num_frames,
        frame_strategy=cfg.data.frame_strategy,
        sample_rate=cfg.data.sample_rate,
        tokenizer_name=cfg.model.text_encoder,
        max_text_length=cfg.data.max_text_length,
        text_map_path=cfg.data.text_map,
    )
    face_dev = dataset.face_detector.device
    print(f"Video face detection device: {face_dev}")

    zh2en = {
        "愤怒": "angry",
        "厌恶": "disgusted",
        "恐惧": "fear",
        "开心": "happy",
        "高兴": "happy",
        "快乐": "happy",
        "中性": "neutral",
        "悲伤": "sad",
        "惊讶": "surprise",
    }

    df = pd.read_excel(xlsx_path)
    rows = []
    skipped_no_seq = []
    for _, row in df.iterrows():
        seq = str(row.get("序号") or row.iloc[0]).strip()
        if not seq:
            skipped_no_seq.append(row.name)
            continue
        text_content: Optional[str] = None
        if "蒙文" in row:
            text_content = str(row["蒙文"]) if not pd.isna(row["蒙文"]) else None
        if (not text_content or text_content == "nan") and "中文" in row:
            text_content = str(row["中文"]) if not pd.isna(row["中文"]) else None
        label_raw = str(row.get("情感类别", "")).strip()
        label_name = zh2en.get(label_raw, label_raw.lower())
        rows.append({"seq": seq, "text": text_content or "", "label_name": label_name})

    total = len(rows)
    print(f"Found {len(df)} rows in {xlsx_path}; usable entries with 序号: {total}; empty 序号 rows: {len(skipped_no_seq)}")

    cached: List[Dict] = []
    shard_paths: List[Path] = []
    missing_entries: List[str] = []
    shard_id = 1
    output_base = Path(args.output) if args.output else default_output_path(data_root, cfg)
    shard_dir = output_base.parent / output_base.stem if args.shard_size > 0 else output_base.parent
    for idx, item in enumerate(rows, 1):
        seq = item["seq"]
        label_name = item["label_name"]
        text_content = item["text"]
        vid_matches = sorted(data_root.rglob(f"{seq}.mp4"))
        wav_matches = sorted(data_root.rglob(f"{seq}.wav"))
        video_path = vid_matches[0] if len(vid_matches) == 1 else None
        audio_path = wav_matches[0] if len(wav_matches) == 1 else None

        if label_name in EMOTIONS:
            label_idx = EMOTIONS.index(label_name)
        else:
            label_idx = None

        status_parts = [f"[{idx}/{total}] seq={seq}"]
        if len(vid_matches) > 1:
            status_parts.append("video=DUPLICATE")
        else:
            status_parts.append(f"video={video_path if video_path else 'MISSING'}")
        if len(wav_matches) > 1:
            status_parts.append("audio=DUPLICATE")
        else:
            status_parts.append(f"audio={audio_path if audio_path else 'MISSING'}")
        status_parts.append(f"label={label_name if label_name else 'UNKNOWN'}")
        status_parts.append(f"face_dev={face_dev}")
        print(" | ".join(status_parts))
        print(f"text={text_content}")

        if len(vid_matches) > 1 or len(wav_matches) > 1:
            msg = f"SKIP seq={seq} duplicate: video={len(vid_matches)}, audio={len(wav_matches)}"
            print(msg)
            missing_entries.append(msg)
            continue

        if video_path is None or audio_path is None or label_idx is None:
            reason = []
            if video_path is None:
                reason.append("video")
            if audio_path is None:
                reason.append("audio")
            if label_idx is None:
                reason.append("label")
            msg = f"SKIP seq={seq} missing: {','.join(reason)}"
            print(msg)
            missing_entries.append(msg)
            continue

        # Decode
        try:
            video_tensor = dataset._load_video(video_path).to(device, non_blocking=True)
            audio_tensor = dataset._load_audio(audio_path).to(device, non_blocking=True)
        except Exception as e:
            msg = f"SKIP seq={seq} decode_error: {type(e).__name__}: {e}"
            print(msg)
            missing_entries.append(msg)
            continue

        text_inputs = _tokenize_once(dataset.tokenizer, text_content, dataset.max_text_length)
        if text_inputs is not None:
            text_inputs = {k: v.to(device, non_blocking=True) for k, v in text_inputs.items()}

        cached.append(
            {
                "video": video_tensor,
                "audio": audio_tensor,
                "labels": torch.tensor(label_idx, dtype=torch.long, device=device),
                "text_inputs": text_inputs,
                "stem": seq,
            }
        )

        if args.shard_size > 0 and len(cached) >= args.shard_size:
            shard_path = (shard_dir / f"{shard_id}{output_base.suffix}").expanduser()
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            shard_obj = {
                "samples": cached,
                "config": {
                    "data_root": str(data_root),
                    "num_frames": cfg.data.num_frames,
                    "frame_strategy": cfg.data.frame_strategy,
                    "sample_rate": cfg.data.sample_rate,
                    "max_text_length": cfg.data.max_text_length,
                    "text_encoder": cfg.model.text_encoder,
                    "text_map": str(cfg.data.text_map) if cfg.data.text_map else None,
                    "emotions": EMOTIONS,
                    "device": str(device),
                },
                "missing": missing_entries,
            }
            torch.save(shard_obj, shard_path)
            shard_paths.append(shard_path)
            print(f"Saved shard {shard_id} with {len(cached)} samples to {shard_path}")
            shard_id += 1
            cached = []

    if len(cached) > 0 or (args.shard_size == 0 and len(shard_paths) == 0):
        final_path = (output_base.expanduser() if args.shard_size == 0 else (shard_dir / f"{shard_id}{output_base.suffix}").expanduser())
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_obj = {
            "samples": cached,
            "config": {
                "data_root": str(data_root),
                "num_frames": cfg.data.num_frames,
                "frame_strategy": cfg.data.frame_strategy,
                "sample_rate": cfg.data.sample_rate,
                "max_text_length": cfg.data.max_text_length,
                "text_encoder": cfg.model.text_encoder,
                "text_map": str(cfg.data.text_map) if cfg.data.text_map else None,
                "emotions": EMOTIONS,
                "device": str(device),
            },
            "missing": missing_entries,
        }
        torch.save(final_obj, final_path)
        shard_paths.append(final_path)
        print(f"Saved {len(cached)} samples to {final_path}")

    if args.shard_size > 0 and args.merge_after and len(shard_paths) > 1:
        print("Merging shards; requires enough RAM to hold all samples...")
        merged_samples: List[Dict] = []
        for p in shard_paths:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            merged_samples.extend(obj["samples"])
        merged_path = output_base.expanduser()
        merged_obj = {
            "samples": merged_samples,
            "config": {
                "data_root": str(data_root),
                "num_frames": cfg.data.num_frames,
                "frame_strategy": cfg.data.frame_strategy,
                "sample_rate": cfg.data.sample_rate,
                "max_text_length": cfg.data.max_text_length,
                "text_encoder": cfg.model.text_encoder,
                "text_map": str(cfg.data.text_map) if cfg.data.text_map else None,
                "emotions": EMOTIONS,
                "device": str(device),
            },
            "missing": missing_entries,
        }
        torch.save(merged_obj, merged_path)
        print(f"Merged {len(merged_samples)} samples to {merged_path}")

    print(
        f"Summary: total_rows={len(df)}, usable={total}, decoded={sum( len(torch.load(p, map_location='cpu')['samples']) for p in shard_paths )}, "
        f"empty_seq={len(skipped_no_seq)}, missing_path_or_label={len(missing_entries)}, shards={len(shard_paths)}"
    )


if __name__ == "__main__":
    main()
