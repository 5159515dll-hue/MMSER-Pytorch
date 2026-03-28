import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


_REPO_ROOT = _ensure_repo_root_on_path()

import pandas as pd
import torch

from hf_compat import ensure_transformers_torch_compat

ensure_transformers_torch_compat()
from transformers import AutoTokenizer

from path_utils import default_databases_dir, default_xlsx_path

from src.config import DEFAULT_CONFIG, EMOTIONS
from src.data.dataset import MultiModalEmotionDataset
from inference import load_model, predict


ZH2EN = {
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


def _pick_text(row: pd.Series) -> str:
    for key in ("蒙文", "中文"):
        if key in row and not pd.isna(row[key]):
            val = str(row[key]).strip()
            if val and val.lower() != "nan":
                return val
    return ""


def _parse_seq(row: pd.Series) -> Optional[str]:
    # Prefer explicit column name; fall back to first column.
    seq = row.get("序号") if "序号" in row else row.iloc[0]
    if pd.isna(seq):
        return None
    s = str(seq).strip()
    return s if s else None


def _resolve_label_en(row: pd.Series) -> Optional[str]:
    raw = str(row.get("情感类别", "")).strip()
    if not raw:
        return None
    label = ZH2EN.get(raw, raw).strip().lower()
    return label if label in EMOTIONS else None


def _paths_for_seq(data_root: Path, label_en: str, seq: str) -> Tuple[Optional[Path], Optional[Path]]:
    # Fast path: follow the dataset folder convention.
    video = data_root / label_en / f"{seq}.mp4"
    audio = data_root / label_en / f"{label_en}_audio" / f"{seq}.wav"
    if video.exists() and audio.exists():
        return video, audio

    # Fallback: search (handles non-standard layouts / stray files)
    vid_matches = sorted(data_root.rglob(f"{seq}.mp4"))
    wav_matches = sorted(data_root.rglob(f"{seq}.wav"))
    video = vid_matches[0] if len(vid_matches) == 1 else None
    audio = wav_matches[0] if len(wav_matches) == 1 else None
    return video, audio


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    p = argparse.ArgumentParser(
        description="Batch inference from the dataset XLSX (序号/蒙文/中文/情感类别)"
    )
    p.add_argument("--data-root", type=Path, default=default_databases_dir(_REPO_ROOT))
    p.add_argument(
        "--xlsx",
        type=Path,
        default=default_xlsx_path(_REPO_ROOT, "video_databases.xlsx"),
        help="Excel path with columns: 序号/蒙文/中文/情感类别",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/best.pt"),
    )
    p.add_argument("--num-frames", type=int, default=cfg.data.num_frames)
    p.add_argument(
        "--frame-strategy",
        choices=["random", "uniform"],
        default=cfg.data.frame_strategy,
    )
    p.add_argument("--sample-rate", type=int, default=cfg.data.sample_rate)
    p.add_argument(
        "--tokenizer",
        default=cfg.model.text_encoder,
        help="HuggingFace tokenizer name; set to empty string to disable text",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/inference_results.jsonl"),
        help="Write JSON Lines here (one result per row)",
    )
    p.add_argument("--limit", type=int, default=0, help="If >0, only run first N rows")
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at first decode/predict error",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP during inference",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print per-sample debug info (slower)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_root = args.data_root.expanduser()
    xlsx = args.xlsx.expanduser()
    out_path = args.output.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not xlsx.exists():
        raise FileNotFoundError(f"XLSX not found: {xlsx}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp

    tokenizer_name = (args.tokenizer or "").strip()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) if tokenizer_name else None

    model = load_model(args.checkpoint, cfg=DEFAULT_CONFIG, device=device)

    # Processor: reuse exact same decode/trim logic as training/predecode.
    processor = MultiModalEmotionDataset(
        root=str(data_root),
        num_frames=args.num_frames,
        frame_strategy=args.frame_strategy,
        sample_rate=args.sample_rate,
        tokenizer_name="",
        max_text_length=DEFAULT_CONFIG.data.max_text_length,
        text_map_path=None,
        is_train=False,
        scan=False,
    )

    df = pd.read_excel(xlsx)
    total = len(df)
    limit = args.limit if args.limit and args.limit > 0 else total

    correct = 0
    seen = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            if seen >= limit:
                break

            seq = _parse_seq(row)
            if not seq:
                skipped += 1
                continue

            label_en = _resolve_label_en(row)
            text = _pick_text(row)

            if not label_en:
                rec = {
                    "seq": seq,
                    "status": "skip",
                    "reason": "unknown_label",
                    "label_raw": str(row.get("情感类别", "")).strip(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                skipped += 1
                continue

            video_path, audio_path = _paths_for_seq(data_root, label_en, seq)
            if video_path is None or audio_path is None:
                rec = {
                    "seq": seq,
                    "status": "skip",
                    "reason": "missing_file",
                    "label": label_en,
                    "video": str(video_path) if video_path else None,
                    "audio": str(audio_path) if audio_path else None,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                skipped += 1
                continue

            try:
                result = predict(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    video_path=video_path,
                    audio_path=audio_path,
                    text=text,
                    num_frames=args.num_frames,
                    frame_strategy=args.frame_strategy,
                    sample_rate=args.sample_rate,
                    device=device,
                    amp=amp,
                    debug=args.debug,
                )
            except Exception as e:
                rec = {
                    "seq": seq,
                    "status": "error",
                    "label": label_en,
                    "video": str(video_path),
                    "audio": str(audio_path),
                    "error": f"{type(e).__name__}: {e}",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if args.fail_fast:
                    raise
                continue

            pred = result.get("emotion")
            ok = pred == label_en
            correct += int(ok)
            seen += 1

            rec = {
                "seq": seq,
                "status": "ok",
                "label": label_en,
                "pred": pred,
                "match": bool(ok),
                "probability": result.get("probability"),
                "intensity": result.get("intensity"),
                "video": str(video_path),
                "audio": str(audio_path),
                "text": text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Console progress (lightweight)
            if (seen % 20) == 0 or args.debug:
                acc = correct / max(1, seen)
                print(f"[{seen}/{limit}] acc={acc:.4f} last seq={seq} label={label_en} pred={pred} p={rec['probability']}")

    acc = correct / max(1, seen)
    print(
        json.dumps(
            {
                "xlsx": str(xlsx),
                "output": str(out_path),
                "processed": seen,
                "skipped": skipped,
                "accuracy": acc,
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
