import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EMOTIONS


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data leakage / alignment checks")
    p.add_argument("--data-root", type=Path, default=Path("databases"))
    p.add_argument("--xlsx", type=Path, default=Path("databases/video_databases.xlsx"))
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to save the text report (stdout content)",
    )
    return p.parse_args()


def scan_files(data_root: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (video_id_to_label, audio_id_to_label)."""
    video_map: Dict[str, str] = {}
    audio_map: Dict[str, str] = {}
    for label in EMOTIONS:
        vdir = data_root / label
        adir = vdir / f"{label}_audio"
        if vdir.exists():
            for mp4 in vdir.glob("*.mp4"):
                stem = mp4.stem
                video_map.setdefault(stem, label)
        if adir.exists():
            for wav in adir.glob("*.wav"):
                stem = wav.stem
                audio_map.setdefault(stem, label)
    return video_map, audio_map


def load_xlsx(xlsx: Path) -> Dict[str, Dict]:
    df = pd.read_excel(xlsx)
    rows: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        seq = row.get("序号") if "序号" in row else row.iloc[0]
        if pd.isna(seq):
            continue
        seq = str(seq).strip()
        if not seq:
            continue
        label_raw = str(row.get("情感类别", "")).strip()
        label = ZH2EN.get(label_raw, label_raw).lower()
        text = None
        for key in ("蒙文", "中文"):
            if key in row and not pd.isna(row[key]):
                val = str(row[key]).strip()
                if val and val.lower() != "nan":
                    text = val
                    break
        rows[seq] = {"label": label, "text": text, "raw_label": label_raw}
    return rows


def check_conflicts(map_a: Dict[str, str], map_b: Dict[str, str]) -> Dict[str, Set[str]]:
    conflicts: Dict[str, Set[str]] = defaultdict(set)
    for k, v in map_a.items():
        conflicts[k].add(v)
    for k, v in map_b.items():
        conflicts[k].add(v)
    return {k: v for k, v in conflicts.items() if len(v) > 1}


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser()
    xlsx = args.xlsx.expanduser()
    if not xlsx.exists():
        raise FileNotFoundError(f"XLSX not found: {xlsx}")

    report_lines = []

    def log(line: str = ""):
        print(line)
        report_lines.append(line)

    video_map, audio_map = scan_files(data_root)
    xlsx_rows = load_xlsx(xlsx)

    ids_video = set(video_map.keys())
    ids_audio = set(audio_map.keys())
    ids_text = set(xlsx_rows.keys())

    # 1) 样本唯一性：同一 ID 不应跨 label
    conflicts_va = check_conflicts(video_map, audio_map)
    # 2) 模态对齐：视频/音频/文本交集应一致
    common_all = ids_video & ids_audio & ids_text
    missing_video = ids_text - ids_video
    missing_audio = ids_text - ids_audio
    missing_text = (ids_video | ids_audio) - ids_text

    # 3) 文本标签与目录标签一致性
    label_mismatch = []
    for seq in common_all:
        lbl_text = xlsx_rows[seq]["label"]
        lbl_video = video_map.get(seq)
        lbl_audio = audio_map.get(seq)
        if lbl_text not in EMOTIONS:
            label_mismatch.append((seq, "text_label_not_in_emotions", lbl_text, lbl_video, lbl_audio))
        elif lbl_text != lbl_video or lbl_text != lbl_audio:
            label_mismatch.append((seq, "label_conflict", lbl_text, lbl_video, lbl_audio))

    # 4) 重复 ID 检查（同一 ID 不同标签）
    conflicts_xlsx = check_conflicts({k: v["label"] for k, v in xlsx_rows.items()}, {})
    conflicts_any = {**conflicts_va, **conflicts_xlsx}

    # 5) 汇总
    log("=== Data Leakage / Alignment Checks ===")
    log(f"data_root: {data_root}")
    log(f"xlsx: {xlsx}")
    log(f"videos: {len(ids_video)} | audios: {len(ids_audio)} | texts: {len(ids_text)}")
    log(f"common(video,audio,text): {len(common_all)}")
    log(f"missing_video (in text not in video): {len(missing_video)}")
    log(f"missing_audio (in text not in audio): {len(missing_audio)}")
    log(f"missing_text  (in video/audio not in text): {len(missing_text)}")
    log(f"conflicts (id with multiple labels across modalities/xlsx): {len(conflicts_any)}")
    log(f"label_mismatch (text label vs dir label): {len(label_mismatch)}")

    if conflicts_any:
        log("-- Conflicts (id -> labels)")
        for k, labels in sorted(conflicts_any.items()):
            log(f"  {k}: {sorted(labels)}")
    if label_mismatch:
        log("-- Label mismatches (seq, issue, text_label, video_label, audio_label)")
        for rec in label_mismatch:
            log("  " + str(rec))
    if missing_video:
        log("-- In text but missing video:")
        for k in sorted(list(missing_video))[:20]:
            log(f"  {k}")
    if missing_audio:
        log("-- In text but missing audio:")
        for k in sorted(list(missing_audio))[:20]:
            log(f"  {k}")
    if missing_text:
        log("-- In video/audio but missing text:")
        for k in sorted(list(missing_text))[:20]:
            log(f"  {k}")

    log("=== Done ===")

    if args.report:
        report_path = args.report.expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"Report saved to {report_path}")

"""
新增脚本： data_leak_check.py
检查点：
ID 唯一性与跨模态标签冲突（视频/音频/文本）
视频/音频/文本的缺失情况
xlsx 中的标签与目录标签是否一致
默认读取：--data-root databases，--xlsx databases/video_databases.xlsx
运行示例（在项目根目录）：
输出会包含：样本数、交集数、缺失列表（截取最多 20 条）、标签冲突/不一致明细等
"""
if __name__ == "__main__":
    main()
