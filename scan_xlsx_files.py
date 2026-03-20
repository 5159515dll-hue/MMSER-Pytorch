import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def index_files(root: Path, suffix: str) -> Dict[str, List[Path]]:
    files: Dict[str, List[Path]] = {}
    for path in root.rglob(f"*.{suffix}"):
        name = path.name
        files.setdefault(name, []).append(path)
    return files


def main():
    parser = argparse.ArgumentParser(description="Scan XLSX and verify mp4/wav presence by sequence id")
    parser.add_argument("--xlsx", default="databases/视频数据集对应文档.xlsx")
    parser.add_argument("--data-root", default="databases")
    parser.add_argument("--report", default="databases/scan_report.txt")
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parents[1]

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.is_absolute():
        xlsx_path = (Path.cwd() / xlsx_path)
    if not xlsx_path.exists():
        xlsx_path = script_root / args.xlsx

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        if not data_root.exists():
            data_root = script_root / args.data_root
        else:
            data_root = Path.cwd() / data_root

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = script_root / args.report

    if not xlsx_path.exists():
        raise FileNotFoundError(f"XLSX not found at {xlsx_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found at {data_root}")

    df = pd.read_excel(xlsx_path)

    video_index = index_files(data_root, "mp4")
    audio_index = index_files(data_root, "wav")

    missing_video = []
    missing_audio = []
    dup_video = []
    dup_audio = []

    for _, row in df.iterrows():
        seq = str(row.get("序号") or row.iloc[0]).strip()
        if not seq:
            continue
        vid_key = f"{seq}.mp4"
        wav_key = f"{seq}.wav"

        vids = video_index.get(vid_key, [])
        auds = audio_index.get(wav_key, [])

        if len(vids) == 0:
            missing_video.append(seq)
        elif len(vids) > 1:
            dup_video.append((seq, vids))

        if len(auds) == 0:
            missing_audio.append(seq)
        elif len(auds) > 1:
            dup_audio.append((seq, auds))

    lines = []
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Missing video: {len(missing_video)} -> {missing_video}")
    lines.append(f"Missing audio: {len(missing_audio)} -> {missing_audio}")
    lines.append("Duplicate videos:")
    for seq, paths in dup_video:
        lines.append(f"  {seq}: {[str(p) for p in paths]}")
    lines.append("Duplicate audios:")
    for seq, paths in dup_audio:
        lines.append(f"  {seq}: {[str(p) for p in paths]}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")

    # Also print brief summary to stdout
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
