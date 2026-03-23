import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

EMOTIONS = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprise"]
CN_TO_EN = {
    "愤怒": "angry",
    "厌恶": "disgusted",
    "恐惧": "fear",
    "高兴": "happy",
    "快乐": "happy",
    "开心": "happy",
    "中性": "neutral",
    "悲伤": "sad",
    "惊讶": "surprise",
}


def parse_args():
    p = argparse.ArgumentParser(description="Batch-level modality alignment check (filesystem + optional text_map JSON/XLSX)")
    p.add_argument("--data-root", type=Path, default=default_databases_dir(_REPO_ROOT))
    p.add_argument("--text-map", type=Path, default=None, help="Optional text_map JSON or XLSX (stem -> text)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for simulated grouping")
    p.add_argument("--report", type=Path, default=None, help="Optional report path")
    return p.parse_args()


def log_factory(report_lines):
    def log(msg=""):
        print(msg)
        report_lines.append(msg)
    return log


def scan_modalities(data_root: Path):
    video_label: Dict[str, str] = {}
    audio_label: Dict[str, str] = {}
    missing_audio_dir: List[str] = []
    for label in EMOTIONS:
        vdir = data_root / label
        adir = vdir / f"{label}_audio"
        if vdir.exists():
            for mp4 in vdir.glob("*.mp4"):
                stem = mp4.stem
                video_label.setdefault(stem, label)
        if adir.exists():
            for wav in adir.glob("*.wav"):
                stem = wav.stem
                audio_label.setdefault(stem, label)
        else:
            missing_audio_dir.append(str(adir))
    return video_label, audio_label, missing_audio_dir


def load_text_map(text_map_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    suffix = text_map_path.suffix.lower()
    if suffix == ".json":
        with open(text_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): v for k, v in data.items()}, {}
    if suffix in {".xlsx", ".xls"}:
        try:
            import openpyxl
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise SystemExit("openpyxl is required to read XLSX; install with pip install openpyxl") from exc

        wb = openpyxl.load_workbook(text_map_path)
        ws = wb.active
        text_map: Dict[str, str] = {}
        label_map: Dict[str, str] = {}
        for idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
            seq, mongolian, chinese, emotion = (row + (None, None, None, None))[:4]
            if seq is None:
                continue
            stem = str(seq).strip()
            text_val = mongolian if mongolian else chinese
            if text_val:
                text_map[stem] = str(text_val).strip()
            if emotion:
                label_map[stem] = str(emotion).strip()
        return text_map, label_map
    raise SystemExit(f"Unsupported text_map format: {text_map_path}")


def main():
    args = parse_args()
    report_lines: List[str] = []
    log = log_factory(report_lines)

    data_root = args.data_root.expanduser()
    text_map, text_labels = load_text_map(args.text_map.expanduser()) if args.text_map else ({}, {})

    video_label, audio_label, missing_audio_dir = scan_modalities(data_root)

    ids_video = set(video_label.keys())
    ids_audio = set(audio_label.keys())
    ids_text = set(text_map.keys()) if text_map else set()

    conflicts = {}
    all_ids = ids_video | ids_audio
    for stem in all_ids:
        labels: Set[str] = set()
        if stem in video_label:
            labels.add(video_label[stem])
        if stem in audio_label:
            labels.add(audio_label[stem])
        if len(labels) > 1:
            conflicts[stem] = labels

    missing_video = ids_audio - ids_video
    missing_audio = ids_video - ids_audio

    missing_text = set()
    text_only = set()
    if text_map:
        missing_text = (ids_video | ids_audio) - ids_text
        text_only = ids_text - (ids_video | ids_audio)

    common_ids = ids_video & ids_audio
    if text_map:
        common_ids = common_ids & ids_text

    bs = max(1, args.batch_size)
    batches = (len(common_ids) + bs - 1) // bs

    log("=== Batch Alignment Check (filesystem scan) ===")
    log(f"data_root: {data_root}")
    log(f"videos: {len(ids_video)} | audios: {len(ids_audio)} | texts: {len(ids_text) if text_map else 'n/a'}")
    log(f"common (video+audio{'+text' if text_map else ''}): {len(common_ids)}")
    log(f"batch_size (simulated): {bs}, batches: {batches}")
    log(f"conflicts (same stem, multiple labels): {len(conflicts)}")
    log(f"missing_video (audio has, video missing): {len(missing_video)}")
    log(f"missing_audio (video has, audio missing): {len(missing_audio)}")
    if text_map:
        log(f"missing_text (video/audio has, text missing): {len(missing_text)}")
        log(f"text_only (only text, no video/audio): {len(text_only)}")
    if missing_audio_dir:
        log(f"audio dirs missing: {len(missing_audio_dir)}")
        for d in missing_audio_dir:
            log(f"  missing audio dir: {d}")

    if conflicts:
        log("-- Conflicts (stem -> labels)")
        for stem, labels in sorted(conflicts.items()):
            log(f"  {stem}: {sorted(labels)}")
    if missing_video:
        log("-- Missing video (show up to 20):")
        for stem in sorted(list(missing_video))[:20]:
            log(f"  {stem}")
    if missing_audio:
        log("-- Missing audio (show up to 20):")
        for stem in sorted(list(missing_audio))[:20]:
            log(f"  {stem}")
    if text_map:
        if missing_text:
            log("-- Missing text (show up to 20):")
            for stem in sorted(list(missing_text))[:20]:
                log(f"  {stem}")
        if text_only:
            log("-- Text only (show up to 20):")
            for stem in sorted(list(text_only))[:20]:
                log(f"  {stem}")

    if text_labels:
        mismatched_labels = {}
        for stem, cn_label in text_labels.items():
            en_label = CN_TO_EN.get(cn_label, cn_label)
            video_lbl = video_label.get(stem)
            audio_lbl = audio_label.get(stem)
            expected = video_lbl or audio_lbl
            if expected and en_label != expected:
                mismatched_labels[stem] = (en_label, expected)
        log(f"text label mismatches vs video/audio: {len(mismatched_labels)}")
        if mismatched_labels:
            log("-- Label mismatches from XLSX (stem: xlsx -> dataset)")
            for stem, (xlsx_label, ds_label) in sorted(mismatched_labels.items())[:20]:
                log(f"  {stem}: {xlsx_label} -> {ds_label}")

    log("=== Done ===")

    if args.report:
        report_path = args.report.expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"Report saved to {report_path}")

"""
功能：检查每个样本的 video/audio 同 stem、同标签，检测同 stem 多标签冲突；支持 text_map JSON 或 video_databases.xlsx；如提供 XLSX 会额外比对情感标签；可输出报告。
"""
if __name__ == "__main__":
    main()

def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


_REPO_ROOT = _ensure_repo_root_on_path()

from path_utils import default_databases_dir
