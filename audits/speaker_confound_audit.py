import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import openpyxl
except ImportError as exc:  # pragma: no cover
    raise SystemExit("openpyxl is required; install with pip install openpyxl") from exc

# 你提供的规则：
# A: surprise, fear, angry
# B: neutral, disgust
# C: happy, sad
SPEAKER_LABELS_EN = {
    "A": {"surprise", "fear", "angry"},
    "B": {"neutral", "disgusted"},
    "C": {"happy", "sad"},
}

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
    p = argparse.ArgumentParser(description="Audit speaker-label confounding in video_databases.xlsx")
    p.add_argument("--xlsx", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    return p.parse_args()


def load_rows(xlsx: Path) -> List[Tuple[str, str]]:
    wb = openpyxl.load_workbook(xlsx)
    ws = wb.active
    rows: List[Tuple[str, str]] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        seq, _mongolian, _chinese, emotion = (row + (None, None, None, None))[:4]
        if seq is None:
            continue
        rows.append((str(seq).strip(), str(emotion or "").strip()))
    return rows


def infer_speaker_from_label_en(label_en: str) -> str:
    for spk, labs in SPEAKER_LABELS_EN.items():
        if label_en in labs:
            return spk
    return "UNKNOWN"


def emit(lines: List[str], msg: str = ""):
    print(msg)
    lines.append(msg)


def main():
    args = parse_args()
    rows = load_rows(args.xlsx)

    # counts
    by_label = Counter()
    by_speaker = Counter()
    label_to_speakers: Dict[str, set] = defaultdict(set)
    speaker_to_labels: Dict[str, set] = defaultdict(set)
    unknown_labels = Counter()

    for stem, label_cn in rows:
        label_en = CN_TO_EN.get(label_cn, label_cn)
        if label_en not in {"angry", "disgusted", "fear", "happy", "neutral", "sad", "surprise"}:
            unknown_labels[label_cn] += 1
        spk = infer_speaker_from_label_en(label_en)

        by_label[label_en] += 1
        by_speaker[spk] += 1
        label_to_speakers[label_en].add(spk)
        speaker_to_labels[spk].add(label_en)

    lines: List[str] = []
    emit(lines, "=== Speaker Confound Audit ===")
    emit(lines, f"xlsx: {args.xlsx}")
    emit(lines, f"rows: {len(rows)}")
    emit(lines, "")

    emit(lines, "-- Label counts")
    for lab, n in by_label.most_common():
        emit(lines, f"  {lab}: {n}")

    emit(lines, "")
    emit(lines, "-- Speaker counts (inferred from label mapping)")
    for spk, n in by_speaker.most_common():
        emit(lines, f"  {spk}: {n}")

    emit(lines, "")
    emit(lines, "-- Speaker -> labels")
    for spk in sorted(speaker_to_labels.keys()):
        emit(lines, f"  {spk}: {sorted(speaker_to_labels[spk])}")

    emit(lines, "")
    emit(lines, "-- Label -> speakers")
    for lab in sorted(label_to_speakers.keys()):
        emit(lines, f"  {lab}: {sorted(label_to_speakers[lab])}")

    emit(lines, "")
    fatal = False
    # If every label appears in only one speaker, dataset is speaker-confounded.
    labels_single_speaker = [lab for lab, spks in label_to_speakers.items() if len(spks - {"UNKNOWN"}) == 1]
    labels_multi_speaker = [lab for lab, spks in label_to_speakers.items() if len(spks - {"UNKNOWN"}) >= 2]

    if len(labels_multi_speaker) == 0 and len(labels_single_speaker) > 0:
        fatal = True
        emit(lines, "FINDING: Every emotion label is tied to exactly one speaker (speaker-confounded).")
        emit(lines, "Impact: Video-only/Audio-only accuracy can be near-perfect by recognizing the speaker, not emotion.")
        emit(lines, "Fix: Need multiple speakers per label (ideally each speaker covers all 7 labels), then do group split by speaker.")
    else:
        emit(lines, "FINDING: At least some labels appear in multiple speakers.")
        emit(lines, "Next: Use group split by speaker to evaluate generalization.")

    if unknown_labels:
        emit(lines, "")
        emit(lines, "-- Unknown/unmapped label strings (check CN_TO_EN)")
        for lab, n in unknown_labels.most_common():
            emit(lines, f"  {lab!r}: {n}")

    emit(lines, "")
    emit(lines, f"fatal_confound: {fatal}")
    emit(lines, "=== Done ===")

    if args.report:
        report_path = args.report.expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
