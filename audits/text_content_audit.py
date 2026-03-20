import argparse
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import openpyxl
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("openpyxl is required; install with pip install openpyxl") from exc

EMOTION_EN = ["happy", "angry", "sad", "fear", "disgust", "surprise", "neutral"]
EMOTION_CN = ["高兴", "开心", "快乐", "愤怒", "生气", "悲伤", "伤心", "恐惧", "害怕", "厌恶", "惊讶", "中性"]
DESC_CN = ["很开心", "非常开心", "特别开心", "很愤怒", "非常愤怒", "特别愤怒", "很生气", "非常生气", "特别生气", "很悲伤", "非常悲伤", "很害怕", "非常害怕", "很恐惧", "非常恐惧", "很惊讶", "非常惊讶", "很厌恶", "非常厌恶"]
LABEL_MAP = {
    "愤怒": "angry",
    "厌恶": "disgust",
    "恐惧": "fear",
    "高兴": "happy",
    "开心": "happy",
    "快乐": "happy",
    "中性": "neutral",
    "悲伤": "sad",
    "惊讶": "surprise",
}


def parse_args():
    p = argparse.ArgumentParser(description="Audit text content for leakage cues in XLSX")
    p.add_argument("--xlsx", type=Path, required=True, help="Path to video_databases.xlsx")
    p.add_argument("--report", type=Path, default=None, help="Optional report output path")
    p.add_argument("--max-show", type=int, default=20, help="Max rows to show per finding category")
    return p.parse_args()


def load_rows(xlsx: Path) -> List[Tuple[str, str, str, str]]:
    wb = openpyxl.load_workbook(xlsx)
    ws = wb.active
    rows: List[Tuple[str, str, str, str]] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        seq, mongolian, chinese, emotion = (row + (None, None, None, None))[:4]
        if seq is None:
            continue
        rows.append((str(seq).strip(), str(mongolian or "").strip(), str(chinese or "").strip(), str(emotion or "").strip()))
    return rows


def contains_keyword(text: str, keywords: List[str], lowercase: bool = True) -> bool:
    t = text.lower() if lowercase else text
    for kw in keywords:
        if lowercase:
            if kw.lower() in t:
                return True
        else:
            if kw in t:
                return True
    return False


def audit_rows(rows: List[Tuple[str, str, str, str]], max_show: int):
    hits_en = []
    hits_cn = []
    hits_desc = []
    hits_label_in_text = []

    for seq, mongolian, chinese, emotion in rows:
        text = " ".join([mongolian, chinese]).strip()
        text_lower = text.lower()
        if text and contains_keyword(text_lower, EMOTION_EN, lowercase=True):
            hits_en.append((seq, text))
        if text and contains_keyword(text, EMOTION_CN, lowercase=False):
            hits_cn.append((seq, text))
        if text and contains_keyword(text, DESC_CN, lowercase=False):
            hits_desc.append((seq, text))
        if emotion:
            en_label = LABEL_MAP.get(emotion, emotion)
            cn_label = emotion
            if (en_label and en_label.lower() in text_lower) or (cn_label and cn_label in text):
                hits_label_in_text.append((seq, text, emotion))

    summary = {
        "emotion_words_en": hits_en,
        "emotion_words_cn": hits_cn,
        "emotion_desc_cn": hits_desc,
        "label_in_text": hits_label_in_text,
    }
    return summary


def emit(log_lines: List[str], msg: str = ""):
    print(msg)
    log_lines.append(msg)


def main():
    args = parse_args()
    rows = load_rows(args.xlsx)
    findings = audit_rows(rows, args.max_show)

    log_lines: List[str] = []
    emit(log_lines, "=== Text Content Audit ===")
    emit(log_lines, f"xlsx: {args.xlsx}")
    emit(log_lines, f"rows: {len(rows)}")

    for key, label in [
        ("emotion_words_en", "Contains emotion words (EN)"),
        ("emotion_words_cn", "Contains emotion words (CN)"),
        ("emotion_desc_cn", "Contains emotion description (CN phrases)"),
        ("label_in_text", "Label string appears inside text"),
    ]:
        hits = findings[key]
        emit(log_lines, f"{label}: {len(hits)}")
        if hits:
            emit(log_lines, f"-- Samples (up to {args.max_show}):")
            for idx, item in enumerate(hits[: args.max_show]):
                if key == "label_in_text":
                    seq, text, emotion = item
                    emit(log_lines, f"  {seq} | label={emotion} | {text}")
                else:
                    seq, text = item
                    emit(log_lines, f"  {seq} | {text}")

    emit(log_lines, "=== Done ===")

    if args.report:
        report_path = args.report.expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(log_lines), encoding="utf-8")
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
