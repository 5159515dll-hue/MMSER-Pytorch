"""评估指标工具。

这里的实现故意保持纯 Python / 轻依赖，原因是这些统计逻辑需要同时在
训练和推理阶段复用，而且希望最终写进 JSON 时结构足够直观。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def confusion_matrix_counts(y_true: list[int], y_pred: list[int], num_classes: int) -> list[list[int]]:
    """根据真实标签和预测标签构造混淆矩阵计数。"""

    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, pred in zip(y_true, y_pred):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
            matrix[truth][pred] += 1
    return matrix


def macro_f1_from_confusion(matrix: list[list[int]]) -> float:
    """从混淆矩阵直接计算 macro-F1。

    对每个类别都使用
    `F1 = 2TP / (2TP + FP + FN)`，
    最后对所有类别做算术平均。
    """

    num_classes = len(matrix)
    if num_classes == 0:
        return 0.0
    total = 0.0
    for cls_idx in range(num_classes):
        tp = matrix[cls_idx][cls_idx]
        fp = sum(matrix[row_idx][cls_idx] for row_idx in range(num_classes) if row_idx != cls_idx)
        fn = sum(matrix[cls_idx][col_idx] for col_idx in range(num_classes) if col_idx != cls_idx)
        denom = 2 * tp + fp + fn
        total += (2 * tp / denom) if denom > 0 else 0.0
    return float(total / num_classes)


def classification_summary(y_true: list[int], y_pred: list[int], label_names: list[str]) -> dict[str, Any]:
    """汇总分类任务常用指标，便于直接写入训练/推理结果。"""

    num_classes = len(label_names)
    matrix = confusion_matrix_counts(y_true, y_pred, num_classes)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(num_classes))
    per_class_recall = {}
    support = {}
    pred_counts = Counter()
    for pred in y_pred:
        if 0 <= pred < num_classes:
            pred_counts[label_names[pred]] += 1
    for idx, label_name in enumerate(label_names):
        row_sum = sum(matrix[idx])
        support[label_name] = int(row_sum)
        per_class_recall[label_name] = float(matrix[idx][idx] / row_sum) if row_sum > 0 else 0.0
    return {
        "n": int(total),
        "accuracy": float(correct / total) if total > 0 else 0.0,
        "macro_f1": float(macro_f1_from_confusion(matrix)),
        "confusion_matrix": matrix,
        "per_class_recall": per_class_recall,
        "support": support,
        "pred_counts": dict(sorted(pred_counts.items())),
    }


def speaker_majority_baseline(
    train_items: list[dict[str, Any]],
    eval_items: list[dict[str, Any]],
    label_names: list[str],
) -> dict[str, Any]:
    """构造一个“按 speaker 记忆多数类”的朴素基线。

    这个基线不是为了性能，而是为了量化 speaker confound 有多严重：
    如果它已经很高，说明模型很可能主要在识别说话人，而不是情绪。
    """

    speaker_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in train_items:
        if not item.get("is_usable"):
            continue
        label_en = item.get("label_en")
        speaker_id = item.get("speaker_id", "UNKNOWN")
        if label_en in label_names and speaker_id:
            speaker_label_counts[str(speaker_id)][str(label_en)] += 1

    speaker_to_majority_label: dict[str, str] = {}
    for speaker_id, counts in speaker_label_counts.items():
        if not counts:
            continue
        speaker_to_majority_label[speaker_id] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    y_true: list[int] = []
    y_pred: list[int] = []
    uncovered = 0
    for item in eval_items:
        if not item.get("is_usable"):
            continue
        label_en = item.get("label_en")
        speaker_id = str(item.get("speaker_id", "UNKNOWN"))
        majority_label = speaker_to_majority_label.get(speaker_id)
        if label_en not in label_names or majority_label not in label_names:
            uncovered += 1
            continue
        y_true.append(label_names.index(str(label_en)))
        y_pred.append(label_names.index(str(majority_label)))

    summary = classification_summary(y_true, y_pred, label_names)
    summary.update(
        {
            "coverage": int(len(y_true)),
            "uncovered": int(uncovered),
            "speaker_to_majority_label": dict(sorted(speaker_to_majority_label.items())),
            "speaker_train_label_counts": {
                speaker_id: dict(sorted(counts.items())) for speaker_id, counts in sorted(speaker_label_counts.items())
            },
        }
    )
    return summary
