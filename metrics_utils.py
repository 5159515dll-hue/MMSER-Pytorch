"""评估指标工具。

这里的实现故意保持纯 Python / 轻依赖，原因是这些统计逻辑需要同时在
训练和推理阶段复用，而且希望最终写进 JSON 时结构足够直观。
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
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


def mean_and_sample_std(values: list[float]) -> tuple[float, float]:
    """Return the sample mean and sample standard deviation."""

    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = float(sum(float(v) for v in values) / n)
    if n <= 1:
        return mean, 0.0
    var = sum((float(v) - mean) ** 2 for v in values) / (n - 1)
    return mean, float(math.sqrt(max(0.0, var)))


def _betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return float(bt * _betacf(a, b, x) / a)
    return float(1.0 - bt * _betacf(b, a, 1.0 - x) / b)


def student_t_cdf(t_value: float, degrees_of_freedom: int) -> float:
    """Evaluate the Student's t CDF with a pure-Python implementation."""

    df = int(degrees_of_freedom)
    if df <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    t = float(t_value)
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    ibeta = _regularized_incomplete_beta(df / 2.0, 0.5, x)
    if t > 0.0:
        return float(1.0 - 0.5 * ibeta)
    return float(0.5 * ibeta)


def student_t_ppf(probability: float, degrees_of_freedom: int) -> float:
    """Inverse CDF for Student's t via monotonic binary search."""

    p = float(probability)
    df = int(degrees_of_freedom)
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be in (0, 1)")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -student_t_ppf(1.0 - p, df)
    lo = 0.0
    hi = 1.0
    while student_t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1.0e6:
            break
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def mean_confidence_interval_t(values: list[float], confidence: float = 0.95) -> dict[str, float | None]:
    """Compute a t-based confidence interval for the sample mean."""

    n = len(values)
    mean, sample_std = mean_and_sample_std(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "low": None, "high": None, "n": 0}
    if n == 1:
        return {"mean": mean, "std": 0.0, "low": mean, "high": mean, "n": 1}
    alpha = 1.0 - float(confidence)
    critical = student_t_ppf(1.0 - alpha / 2.0, n - 1)
    half_width = critical * (sample_std / math.sqrt(n))
    return {
        "mean": mean,
        "std": sample_std,
        "low": float(mean - half_width),
        "high": float(mean + half_width),
        "n": int(n),
    }


def paired_t_test(left: list[float], right: list[float], confidence: float = 0.95) -> dict[str, float | int | None]:
    """Paired t-test and CI for right-minus-left differences."""

    if len(left) != len(right):
        raise ValueError("paired_t_test requires equal-length inputs")
    diffs = [float(r) - float(l) for l, r in zip(left, right)]
    n = len(diffs)
    mean_diff, sample_std = mean_and_sample_std(diffs)
    positive_gain_count = int(sum(1 for d in diffs if d > 0.0))
    non_negative_gain_count = int(sum(1 for d in diffs if d >= 0.0))
    if n == 0:
        return {
            "n": 0,
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "t_statistic": None,
            "p_value": None,
            "ci_low": None,
            "ci_high": None,
            "positive_gain_count": positive_gain_count,
            "non_negative_gain_count": non_negative_gain_count,
        }
    if n == 1 or sample_std == 0.0:
        p_value = 0.0 if mean_diff != 0.0 else 1.0
        return {
            "n": int(n),
            "mean_diff": float(mean_diff),
            "std_diff": float(sample_std),
            "t_statistic": None if n == 1 else float("inf") if mean_diff != 0.0 else 0.0,
            "p_value": float(p_value),
            "ci_low": float(mean_diff),
            "ci_high": float(mean_diff),
            "positive_gain_count": positive_gain_count,
            "non_negative_gain_count": non_negative_gain_count,
        }
    standard_error = sample_std / math.sqrt(n)
    t_statistic = mean_diff / standard_error
    tail_prob = max(0.0, 1.0 - student_t_cdf(abs(t_statistic), n - 1))
    alpha = 1.0 - float(confidence)
    critical = student_t_ppf(1.0 - alpha / 2.0, n - 1)
    half_width = critical * standard_error
    return {
        "n": int(n),
        "mean_diff": float(mean_diff),
        "std_diff": float(sample_std),
        "t_statistic": float(t_statistic),
        "p_value": float(min(1.0, max(0.0, 2.0 * tail_prob))),
        "ci_low": float(mean_diff - half_width),
        "ci_high": float(mean_diff + half_width),
        "positive_gain_count": positive_gain_count,
        "non_negative_gain_count": non_negative_gain_count,
    }


def holm_bonferroni_adjust(p_values: list[float | None]) -> list[float | None]:
    """Apply Holm-Bonferroni correction while preserving original order."""

    indexed = [(idx, float(p)) for idx, p in enumerate(p_values) if p is not None]
    if not indexed:
        return [None for _ in p_values]
    sorted_pairs = sorted(indexed, key=lambda pair: pair[1])
    adjusted: list[float | None] = [None for _ in p_values]
    running_max = 0.0
    total = len(sorted_pairs)
    for rank, (original_idx, p_value) in enumerate(sorted_pairs):
        factor = total - rank
        candidate = min(1.0, max(0.0, factor * p_value))
        running_max = max(running_max, candidate)
        adjusted[original_idx] = running_max
    return adjusted


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


def speaker_only_baseline(
    train_items: list[dict[str, Any]],
    eval_items: list[dict[str, Any]],
    label_names: list[str],
) -> dict[str, Any]:
    """构造一个“只看 speaker_id”的分布基线。

    这个基线和 majority baseline 的区别是，它保留了每个 speaker 在训练集中的
    标签后验分布。最终 top-1 预测仍然会退化到每个 speaker 的多数类，但结果里
    会明确暴露“模型只凭 speaker 最多能知道什么”。
    """

    speaker_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in train_items:
        if not item.get("is_usable"):
            continue
        label_en = item.get("label_en")
        speaker_id = item.get("speaker_id", "UNKNOWN")
        if label_en in label_names and speaker_id:
            speaker_label_counts[str(speaker_id)][str(label_en)] += 1

    speaker_posteriors: dict[str, dict[str, float]] = {}
    speaker_argmax_label: dict[str, str] = {}
    for speaker_id, counts in sorted(speaker_label_counts.items()):
        total = sum(counts.values())
        if total <= 0:
            continue
        posterior = {
            label_name: float(counts.get(label_name, 0) / total)
            for label_name in label_names
        }
        speaker_posteriors[speaker_id] = posterior
        speaker_argmax_label[speaker_id] = sorted(
            posterior.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[0][0]

    y_true: list[int] = []
    y_pred: list[int] = []
    uncovered = 0
    for item in eval_items:
        if not item.get("is_usable"):
            continue
        label_en = item.get("label_en")
        speaker_id = str(item.get("speaker_id", "UNKNOWN"))
        pred_label = speaker_argmax_label.get(speaker_id)
        if label_en not in label_names or pred_label not in label_names:
            uncovered += 1
            continue
        y_true.append(label_names.index(str(label_en)))
        y_pred.append(label_names.index(str(pred_label)))

    summary = classification_summary(y_true, y_pred, label_names)
    summary.update(
        {
            "coverage": int(len(y_true)),
            "uncovered": int(uncovered),
            "speaker_label_posteriors": speaker_posteriors,
            "speaker_argmax_label": speaker_argmax_label,
            "speaker_train_label_counts": {
                speaker_id: dict(sorted(counts.items())) for speaker_id, counts in sorted(speaker_label_counts.items())
            },
        }
    )
    return summary
