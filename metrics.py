"""
metrics.py
==========
手写的评估指标：准确率、混淆矩阵、分类报告、回归误差。

不调用 sklearn.metrics —— 本作业要求指标也由自己实现。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "accuracy_score",
    "confusion_matrix",
    "classification_report",
    "mse",
    "rmse",
    "mae",
    "r2_score",
]


def accuracy_score(y_true, y_pred) -> float:
    """准确率 = 预测正确的样本数 / 总样本数。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"长度不一致：y_true {y_true.shape[0]} 个，y_pred {y_pred.shape[0]} 个")
    if y_true.shape[0] == 0:
        raise ValueError("样本为空，无法计算准确率")
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, labels=None):
    """混淆矩阵。

    返回
    ----
    cm     : (n_classes, n_classes) 矩阵，行 = 真实类别，列 = 预测类别
    labels : 与行列顺序对应的类别名
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true 与 y_pred 长度不一致")
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((labels.size, labels.size), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm, labels


def classification_report(y_true, y_pred, labels=None, digits: int = 4) -> str:
    """简化版分类报告：每个类别的 precision / recall / f1 / support + 宏平均与加权平均。"""
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    y_true = np.asarray(y_true)
    total = cm.sum()
    if total == 0:
        raise ValueError("样本为空，无法生成分类报告")

    lines = []
    header = f"{'类别':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>9}"
    lines.append(header)
    lines.append("-" * len(header))

    supports = cm.sum(axis=1).astype(float)
    precisions, recalls, f1s = [], [], []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        lines.append(
            f"{str(label):>12} {precision:>10.{digits}f} {recall:>10.{digits}f} "
            f"{f1:>10.{digits}f} {int(supports[i]):>9d}"
        )

    acc = float(np.trace(cm) / total)
    macro_p, macro_r, macro_f1 = (
        float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))
    )
    weights = supports / total
    weighted_p = float(np.dot(weights, precisions))
    weighted_r = float(np.dot(weights, recalls))
    weighted_f1 = float(np.dot(weights, f1s))

    lines.append("-" * len(header))
    lines.append(
        f"{'accuracy':>12} {'':>10} {'':>10} {acc:>10.{digits}f} {int(total):>9d}"
    )
    lines.append(
        f"{'macro avg':>12} {macro_p:>10.{digits}f} {macro_r:>10.{digits}f} "
        f"{macro_f1:>10.{digits}f} {int(total):>9d}"
    )
    lines.append(
        f"{'weighted avg':>12} {weighted_p:>10.{digits}f} {weighted_r:>10.{digits}f} "
        f"{weighted_f1:>10.{digits}f} {int(total):>9d}"
    )
    return "\n".join(lines)


def mse(y_true, y_pred) -> float:
    """均方误差。"""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    """均方根误差。"""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """平均绝对误差。"""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    """决定系数 R^2，越接近 1 越好。"""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
