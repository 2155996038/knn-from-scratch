"""
preprocess.py
=============
数据读取、探索、划分与标准化。

关键点：**标准化器只能在训练集上 fit**，测试集只能 transform。
否则训练集的均值/方差里混入了测试集信息，属于典型的数据泄漏。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["load_csv", "describe_dataset", "train_test_split", "StandardScaler"]


def load_csv(path, target_col: str = "target", drop_cols=()):
    """读取 CSV，返回 (X, y, feature_names)。

    参数
    ----
    path        : CSV 路径
    target_col  : 标签列名
    drop_cols   : 额外需要丢弃的列（如 id）
    """
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"CSV 中找不到标签列 {target_col!r}，现有列：{list(df.columns)}")

    drop = [c for c in drop_cols if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    y = df[target_col].to_numpy()
    X = df.drop(columns=[target_col]).to_numpy(dtype=float)
    feature_names = [c for c in df.columns if c != target_col]
    return X, y, feature_names


def describe_dataset(X, y, feature_names=None) -> dict:
    """输出样本数、特征数、类别分布、缺失值等基本信息。"""
    df = pd.DataFrame(X, columns=feature_names)
    info = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": list(feature_names) if feature_names is not None else None,
        "missing_values": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
    }

    y_arr = np.asarray(y)
    # 数值型且取值很多 -> 视为回归目标
    if np.issubdtype(y_arr.dtype, np.floating) and np.unique(y_arr).size > 20:
        info["task"] = "regression"
        info["target_stats"] = {
            "min": float(y_arr.min()),
            "max": float(y_arr.max()),
            "mean": float(y_arr.mean()),
            "std": float(y_arr.std(ddof=1)),
        }
    else:
        info["task"] = "classification"
        values, counts = np.unique(y_arr, return_counts=True)
        info["class_distribution"] = {
            str(v): {"count": int(c), "ratio": round(float(c) / len(y_arr), 4)}
            for v, c in zip(values, counts)
        }
        info["n_classes"] = int(values.size)

    info["feature_stats"] = {
        name: {
            "mean": round(float(df[name].mean()), 4),
            "std": round(float(df[name].std(ddof=1)), 4),
            "min": round(float(df[name].min()), 4),
            "max": round(float(df[name].max()), 4),
        }
        for name in df.columns
    }
    return info


def train_test_split(X, y, test_size: float = 0.3, random_state: int | None = None,
                     stratify=None, shuffle: bool = True):
    """划分训练集/测试集。

    参数
    ----
    test_size   : 测试集比例 (0, 1)
    random_state: 随机种子，固定后划分结果可复现
    stratify    : 传入标签数组时按类别分层抽样，保持训练/测试集类别比例一致

    返回
    ----
    X_train, X_test, y_train, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    if n != y.shape[0]:
        raise ValueError(f"样本数不一致：X 有 {n} 行，y 有 {y.shape[0]} 个")
    if not 0 < test_size < 1:
        raise ValueError(f"test_size 必须在 (0, 1) 之间，收到 {test_size}")
    n_test = int(round(n * test_size))
    if n_test == 0 or n_test >= n:
        raise ValueError(f"按 test_size={test_size} 划分后样本为空，请调整划分比例")

    rng = np.random.default_rng(random_state)

    if stratify is not None:
        strat = np.asarray(stratify)
        if strat.shape[0] != n:
            raise ValueError("stratify 长度必须与样本数一致")
        test_idx = []
        for cls in np.unique(strat):
            cls_idx = np.flatnonzero(strat == cls)
            if shuffle:
                cls_idx = rng.permutation(cls_idx)
            n_cls_test = max(1, int(round(cls_idx.size * test_size)))
            n_cls_test = min(n_cls_test, cls_idx.size - 1)  # 每类至少留 1 个训练样本
            test_idx.extend(cls_idx[:n_cls_test].tolist())
        test_idx = np.array(sorted(test_idx), dtype=int)
    else:
        perm = rng.permutation(n) if shuffle else np.arange(n)
        test_idx = np.sort(perm[:n_test])

    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.flatnonzero(train_mask)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class StandardScaler:
    """Z-score 标准化：x' = (x - mean) / std。

    必须在训练集上 fit，测试集只 transform —— 这是"避免信息泄漏"的核心。

    >>> scaler = StandardScaler().fit(X_train)
    >>> X_train_s = scaler.transform(X_train)
    >>> X_test_s  = scaler.transform(X_test)
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X 必须是二维数组，收到 {X.ndim} 维")
        if X.shape[0] == 0:
            raise ValueError("X 为空，无法拟合标准化器")
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # 常数列的标准差为 0，置 1 避免除零
        self.std_ = np.where(std == 0, 1.0, std)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("标准化器尚未拟合，请先调用 fit()")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X 必须是二维数组，收到 {X.ndim} 维")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"特征数不一致：fit 时是 {self.n_features_in_}，现在是 {X.shape[1]}"
            )
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        if self.mean_ is None:
            raise RuntimeError("标准化器尚未拟合，请先调用 fit()")
        return np.asarray(X_scaled, dtype=float) * self.std_ + self.mean_
