"""
model_selection.py
==================
K 折交叉验证。

为什么不用测试集调参：测试集只能用一次、只在最后做最终评估。
选 k 必须在训练集内部用交叉验证完成，否则同样属于数据泄漏。

另外，标准化器必须**在每个折内部重新 fit**，只使用该折的训练部分。
"""

from __future__ import annotations

import numpy as np

from preprocess import StandardScaler

__all__ = ["KFold", "StratifiedKFold", "cross_val_score"]


class KFold:
    """K 折划分，返回 (train_idx, val_idx) 生成器。"""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int | None = 42):
        if n_splits < 2:
            raise ValueError(f"n_splits 至少为 2，收到 {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        n = np.asarray(X).shape[0]
        if self.n_splits > n:
            raise ValueError(f"n_splits={self.n_splits} 大于样本数 {n}")
        idx = np.arange(n)
        if self.shuffle:
            idx = np.random.default_rng(self.random_state).permutation(idx)
        folds = np.array_split(idx, self.n_splits)
        for i in range(self.n_splits):
            val_idx = np.sort(folds[i])
            train_idx = np.sort(np.concatenate([folds[j] for j in range(self.n_splits) if j != i]))
            yield train_idx, val_idx


class StratifiedKFold:
    """分层 K 折：每个折内的类别比例与整体一致，适合类别不均衡的分类任务。"""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int | None = 42):
        if n_splits < 2:
            raise ValueError(f"n_splits 至少为 2，收到 {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n = X.shape[0]
        if self.n_splits > n:
            raise ValueError(f"n_splits={self.n_splits} 大于样本数 {n}")
        rng = np.random.default_rng(self.random_state)
        fold_of = np.empty(n, dtype=int)
        for cls in np.unique(y):
            cls_idx = np.flatnonzero(y == cls)
            if self.shuffle:
                cls_idx = rng.permutation(cls_idx)
            # 把该类的样本轮流分配到各折，保证比例均衡
            fold_of[cls_idx] = np.arange(cls_idx.size) % self.n_splits
        for i in range(self.n_splits):
            val_idx = np.flatnonzero(fold_of == i)
            train_idx = np.flatnonzero(fold_of != i)
            yield train_idx, val_idx


def cross_val_score(estimator_factory, X, y, cv, scale: bool = True):
    """交叉验证。

    参数
    ----
    estimator_factory : 无参可调用对象，每次调用返回一个**未训练**的模型
                        （用工厂而不是实例，保证每折都是干净的新模型）
    scale             : 是否在每个折内部做 Z-score（fit 只用该折训练部分）

    返回
    ----
    scores : 每折的得分数组
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    scores = []
    for train_idx, val_idx in cv.split(X, y) if isinstance(cv, StratifiedKFold) else cv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        if scale:
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_val = scaler.transform(X_val)
        model = estimator_factory().fit(X_tr, y_tr)
        scores.append(float(model.score(X_val, y_val)))
    return np.asarray(scores, dtype=float)
