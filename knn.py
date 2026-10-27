"""
knn.py
======
K 近邻（K-Nearest Neighbors）算法的**从零实现**。

本模块不依赖 scikit-learn 的任何 KNN 组件，只使用 NumPy 完成：
距离计算、邻居搜索、多数投票、分类与回归预测。

模块结构
--------
1. 距离函数     euclidean_distance / manhattan_distance / cdist
2. 最近邻搜索   nearest_neighbors
3. 投票与平票   majority_vote
4. 分类器       KNNClassifier
5. 回归器       KNNRegressor
6. 性能对照     循环实现 vs 向量化实现

统一约定
--------
* X 形如 (n_samples, n_features)；单样本 (n_features,) 会自动升维。
* 所有距离都是"越小越近"。
* 平票规则 tie_break：
    - 'nearest'  默认。平票的类别中，谁的最近邻更近谁获胜（更符合直觉，
                 因为"最近的那个样本"本身携带最强证据）；
                 若最近邻距离也完全相同，取标签较小者。
    - 'smallest' 直接取标签较小者。**与 sklearn 的平票行为一致** ——
                 sklearn 内部用 scipy.stats.mode 取众数，平票时返回最小值。
                 代价是有系统性偏向（总是偏爱字典序靠前的类别）。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SUPPORTED_METRICS",
    "euclidean_distance",
    "manhattan_distance",
    "cdist",
    "nearest_neighbors",
    "majority_vote",
    "KNNClassifier",
    "KNNRegressor",
]

# 距离为 0 时距离加权会除零，用它兜底
EPS = 1e-12

SUPPORTED_METRICS = ("euclidean", "manhattan")


# ---------------------------------------------------------------------------
# 0. 内部工具
# ---------------------------------------------------------------------------
def _as_2d(X, name: str = "X") -> np.ndarray:
    """把输入统一成二维浮点数组，并做基本合法性检查。"""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组 (n_samples, n_features)，收到 {X.ndim} 维")
    if X.shape[0] == 0:
        raise ValueError(f"{name} 为空：至少需要 1 个样本")
    if X.shape[1] == 0:
        raise ValueError(f"{name} 的特征数为 0：至少需要 1 个特征")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} 含有 NaN 或无穷值，请先做缺失值处理")
    return X


def _check_metric(metric: str) -> str:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"不支持的距离度量 {metric!r}，可选：{SUPPORTED_METRICS}")
    return metric


def _check_pair_dims(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"维度不一致：a 有 {a.shape[-1]} 个特征，b 有 {b.shape[-1]} 个特征，无法计算距离"
        )


# ---------------------------------------------------------------------------
# 1. 距离函数
# ---------------------------------------------------------------------------
def euclidean_distance(a, b) -> float:
    """两个样本之间的欧氏距离 L2 = sqrt(sum((a_i - b_i)^2))。"""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    _check_pair_dims(a, b)
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def manhattan_distance(a, b) -> float:
    """两个样本之间的曼哈顿距离 L1 = sum(|a_i - b_i|)。"""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    _check_pair_dims(a, b)
    return float(np.abs(a - b).sum())


def euclidean_distances(X, Y) -> np.ndarray:
    """批量欧氏距离，返回 (len(X), len(Y)) 矩阵。

    使用 (a-b)^2 = a^2 + b^2 - 2ab 展开，避免构造 (n, m, d) 的大数组。
    """
    X = _as_2d(X, "X")
    Y = _as_2d(Y, "Y")
    _check_pair_dims(X, Y)
    sq = (X * X).sum(axis=1)[:, None] + (Y * Y).sum(axis=1)[None, :] - 2.0 * (X @ Y.T)
    np.maximum(sq, 0.0, out=sq)  # 浮点误差可能让平方距离略小于 0
    return np.sqrt(sq, out=sq)


def manhattan_distances(X, Y) -> np.ndarray:
    """批量曼哈顿距离，返回 (len(X), len(Y)) 矩阵。

    注意：广播会生成 (n, m, d) 的中间数组，超大数据集应分块计算。
    """
    X = _as_2d(X, "X")
    Y = _as_2d(Y, "Y")
    _check_pair_dims(X, Y)
    return np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=-1)


def _euclidean_distances_loop(X, Y) -> np.ndarray:
    """欧氏距离的纯 Python 循环版，仅用于性能对照。"""
    X = _as_2d(X, "X")
    Y = _as_2d(Y, "Y")
    _check_pair_dims(X, Y)
    out = np.empty((X.shape[0], Y.shape[0]), dtype=float)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            out[i, j] = euclidean_distance(X[i], Y[j])
    return out


def _manhattan_distances_loop(X, Y) -> np.ndarray:
    """曼哈顿距离的纯 Python 循环版，仅用于性能对照。"""
    X = _as_2d(X, "X")
    Y = _as_2d(Y, "Y")
    _check_pair_dims(X, Y)
    out = np.empty((X.shape[0], Y.shape[0]), dtype=float)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            out[i, j] = manhattan_distance(X[i], Y[j])
    return out


def cdist(X, Y, metric: str = "euclidean", vectorized: bool = True) -> np.ndarray:
    """统一的批量距离入口。

    参数
    ----
    X, Y       : 二维数组，支持单样本与多样本
    metric     : 'euclidean' | 'manhattan'
    vectorized : True 走向量化实现，False 走循环实现（用于性能对照）
    """
    _check_metric(metric)
    if metric == "euclidean":
        return euclidean_distances(X, Y) if vectorized else _euclidean_distances_loop(X, Y)
    return manhattan_distances(X, Y) if vectorized else _manhattan_distances_loop(X, Y)


# ---------------------------------------------------------------------------
# 2. 最近邻搜索
# ---------------------------------------------------------------------------
def nearest_neighbors(dists, k: int, return_distance: bool = True):
    """在一维距离数组里取最近的 k 个邻居。

    使用 np.argsort(kind='stable')：距离相同时保持原始顺序，
    保证结果可复现（这正是 random_state 之外的另一处确定性来源）。
    """
    d = np.asarray(dists, dtype=float).ravel()
    n = d.shape[0]
    if n == 0:
        raise ValueError("距离数组为空，无法搜索最近邻")
    if not isinstance(k, (int, np.integer)):
        raise ValueError(f"k 必须是整数，收到 {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"k 必须为正整数，收到 {k}")
    if k > n:
        raise ValueError(f"k={k} 大于可用训练样本数 {n}，请减小 k 或补充训练数据")

    idx = np.argsort(d, kind="stable")[:k]
    return (d[idx], idx) if return_distance else idx


# ---------------------------------------------------------------------------
# 3. 投票与平票
# ---------------------------------------------------------------------------
def majority_vote(labels, distances=None, tie_break: str = "nearest"):
    """多数投票，并显式处理平票。

    参数
    ----
    labels     : k 个邻居的标签
    distances  : k 个邻居对应的距离，tie_break='nearest' 时需要
    tie_break  : 'nearest'（默认）或 'smallest'
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("邻居标签为空，无法投票")
    if tie_break not in ("nearest", "smallest"):
        raise ValueError(f"不支持的平票策略 {tie_break!r}，可选：('nearest', 'smallest')")

    classes, counts = np.unique(labels, return_counts=True)
    top = classes[counts == counts.max()]
    if top.size == 1:
        return top[0]

    # 走到这里说明平票
    if tie_break == "nearest" and distances is not None:
        dists = np.asarray(distances, dtype=float).ravel()
        best_label, best_dist = None, np.inf
        for label in top.tolist():
            d_min = float(dists[labels == label].min())
            # 距离更近者胜；距离相同则取较小标签，保证确定性
            if d_min < best_dist - EPS or (abs(d_min - best_dist) <= EPS
                                           and best_label is not None and label < best_label):
                best_label, best_dist = label, d_min
        return best_label

    # 注意：标签可能是字符串，np.ndarray.min() 对字符串类型不支持，故用 Python 的 min
    return min(top.tolist())


# ---------------------------------------------------------------------------
# 4. KNN 分类器
# ---------------------------------------------------------------------------
class KNNClassifier:
    """K 近邻分类器。

    参数
    ----
    k          : 邻居数，必须为正整数且不超过训练样本数
    metric     : 'euclidean' | 'manhattan'
    weights    : 'uniform'（等权投票）| 'distance'（距离倒数加权）
    tie_break  : 平票策略，见模块说明
    vectorized : 是否使用向量化距离（False 时切到循环实现，用于性能对照）

    示例
    ----
    >>> clf = KNNClassifier(k=5).fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(self, k: int = 5, metric: str = "euclidean",
                 weights: str = "uniform", tie_break: str = "nearest",
                 vectorized: bool = True):
        self.k = k
        self.metric = _check_metric(metric)
        if weights not in ("uniform", "distance"):
            raise ValueError(f"不支持的 weights {weights!r}，可选：('uniform', 'distance')")
        self.weights = weights
        self.tie_break = tie_break
        self.vectorized = vectorized
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None

    # -- 训练：KNN 是惰性学习，只记住训练集 ------------------------------
    def fit(self, X, y):
        X = _as_2d(X, "X_train")
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y_train 必须是一维标签数组，收到 {y.ndim} 维")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"样本数不一致：X_train 有 {X.shape[0]} 行，y_train 有 {y.shape[0]} 个")
        if self.k > X.shape[0]:
            raise ValueError(
                f"k={self.k} 大于训练样本数 {X.shape[0]}，请减小 k 或补充训练数据"
            )
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        return self

    # -- 邻居检索 --------------------------------------------------------
    def kneighbors(self, X, n_neighbors: int | None = None):
        """返回 (距离, 索引)，形状均为 (n_query, n_neighbors)。"""
        if self.X_train_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        X = _as_2d(X, "X")
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"维度不一致：训练集 {self.X_train_.shape[1]} 个特征，预测数据 {X.shape[1]} 个特征"
            )
        k = self.k if n_neighbors is None else n_neighbors
        D = cdist(X, self.X_train_, metric=self.metric, vectorized=self.vectorized)
        dists = np.empty((X.shape[0], k), dtype=float)
        idxs = np.empty((X.shape[0], k), dtype=int)
        for i in range(X.shape[0]):
            dists[i], idxs[i] = nearest_neighbors(D[i], k)
        return dists, idxs

    # -- 预测 ------------------------------------------------------------
    def predict(self, X):
        if self.X_train_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        X = _as_2d(X, "X")
        D = cdist(X, self.X_train_, metric=self.metric, vectorized=self.vectorized)
        k = self.k
        y_pred = np.empty(X.shape[0], dtype=self.y_train_.dtype)
        for i in range(X.shape[0]):
            d_k, idx_k = nearest_neighbors(D[i], k)
            labels_k = self.y_train_[idx_k]
            if self.weights == "distance":
                y_pred[i] = _weighted_vote(labels_k, d_k, self.tie_break)
            else:
                y_pred[i] = majority_vote(labels_k, d_k, self.tie_break)
        return y_pred

    def predict_proba(self, X):
        """返回 (n_query, n_classes) 的投票占比，便于画置信度或做 ROC。"""
        if self.X_train_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        X = _as_2d(X, "X")
        D = cdist(X, self.X_train_, metric=self.metric, vectorized=self.vectorized)
        k = self.k
        proba = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=float)
        for i in range(X.shape[0]):
            d_k, idx_k = nearest_neighbors(D[i], k)
            labels_k = self.y_train_[idx_k]
            w = _neighbor_weights(d_k) if self.weights == "distance" else np.ones(k)
            for c_idx, c in enumerate(self.classes_):
                proba[i, c_idx] = w[labels_k == c].sum()
            proba[i] /= proba[i].sum()
        return proba

    def score(self, X, y) -> float:
        """准确率。"""
        y = np.asarray(y)
        return float((self.predict(X) == y).mean())


def _neighbor_weights(distances) -> np.ndarray:
    """距离倒数权重。距离为 0 的邻居独占全部权重（完全匹配优先）。"""
    d = np.asarray(distances, dtype=float).ravel()
    zero_mask = d <= EPS
    if zero_mask.any():
        w = np.zeros_like(d)
        w[zero_mask] = 1.0 / zero_mask.sum()
        return w
    return 1.0 / d


def _weighted_vote(labels, distances, tie_break: str = "nearest"):
    labels = np.asarray(labels)
    w = _neighbor_weights(distances)
    classes = np.unique(labels)
    scores = np.array([w[labels == c].sum() for c in classes])
    best = classes[scores == scores.max()]
    if best.size == 1:
        return best[0]
    # 加权后仍平票：沿用同一套平票策略
    return majority_vote(labels, distances, tie_break)


# ---------------------------------------------------------------------------
# 5. KNN 回归器
# ---------------------------------------------------------------------------
class KNNRegressor:
    """K 近邻回归器。

    weights='uniform'  -> k 个邻居的算术平均
    weights='distance' -> 距离倒数加权平均，距离越近权重越大
    """

    def __init__(self, k: int = 5, metric: str = "euclidean",
                 weights: str = "uniform", vectorized: bool = True):
        self.k = k
        self.metric = _check_metric(metric)
        if weights not in ("uniform", "distance"):
            raise ValueError(f"不支持的 weights {weights!r}，可选：('uniform', 'distance')")
        self.weights = weights
        self.vectorized = vectorized
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        X = _as_2d(X, "X_train")
        y = np.asarray(y, dtype=float).ravel()
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"样本数不一致：X_train 有 {X.shape[0]} 行，y_train 有 {y.shape[0]} 个")
        if self.k > X.shape[0]:
            raise ValueError(f"k={self.k} 大于训练样本数 {X.shape[0]}，请减小 k 或补充训练数据")
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        if self.X_train_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        X = _as_2d(X, "X")
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"维度不一致：训练集 {self.X_train_.shape[1]} 个特征，预测数据 {X.shape[1]} 个特征"
            )
        D = cdist(X, self.X_train_, metric=self.metric, vectorized=self.vectorized)
        y_pred = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            d_k, idx_k = nearest_neighbors(D[i], self.k)
            target_k = self.y_train_[idx_k]
            if self.weights == "distance":
                w = _neighbor_weights(d_k)
                y_pred[i] = float((w * target_k).sum() / w.sum())
            else:
                y_pred[i] = float(target_k.mean())
        return y_pred

    def score(self, X, y) -> float:
        """R^2 决定系数。"""
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = float(((y - y_pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
