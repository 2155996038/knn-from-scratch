"""
test_knn.py
===========
单元测试。运行：pytest -v

覆盖距离函数、最近邻搜索、投票（含平票）、分类器、回归器、
预处理（含防泄漏）、评估指标，共 30 个用例。
"""

from __future__ import annotations

import numpy as np
import pytest

import metrics as M
from knn import (
    KNNClassifier,
    KNNRegressor,
    cdist,
    euclidean_distance,
    majority_vote,
    manhattan_distance,
    nearest_neighbors,
)
from model_selection import KFold, StratifiedKFold, cross_val_score
from preprocess import StandardScaler, train_test_split


# ---------------------------------------------------------------------------
# 1. 距离函数
# ---------------------------------------------------------------------------
def test_euclidean_distance_hand_calculation():
    """欧氏距离与手算结果一致。"""
    assert euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)
    assert euclidean_distance([1, -2, 3], [-1, 2, 0]) == pytest.approx(np.sqrt(29))


def test_manhattan_distance_hand_calculation():
    """曼哈顿距离与手算结果一致。"""
    assert manhattan_distance([0, 0], [3, 4]) == pytest.approx(7.0)
    assert manhattan_distance([1, -2, 3], [-1, 2, 0]) == pytest.approx(9.0)


def test_distance_is_symmetric_and_zero_on_self():
    """距离满足对称性，且样本到自身为 0。"""
    a, b = np.array([1.5, -2.0, 0.5]), np.array([-1.0, 0.0, 3.0])
    for fn in (euclidean_distance, manhattan_distance):
        assert fn(a, b) == pytest.approx(fn(b, a))
        assert fn(a, a) == pytest.approx(0.0)


def test_euclidean_satisfies_triangle_inequality():
    """欧氏距离满足三角不等式。"""
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b, c = rng.normal(size=(3, 4))
        assert euclidean_distance(a, c) <= euclidean_distance(a, b) + euclidean_distance(b, c) + 1e-12


def test_batch_matrix_matches_loop_implementation():
    """向量化距离矩阵与纯循环实现逐元素一致。"""
    rng = np.random.default_rng(42)
    X, Y = rng.normal(size=(17, 5)), rng.normal(size=(11, 5))
    for metric in ("euclidean", "manhattan"):
        assert np.allclose(cdist(X, Y, metric, vectorized=True),
                           cdist(X, Y, metric, vectorized=False), atol=1e-10)


def test_batch_matrix_matches_manual_expansion():
    """批量欧氏距离与 NumPy 直接展开实现一致。"""
    rng = np.random.default_rng(7)
    X, Y = rng.normal(size=(9, 3)), rng.normal(size=(6, 3))
    ref = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    assert np.allclose(cdist(X, Y, "euclidean"), ref, atol=1e-12)


def test_single_sample_input_is_promoted():
    """单样本（一维）输入会被自动升维，结果形状为 (1, n)。"""
    assert cdist([1.0, 2.0], [[0.0, 0.0], [1.0, 2.0]]).shape == (1, 2)


def test_unsupported_metric_raises():
    """不支持的距离度量必须报错并列出可选值。"""
    with pytest.raises(ValueError, match="不支持的距离度量"):
        cdist(np.zeros((2, 2)), np.zeros((2, 2)), metric="cosine")


def test_dimension_mismatch_raises():
    """特征维度不一致时必须报错。"""
    with pytest.raises(ValueError, match="维度不一致"):
        euclidean_distance([1.0, 2.0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# 2. 最近邻搜索
# ---------------------------------------------------------------------------
def test_nearest_neighbors_returns_sorted_topk():
    """返回的 k 个距离必须升序，且索引对应最近的点。"""
    d = np.array([5.0, 1.0, 3.0, 0.5, 9.0])
    dists, idx = nearest_neighbors(d, 3)
    assert idx.tolist() == [3, 1, 2]
    assert dists.tolist() == pytest.approx([0.5, 1.0, 3.0])
    assert np.all(np.diff(dists) >= 0)


def test_nearest_neighbors_k_equals_one_picks_minimum():
    """k=1 时取到的就是最小距离。"""
    d = np.array([4.0, 2.0, 8.0, 1.5])
    dists, idx = nearest_neighbors(d, 1)
    assert idx[0] == int(np.argmin(d))
    assert dists[0] == pytest.approx(d.min())


def test_nearest_neighbors_k_greater_than_n_raises():
    """k 大于样本数时报错，提示减小 k 或补充数据。"""
    with pytest.raises(ValueError, match="大于可用训练样本数"):
        nearest_neighbors(np.array([1.0, 2.0]), 3)


def test_nearest_neighbors_invalid_k_raises():
    """k 非正数时报错。"""
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="k 必须为正整数"):
            nearest_neighbors(np.array([1.0, 2.0]), bad_k)


def test_nearest_neighbors_is_deterministic_on_ties():
    """距离完全相同时，索引顺序稳定可复现。"""
    d = np.array([1.0, 1.0, 1.0, 1.0])
    _, idx1 = nearest_neighbors(d, 2)
    _, idx2 = nearest_neighbors(d, 2)
    assert idx1.tolist() == idx2.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# 3. 投票与平票
# ---------------------------------------------------------------------------
def test_majority_vote_single_class():
    """所有邻居同一类时直接返回该类。"""
    assert majority_vote(["cat", "cat", "cat"]) == "cat"


def test_majority_vote_clear_winner():
    """票数不同时返回票数最多者。"""
    assert majority_vote(["A", "B", "B", "B", "A"]) == "B"


def test_majority_vote_tie_prefers_nearest():
    """平票时 tie_break='nearest' 让最近邻所属类别获胜。"""
    labels = np.array(["B", "A", "B", "A"])
    dists = np.array([0.10, 0.40, 0.50, 0.90])
    # A 的最近邻距离 0.40，B 的最近邻距离 0.10 -> B 胜
    assert majority_vote(labels, dists, tie_break="nearest") == "B"
    # 交换标签后 A 胜，证明结论由距离而非标签本身决定
    assert majority_vote(np.array(["A", "B", "A", "B"]), dists, tie_break="nearest") == "A"


def test_majority_vote_tie_smallest_label():
    """平票时 tie_break='smallest' 返回标签较小者。"""
    labels = np.array(["B", "A", "B", "A"])
    dists = np.array([0.10, 0.40, 0.50, 0.90])
    assert majority_vote(labels, dists, tie_break="smallest") == "A"


def test_majority_vote_empty_raises():
    """空邻居列表无法投票，必须报错。"""
    with pytest.raises(ValueError, match="邻居标签为空"):
        majority_vote([])


def test_majority_vote_invalid_tie_break_raises():
    """非法平票策略报错。"""
    with pytest.raises(ValueError, match="不支持的平票策略"):
        majority_vote(["A", "B"], [0.1, 0.2], tie_break="random")


# ---------------------------------------------------------------------------
# 4. KNN 分类器
# ---------------------------------------------------------------------------
def test_classifier_perfect_on_separable_data():
    """线性可分数据上 k=1 应达到 100% 准确率。"""
    X_train = np.array([[0.0], [1.0], [10.0], [11.0]])
    y_train = np.array([0, 0, 1, 1])
    clf = KNNClassifier(k=1).fit(X_train, y_train)
    X_test = np.array([[0.2], [0.8], [10.5], [11.2]])
    assert clf.score(X_test, np.array([0, 0, 1, 1])) == pytest.approx(1.0)


def test_classifier_matches_brute_force_reference():
    """向量化实现与逐点暴力实现预测完全一致。"""
    rng = np.random.default_rng(2026)
    X_train = rng.normal(size=(80, 4))
    y_train = rng.integers(0, 3, 80)
    X_test = rng.normal(size=(25, 4))

    clf = KNNClassifier(k=7).fit(X_train, y_train)
    pred_fast = clf.predict(X_test)

    pred_ref = []
    for q in X_test:
        d = np.array([euclidean_distance(q, x) for x in X_train])
        dists, idx = nearest_neighbors(d, 7)
        pred_ref.append(majority_vote(y_train[idx], dists))
    assert pred_fast.tolist() == pred_ref


def test_classifier_predict_proba_sums_to_one():
    """predict_proba 每行概率之和为 1，且形状为 (n_query, n_classes)。"""
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(60, 3)), rng.integers(0, 3, 60)
    clf = KNNClassifier(k=5).fit(X, y)
    proba = clf.predict_proba(rng.normal(size=(12, 3)))
    assert proba.shape == (12, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_classifier_distance_weights_favor_nearest():
    """距离加权时，越近的邻居影响越大（重合点应完全复制其标签）。"""
    X = np.array([[0.0], [1.0]])
    y = np.array([10, 20])
    clf = KNNClassifier(k=2, weights="distance").fit(X, y)
    assert clf.predict(np.array([[0.0]]))[0] == 10
    assert clf.predict(np.array([[0.999]]))[0] == 20


def test_classifier_dimension_mismatch_raises():
    """预测数据特征数与训练集不一致时报错。"""
    clf = KNNClassifier(k=1).fit(np.zeros((5, 3)), np.zeros(5))
    with pytest.raises(ValueError, match="维度不一致"):
        clf.predict(np.zeros((2, 4)))


def test_classifier_empty_training_set_raises():
    """空训练集报错。"""
    with pytest.raises(ValueError, match="为空"):
        KNNClassifier(k=1).fit(np.empty((0, 3)), np.array([]))


def test_classifier_nan_raises():
    """含 NaN 的数据报错，提示先做缺失值处理。"""
    with pytest.raises(ValueError, match="NaN"):
        KNNClassifier(k=1).fit(np.full((5, 2), np.nan), np.zeros(5))


def test_classifier_k_greater_than_samples_raises():
    """k 超过训练样本数时报错。"""
    with pytest.raises(ValueError, match="大于训练样本数"):
        KNNClassifier(k=10).fit(np.zeros((5, 2)), np.zeros(5))


def test_classifier_requires_fit_before_predict():
    """未训练就预测必须报错。"""
    with pytest.raises(RuntimeError, match="尚未训练"):
        KNNClassifier(k=1).predict(np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# 5. KNN 回归器
# ---------------------------------------------------------------------------
def test_regressor_fits_constant_target_exactly():
    """目标恒定时，任意 k 都应精确预测该常数。"""
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    y = np.full(30, 3.75)
    reg = KNNRegressor(k=5).fit(X, y)
    assert np.allclose(reg.predict(X[:5]), 3.75)


def test_regressor_distance_weight_no_overflow_on_exact_match():
    """查询点与训练点重合时，距离加权不应出现 inf / nan，且应复制该点目标值。"""
    X = np.array([[0.0], [1.0]])
    y = np.array([10.0, 20.0])
    reg = KNNRegressor(k=2, weights="distance").fit(X, y)
    pred = reg.predict(np.array([[0.0]]))
    assert np.isfinite(pred).all()
    assert pred[0] == pytest.approx(10.0)


def test_regressor_mean_equals_neighbor_average():
    """uniform 权重下的预测值等于 k 个邻居目标值的算术平均。"""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 2.0, 4.0, 6.0])
    reg = KNNRegressor(k=2, weights="uniform").fit(X, y)
    # 查询点 0.4：最近两个邻居是 0.0 和 1.0，目标 0.0 和 2.0
    assert reg.predict(np.array([[0.4]]))[0] == pytest.approx(1.0)


def test_regressor_k_too_large_raises():
    """回归器 k 过大同样报错。"""
    with pytest.raises(ValueError, match="大于训练样本数"):
        KNNRegressor(k=99).fit(np.zeros((10, 1)), np.zeros(10))


def test_regressor_r2_score_on_training_data():
    """k=1 在训练集上预测自身，R² 应为 1。"""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 2))
    y = X[:, 0] * 2 + 1
    reg = KNNRegressor(k=1).fit(X, y)
    assert reg.score(X, y) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. 预处理
# ---------------------------------------------------------------------------
def test_standard_scaler_gives_zero_mean_unit_std():
    """标准化后训练集均值为 0、标准差为 1。"""
    rng = np.random.default_rng(5)
    X = rng.normal(loc=5.0, scale=3.0, size=(200, 3))
    Z = StandardScaler().fit_transform(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-12)


def test_standard_scaler_constant_column_is_safe():
    """常数列标准差为 0，不应产生除零。"""
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    Z = StandardScaler().fit_transform(X)
    assert np.all(np.isfinite(Z))
    assert np.allclose(Z[:, 0], 0.0)


def test_scaler_fit_uses_only_train_data():
    """防泄漏：标准化参数只由传入的 X 决定，与测试集无关。"""
    rng = np.random.default_rng(9)
    X_train = rng.normal(size=(50, 2))
    X_test = rng.normal(loc=100.0, size=(20, 2))  # 故意均值差异很大

    scaler = StandardScaler().fit(X_train)
    mean_before = scaler.mean_.copy()
    scaler.transform(X_test)  # 只 transform，不应改变已拟合的参数
    assert np.allclose(scaler.mean_, mean_before)
    assert np.allclose(scaler.mean_, X_train.mean(axis=0))


def test_train_test_split_is_reproducible():
    """相同 random_state 下划分结果完全一致。"""
    X, y = np.arange(100).reshape(50, 2), np.arange(50)
    a = train_test_split(X, y, test_size=0.3, random_state=42)
    b = train_test_split(X, y, test_size=0.3, random_state=42)
    for arr_a, arr_b in zip(a, b):
        assert np.array_equal(arr_a, arr_b)
    c = train_test_split(X, y, test_size=0.3, random_state=7)
    assert not np.array_equal(a[1], c[1])


def test_train_test_split_has_no_overlap_and_keeps_all_samples():
    """训练集与测试集不重叠，且并集等于全量样本。"""
    X, y = np.arange(60).reshape(30, 2), np.arange(30)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
    assert len(y_tr) + len(y_te) == 30
    assert not set(y_tr.tolist()) & set(y_te.tolist())
    assert set(y_tr.tolist()) | set(y_te.tolist()) == set(y.tolist())


def test_train_test_split_stratify_keeps_ratio():
    """分层抽样后测试集类别比例与整体一致。"""
    y = np.array([0] * 70 + [1] * 30)
    X = np.arange(y.size).reshape(-1, 1)
    _, _, _, y_te = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    assert abs(y_te.mean() - y.mean()) < 1e-12


def test_invalid_test_size_raises():
    """非法 test_size 报错。"""
    X, y = np.zeros((10, 2)), np.zeros(10)
    with pytest.raises(ValueError, match="test_size"):
        train_test_split(X, y, test_size=1.5)


# ---------------------------------------------------------------------------
# 7. 评估指标
# ---------------------------------------------------------------------------
def test_accuracy_score_basic():
    """准确率 = 命中数 / 总数。"""
    assert M.accuracy_score([0, 1, 1, 0], [0, 1, 0, 0]) == pytest.approx(0.75)


def test_accuracy_score_empty_raises():
    """空输入报错。"""
    with pytest.raises(ValueError, match="样本为空"):
        M.accuracy_score([], [])


def test_confusion_matrix_structure():
    """混淆矩阵的对角线之和等于预测正确数，且行列和等于真实/预测分布。"""
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 1, 2]
    cm, labels = M.confusion_matrix(y_true, y_pred)
    assert labels.tolist() == [0, 1, 2]
    assert np.trace(cm) == 4
    assert cm.sum() == 5
    assert cm.sum(axis=1).tolist() == [2, 2, 1]  # 真实分布
    assert cm.sum(axis=0).tolist() == [1, 3, 1]  # 预测分布


def test_classification_report_contains_all_sections():
    """分类报告包含 accuracy / macro avg / weighted avg 三段。"""
    y_true = ["a", "b", "a", "c"]
    y_pred = ["a", "b", "b", "c"]
    report = M.classification_report(y_true, y_pred)
    assert "accuracy" in report
    assert "macro avg" in report
    assert "weighted avg" in report
    assert "precision" in report


def test_error_metrics():
    """MSE / RMSE / MAE / R² 与手算一致。"""
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    assert M.mse(y_true, y_pred) == pytest.approx(0.375)
    assert M.rmse(y_true, y_pred) == pytest.approx(np.sqrt(0.375))
    assert M.mae(y_true, y_pred) == pytest.approx(0.5)
    assert M.r2_score(y_true, y_true) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. 交叉验证
# ---------------------------------------------------------------------------
def test_kfold_covers_each_sample_exactly_once():
    """K 折验证中，每个样本在验证集中恰好出现一次。"""
    X = np.zeros((23, 2))
    counter = np.zeros(23, dtype=int)
    for _, val_idx in KFold(n_splits=5, random_state=0).split(X):
        counter[val_idx] += 1
    assert np.all(counter == 1)


def test_stratified_kfold_keeps_class_ratio():
    """分层 K 折的每个验证集都保持类别比例。"""
    y = np.array([0] * 60 + [1] * 40)
    X = np.zeros((100, 2))
    for _, val_idx in StratifiedKFold(n_splits=5, random_state=0).split(X, y):
        assert abs(y[val_idx].mean() - 0.4) < 1e-9


def test_cross_val_score_returns_one_score_per_fold():
    """交叉验证返回每折一个得分，且取值在 [0, 1]。"""
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.5, (30, 2)), rng.normal(5, 0.5, (30, 2))])
    y = np.array([0] * 30 + [1] * 30)
    scores = cross_val_score(lambda: KNNClassifier(k=3), X, y,
                             cv=StratifiedKFold(n_splits=5, random_state=0))
    assert scores.shape == (5,)
    assert np.all((scores >= 0) & (scores <= 1))
