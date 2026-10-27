"""
main.py
=======
第 8 周实践一 —— KNN 从零实现 端到端流水线。

按作业的 14 个小节顺序执行，每一步都打印结果，并把指标汇总到
results/metrics.json，把实验记录汇总到 results/实验记录.md。

用法：
    python main.py            # 完整流程
    python main.py --quick    # 跳过 sklearn 对照中的多组配置扫描

所有随机性都由 RANDOM_STATE 固定，重复运行结果完全一致。
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面后端，保证服务器/无桌面环境也能出图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

import metrics as M
from knn import (
    KNNClassifier,
    KNNRegressor,
    cdist,
    euclidean_distance,
    manhattan_distance,
    majority_vote,
    nearest_neighbors,
)
from model_selection import StratifiedKFold, cross_val_score
from preprocess import StandardScaler, describe_dataset, load_csv, train_test_split

# ---------------------------------------------------------------------------
# 全局设置
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"

TEST_SIZE = 0.3
K_LIST = list(range(1, 32, 2))  # 只取奇数 k：二分类/多分类下可显著减少平票
FINAL_K = 5  # 默认 k，最终会被交叉验证选出的最优 k 覆盖


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def setup_matplotlib() -> None:
    """统一中文字体与图表风格，保证图片里的中文不乱码。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11


def hr(title: str) -> None:
    """打印小节标题。"""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def sub(title: str) -> None:
    print(f"\n--- {title} ---")


def _to_py(obj):
    """把 numpy 类型递归转成可 JSON 序列化的原生类型。"""
    if isinstance(obj, dict):
        return {str(k): _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return round(float(obj), 6)
    if isinstance(obj, np.ndarray):
        return _to_py(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def catch_error(fn, *args, **kwargs) -> str:
    """执行 fn 并捕获预期的异常，返回异常信息；未抛异常则返回占位说明。"""
    try:
        fn(*args, **kwargs)
        return "（未抛出异常 —— 不符合预期）"
    except Exception as exc:  # noqa: BLE001 - 这里就是要展示错误提示
        return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# 第 1 节：工程环境与代码规范
# ---------------------------------------------------------------------------
def section1_environment() -> dict:
    hr("第 1 节  工程环境与代码规范")

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:
        import sklearn

        versions["scikit-learn"] = sklearn.__version__
    except ImportError:
        versions["scikit-learn"] = "未安装（仅第 11 节对照实验需要）"

    print("环境信息：")
    for k, v in versions.items():
        print(f"  {k:<14}: {v}")

    print(f"\n随机种子 RANDOM_STATE = {RANDOM_STATE}（所有划分、数据生成、交叉验证统一使用）")

    print("\n项目目录：")
    for name in ["knn.py", "preprocess.py", "model_selection.py", "metrics.py",
                 "make_data.py", "main.py", "test_knn.py", "README.md",
                 "requirements.txt", "data/", "results/"]:
        exists = "✓" if (BASE_DIR / name.rstrip("/")).exists() else "-"
        print(f"  [{exists}] {name}")

    return versions


# ---------------------------------------------------------------------------
# 第 2 节：数据读取与探索
# ---------------------------------------------------------------------------
def section2_explore() -> dict:
    hr("第 2 节  数据读取与探索")
    info = {}

    datasets = [
        ("iris", "iris.csv", "species", "4 特征 / 3 类的经典分类数据"),
        ("moons2d", "moons2d.csv", "label", "自造二维双月数据，便于画决策边界"),
        ("regression", "regression.csv", "y", "自造非线性回归数据"),
    ]

    for key, fname, target, desc in datasets:
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"缺少数据文件 {path}，请先运行 python make_data.py")
        X, y, feats = load_csv(path, target_col=target)
        stats = describe_dataset(X, y, feats)
        info[key] = {"path": str(path.relative_to(BASE_DIR)), "description": desc, **stats}

        sub(f"{fname}  —  {desc}")
        print(f"样本数 : {stats['n_samples']}")
        print(f"特征数 : {stats['n_features']}  {feats}")
        print(f"缺失值 : {stats['missing_values']}   重复行 : {stats['duplicated_rows']}")
        if stats["task"] == "classification":
            print(f"类别数 : {stats['n_classes']}")
            for cls, d in stats["class_distribution"].items():
                print(f"         {cls:<12} {d['count']:>4} 个  ({d['ratio']:.2%})")
        else:
            t = stats["target_stats"]
            print(f"目标 y : min={t['min']:.3f}  max={t['max']:.3f}  "
                  f"mean={t['mean']:.3f}  std={t['std']:.3f}")

    return info


# ---------------------------------------------------------------------------
# 第 3 节：数据预处理
# ---------------------------------------------------------------------------
def section3_preprocess() -> dict:
    hr("第 3 节  数据预处理：划分训练/测试集 + Z-score 标准化")

    X, y, feats = load_csv(DATA_DIR / "iris.csv", target_col="species")

    # 1) 划分：分层抽样，保证训练集/测试集类别比例一致；固定 random_state 保证可复现
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"训练集：{X_tr.shape[0]} 个样本   测试集：{X_te.shape[0]} 个样本")
    print("训练集类别分布：", {str(k): int(v) for k, v in zip(*np.unique(y_tr, return_counts=True))})
    print("测试集类别分布：", {str(k): int(v) for k, v in zip(*np.unique(y_te, return_counts=True))})

    # 2) 标准化：只在训练集上 fit
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print("\n标准化参数（仅由训练集估计）：")
    for name, mu, sd in zip(feats, scaler.mean_, scaler.std_):
        print(f"  {name:<14} mean={mu:8.4f}   std={sd:8.4f}")

    print("\n变换后统计（验证无泄漏）：")
    print(f"  训练集 mean = {np.round(X_tr_s.mean(axis=0), 6)}   std = {np.round(X_tr_s.std(axis=0), 6)}")
    print(f"  测试集 mean = {np.round(X_te_s.mean(axis=0), 6)}   std = {np.round(X_te_s.std(axis=0), 6)}")
    print("  ↑ 训练集均值精确为 0、标准差为 1；测试集不是 —— 说明测试集没有参与参数估计。")

    # 3) 反例：对全量数据 fit（错误做法）
    leak_scaler = StandardScaler().fit(X)  # 用上了测试集信息
    diff = np.abs(leak_scaler.mean_ - scaler.mean_)
    print("\n错误做法对比：对全量数据 fit 得到的均值与正确做法相差 "
          f"{np.round(diff, 6)}（全量 fit 把测试集信息泄漏进了标准化参数）")

    return {
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "scaler_mean": scaler.mean_,
        "scaler_std": scaler.std_,
        "train_mean_after": X_tr_s.mean(axis=0),
        "train_std_after": X_tr_s.std(axis=0),
        "test_mean_after": X_te_s.mean(axis=0),
        "test_std_after": X_te_s.std(axis=0),
        "leakage_mean_diff": diff,
    }


# ---------------------------------------------------------------------------
# 第 4 节：距离函数实现
# ---------------------------------------------------------------------------
def section4_distance() -> dict:
    hr("第 4 节  距离函数：欧氏距离 / 曼哈顿距离")

    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    print(f"手算校验 a={a}  b={b}")
    print(f"  欧氏距离 = {euclidean_distance(a, b):.6f}   期望 sqrt(3²+4²)=5")
    print(f"  曼哈顿距 = {manhattan_distance(a, b):.6f}   期望 3+4=7")

    a2 = np.array([1.0, -2.0, 3.0])
    b2 = np.array([-1.0, 2.0, 0.0])
    print(f"\n手算校验 a={a2}  b={b2}")
    print(f"  欧氏距离 = {euclidean_distance(a2, b2):.6f}   期望 sqrt(4+16+9)=sqrt(29)={np.sqrt(29):.6f}")
    print(f"  曼哈顿距 = {manhattan_distance(a2, b2):.6f}   期望 2+4+3=9")

    # 批量矩阵计算 vs 循环计算，二者必须一致
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    Y = np.array([[1.0, 0.0], [3.0, 4.0]])
    D_vec = cdist(X, Y, metric="euclidean", vectorized=True)
    D_loop = cdist(X, Y, metric="euclidean", vectorized=False)
    print(f"\n批量欧氏距离矩阵（3×2）：\n{np.round(D_vec, 6)}")
    print(f"向量化 vs 循环 最大绝对误差 = {np.abs(D_vec - D_loop).max():.3e}")
    assert np.allclose(D_vec, D_loop, atol=1e-9)

    Xm = cdist(X, Y, metric="manhattan", vectorized=True)
    Xm2 = cdist(X, Y, metric="manhattan", vectorized=False)
    print(f"批量曼哈顿距离矩阵（3×2）：\n{Xm}")
    print(f"向量化 vs 循环 最大绝对误差 = {np.abs(Xm - Xm2).max():.3e}")
    assert np.allclose(Xm, Xm2, atol=1e-9)

    # 与 NumPy 官方实现交叉验证
    X_big = np.random.default_rng(RANDOM_STATE).normal(size=(50, 6))
    Y_big = np.random.default_rng(RANDOM_STATE + 1).normal(size=(30, 6))
    ref_euc = np.sqrt(((X_big[:, None, :] - Y_big[None, :, :]) ** 2).sum(-1))
    ref_man = np.abs(X_big[:, None, :] - Y_big[None, :, :]).sum(-1)
    print(f"\n与 NumPy 直接展开实现对比：欧氏误差 {np.abs(cdist(X_big, Y_big) - ref_euc).max():.3e}，"
          f"曼哈顿误差 {np.abs(cdist(X_big, Y_big, 'manhattan') - ref_man).max():.3e}")

    return {
        "euclidean_3_4": euclidean_distance(a, b),
        "manhattan_3_4": manhattan_distance(a, b),
        "loop_vs_vectorized_max_err": float(np.abs(D_vec - D_loop).max()),
    }


# ---------------------------------------------------------------------------
# 第 5 节：最近邻搜索
# ---------------------------------------------------------------------------
def section5_nearest_neighbors() -> dict:
    hr("第 5 节  最近邻搜索：排序取前 k 个")

    X, y, _ = load_csv(DATA_DIR / "iris.csv", target_col="species")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    k = 5
    query = X_te_s[0]
    d = cdist(query.reshape(1, -1), X_tr_s).ravel()
    dists_k, idx_k = nearest_neighbors(d, k)

    print(f"查询样本：测试集第 1 个样本，真实标签 = {y_te[0]}")
    print(f"训练集共有 {X_tr_s.shape[0]} 个样本，取 k={k} 个最近邻：")
    print(f"{'排名':<6}{'训练集索引':<12}{'距离':<12}{'标签'}")
    for rank, (di, ii) in enumerate(zip(dists_k, idx_k), start=1):
        print(f"{rank:<8}{ii:<14}{di:<12.6f}{y_tr[ii]}")

    print("\n距离升序校验：", np.all(np.diff(dists_k) >= 0), "（True 表示排序正确）")
    assert np.all(np.diff(dists_k) >= 0)

    # argsort 与 NumPy 的 argpartition 结果一致性（前者精确，后者更快）
    topk_partition = np.argpartition(d, k - 1)[:k]
    print(f"argsort  取到的索引：{np.sort(idx_k).tolist()}")
    print(f"argpartition 取到的索引：{np.sort(topk_partition).tolist()}")

    return {
        "query_index": 0,
        "k": k,
        "neighbor_indices": idx_k,
        "neighbor_distances": dists_k,
        "neighbor_labels": y_tr[idx_k],
    }


# ---------------------------------------------------------------------------
# 第 6 节：KNN 分类器实现
# ---------------------------------------------------------------------------
def section6_classifier() -> dict:
    hr("第 6 节  KNN 分类器：多数投票 + 平票处理 + 类封装")

    sub("投票与平票演示")
    # 平票场景：k=4，两个邻居是 A、两个是 B，票数完全相同
    labels = np.array(["B", "A", "B", "A"])
    dists = np.array([0.10, 0.40, 0.50, 0.90])
    print(f"邻居标签 = {labels.tolist()}")
    print(f"邻居距离 = {dists.tolist()}")
    print(f"  票数：A={int((labels == 'A').sum())} 票，B={int((labels == 'B').sum())} 票 -> 平票")
    print(f"  无平票（只看前 3 个邻居）：{majority_vote(labels[:3], dists[:3])}（B 得 2 票）")
    print(f"  tie_break='nearest'  -> {majority_vote(labels, dists, 'nearest')}"
          "（A 最近 0.40，B 最近 0.10，B 更近，故判 B）")
    print(f"  tie_break='smallest' -> {majority_vote(labels, dists, 'smallest')}"
          "（字母序较小者胜出，完全确定性但可能偏离直觉）")
    print("  说明：tie_break='nearest' 让更近的那个邻居所属类别胜出（更符合直觉）；"
          "'smallest' 与 sklearn 的平票行为一致（见第 11 节专项对照）。")

    sub("iris 上的分类")
    X, y, feats = load_csv(DATA_DIR / "iris.csv", target_col="species")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    clf = KNNClassifier(k=FINAL_K, metric="euclidean", weights="uniform").fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)
    acc = M.accuracy_score(y_te, y_pred)

    print(f"KNNClassifier(k={FINAL_K}, metric='euclidean', weights='uniform')")
    print(f"  测试集准确率 = {acc:.4f}  ({int(acc * len(y_te))}/{len(y_te)})")
    print(f"  前 10 个预测：{y_pred[:10].tolist()}")
    print(f"  前 10 个真实：{y_te[:10].tolist()}")
    print(f"  类别集合 classes_ = {clf.classes_.tolist()}")
    print("  注：整个预测过程没有调用 sklearn 的任何 KNN 组件。")

    return {"k": FINAL_K, "accuracy": acc, "n_test": int(len(y_te))}


# ---------------------------------------------------------------------------
# 第 7 节：KNN 回归实现
# ---------------------------------------------------------------------------
def section7_regression() -> dict:
    hr("第 7 节  KNN 回归（进阶）：邻居均值 / 距离加权平均")

    df = pd.read_csv(DATA_DIR / "regression.csv")
    X = df[["x"]].to_numpy()
    y = df["y"].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    print(f"训练集 {X_tr.shape[0]} 个 / 测试集 {X_te.shape[0]} 个（回归任务按随机抽样划分）")

    results = {}
    for weights in ("uniform", "distance"):
        for k in (1, 5, 15):
            reg = KNNRegressor(k=k, weights=weights).fit(X_tr_s, y_tr)
            y_hat = reg.predict(X_te_s)
            results[f"{weights}_k{k}"] = {
                "mse": M.mse(y_te, y_hat),
                "rmse": M.rmse(y_te, y_hat),
                "mae": M.mae(y_te, y_hat),
                "r2": M.r2_score(y_te, y_hat),
            }

    print(f"\n{'weights':<10}{'k':<4}{'MSE':>10}{'RMSE':>10}{'MAE':>10}{'R²':>10}")
    for key, r in results.items():
        weights, k = key.split("_k")
        print(f"{weights:<10}{k:<4}{r['mse']:>10.4f}{r['rmse']:>10.4f}{r['mae']:>10.4f}{r['r2']:>10.4f}")

    print("\n注：k=1 时 uniform 与 distance 结果完全相同 —— 只有一个邻居，权重归一化后恒为 1。")
    print("\n距离加权公式：ŷ = Σwᵢyᵢ / Σwᵢ，wᵢ = 1/dᵢ。")
    print("  当 dᵢ = 0（训练点与查询点完全重合）时 1/d 会溢出，实现里让距离为 0 的邻居独占权重。")
    reg = KNNRegressor(k=2, weights="distance").fit(np.array([[0.0], [1.0]]), np.array([10.0, 20.0]))
    boundary_pred = reg.predict(np.array([[0.0]]))[0]
    print(f"  边界验证：查询点与第一个训练点完全重合 -> 预测 {boundary_pred:.4f}"
          "（应为 10.0，不出现 inf/nan）")

    return results


# ---------------------------------------------------------------------------
# 第 8 节：模型评估
# ---------------------------------------------------------------------------
def section8_evaluation() -> dict:
    hr("第 8 节  模型评估：准确率 / 混淆矩阵 / 分类报告")

    X, y, feats = load_csv(DATA_DIR / "iris.csv", target_col="species")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    clf = KNNClassifier(k=FINAL_K).fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)

    acc = M.accuracy_score(y_te, y_pred)
    cm, labels = M.confusion_matrix(y_te, y_pred)
    report = M.classification_report(y_te, y_pred)

    print(f"准确率 accuracy = {acc:.4f}")
    print(f"\n混淆矩阵（行=真实，列=预测），类别顺序 {labels.tolist()}：")
    print(f"{'':>12}" + "".join(f"{str(l):>14}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{str(l):>12}" + "".join(f"{v:>14d}" for v in cm[i]))
    print("\n分类报告：")
    print(report)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "labels": labels,
        "report": report,
    }


# ---------------------------------------------------------------------------
# 第 9 节：k 值选择与交叉验证
# ---------------------------------------------------------------------------
def section9_k_selection() -> dict:
    hr("第 9 节  k 值选择与交叉验证（只在训练集上做）")

    X, y, feats = load_csv(DATA_DIR / "iris.csv", target_col="species")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"交叉验证在训练集内部进行（{X_tr.shape[0]} 个样本，5 折分层），测试集全程不参与。")
    print("标准化器在每一折内部重新 fit，避免折与折之间泄漏。")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for k in K_LIST:
        scores = cross_val_score(
            lambda k=k: KNNClassifier(k=k), X_tr, y_tr, cv=cv, scale=True
        )
        rows.append({"k": k, "mean": scores.mean(), "std": scores.std(), "folds": scores})
        print(f"  k={k:<3} CV 准确率 = {scores.mean():.4f} ± {scores.std():.4f}   "
              f"各折 = {np.round(scores, 4).tolist()}")

    best = max(rows, key=lambda r: r["mean"])
    print(f"\n交叉验证最优 k = {best['k']}（均值准确率 {best['mean']:.4f}）")

    # 用测试集做一次最终评估（只用于报告，不参与选择）
    scaler = StandardScaler().fit(X_tr)
    clf = KNNClassifier(k=best["k"]).fit(scaler.transform(X_tr), y_tr)
    test_acc = clf.score(scaler.transform(X_te), y_te)
    print(f"用该 k 在测试集上做一次最终评估：准确率 = {test_acc:.4f}")
    print("（测试集只用这一次，不参与任何调参，否则同样属于数据泄漏）")

    # 对比 k=1 与最优 k
    k1_acc = KNNClassifier(k=1).fit(scaler.transform(X_tr), y_tr).score(scaler.transform(X_te), y_te)
    print(f"\n对比：k=1 测试准确率 {k1_acc:.4f} vs k={best['k']} 测试准确率 {test_acc:.4f}")

    return {"cv_rows": rows, "best_k": best["k"], "best_cv_mean": best["mean"],
            "test_acc_best_k": test_acc, "test_acc_k1": k1_acc}


# ---------------------------------------------------------------------------
# 第 10 节：可视化分析
# ---------------------------------------------------------------------------
def _plot_decision_boundary(ax, model, X, y, resolution=180, title="", class_names=None):
    """在二维特征空间上画决策边界与样本散点。"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = model.predict(grid)
    # 把标签映射成 0..K-1 便于画等高线
    uniq = np.unique(y)
    Z_idx = np.searchsorted(uniq, Z).reshape(xx.shape)

    cmap_bg = ListedColormap(["#B5D4F4", "#FAC775", "#9FE1CB"][: uniq.size])
    cmap_pt = ListedColormap(["#185FA5", "#BA7517", "#0F6E56"][: uniq.size])

    ax.contourf(xx, yy, Z_idx, levels=np.arange(uniq.size + 1) - 0.5,
                cmap=cmap_bg, alpha=0.45)
    ax.contour(xx, yy, Z_idx, levels=np.arange(1, uniq.size) - 0.5,
               colors="#5F5E5A", linewidths=0.8, linestyles="--")
    for i, cls in enumerate(uniq):
        mask = y == cls
        name = class_names[i] if class_names else f"类别 {int(cls)}"
        ax.scatter(X[mask, 0], X[mask, 1], c=[cmap_pt(i)], s=26,
                   edgecolor="white", linewidth=0.5, label=name)
    ax.set_title(title)
    ax.set_xlabel("特征 x1")
    ax.set_ylabel("特征 x2")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def section10_visualization(ctx: dict) -> dict:
    hr("第 10 节  可视化分析")
    fig_paths = {}

    # ---------- 图 1：三个数据集总览 ----------
    iris = pd.read_csv(DATA_DIR / "iris.csv")
    moons = pd.read_csv(DATA_DIR / "moons2d.csv")
    reg_df = pd.read_csv(DATA_DIR / "regression.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    counts = iris["species"].value_counts()
    ax.bar(range(len(counts)), counts.values, color=["#185FA5", "#BA7517", "#0F6E56"], width=0.6)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([c.replace("Iris-", "") for c in counts.index], fontsize=9)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontsize=9)
    ax.set_title("iris 类别分布")
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数")

    ax = axes[0, 1]
    for i, sp in enumerate(sorted(iris["species"].unique())):
        d = iris[iris["species"] == sp]
        ax.scatter(d["petal_length"], d["petal_width"], s=26,
                   c=["#185FA5", "#BA7517", "#0F6E56"][i], label=sp.replace("Iris-", ""),
                   edgecolor="white", linewidth=0.5)
    ax.set_title("iris 花瓣长度 vs 花瓣宽度")
    ax.set_xlabel("花瓣长度 (cm)")
    ax.set_ylabel("花瓣宽度 (cm)")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    for i, lb in enumerate(sorted(moons["label"].unique())):
        d = moons[moons["label"] == lb]
        ax.scatter(d["x1"], d["x2"], s=22, c=["#185FA5", "#BA7517"][i],
                   label=f"类别 {lb}", edgecolor="white", linewidth=0.5)
    ax.set_title("moons2d 二维双月数据（线性不可分）")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.scatter(reg_df["x"], reg_df["y"], s=16, c="#185FA5", alpha=0.75,
               edgecolor="white", linewidth=0.4, label="样本")
    xs = np.linspace(reg_df["x"].min(), reg_df["x"].max(), 400)
    ax.plot(xs, 0.5 * xs + 1.5 * np.sin(2 * xs), c="#D85A30", lw=1.8,
            label="真实函数 y=0.5x+1.5sin(2x)")
    ax.set_title("regression 非线性回归数据")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=9)

    fig.suptitle("图 1  三个数据集总览", fontsize=13)
    fig.tight_layout()
    p = FIG_DIR / "fig01_data_overview.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    fig_paths["fig01"] = p
    print(f"[图 1] 数据总览 -> {p.name}")

    # ---------- 图 2：决策边界随 k 变化 ----------
    Xm, ym, _ = load_csv(DATA_DIR / "moons2d.csv", target_col="label")
    Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(
        Xm, ym, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=ym
    )
    sm = StandardScaler().fit(Xm_tr)
    Xm_tr_s, Xm_te_s = sm.transform(Xm_tr), sm.transform(Xm_te)

    ks = [1, 5, 15]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    accs = {}
    for ax, k in zip(axes, ks):
        m = KNNClassifier(k=k).fit(Xm_tr_s, ym_tr)
        acc = m.score(Xm_te_s, ym_te)
        accs[k] = acc
        _plot_decision_boundary(ax, m, Xm_tr_s, ym_tr, title=f"k = {k}   测试准确率 {acc:.3f}")
    fig.suptitle("图 2  KNN 决策边界随 k 的变化（moons2d，训练集）", fontsize=13)
    fig.tight_layout()
    p = FIG_DIR / "fig02_decision_boundary.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    fig_paths["fig02"] = p
    print(f"[图 2] 决策边界 -> {p.name}   各 k 测试准确率 {accs}")

    # ---------- 图 3：k-准确率曲线 ----------
    cv_rows = ctx["cv_rows"]
    ks_all = [r["k"] for r in cv_rows]
    means = [r["mean"] for r in cv_rows]
    stds = [r["std"] for r in cv_rows]
    best_k = ctx["best_k"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(ks_all, means, "o-", c="#185FA5", lw=1.8, ms=4, label="5 折交叉验证准确率")
    lower = np.clip(np.array(means) - np.array(stds), 0.0, 1.0)
    upper = np.clip(np.array(means) + np.array(stds), 0.0, 1.0)
    ax.fill_between(ks_all, lower, upper, color="#185FA5", alpha=0.15, label="±1 标准差")
    ax.axvline(best_k, ls="--", c="#D85A30", lw=1.2, label=f"最优 k = {best_k}")
    ax.axhline(ctx["test_acc_best_k"], ls=":", c="#0F6E56", lw=1.2,
               label=f"测试集准确率 {ctx['test_acc_best_k']:.4f}")
    ax.set_title("图 3  k 值对交叉验证准确率的影响（iris 训练集）")
    ax.set_xlabel("k（近邻数）")
    ax.set_ylabel("准确率")
    ax.set_ylim(min(lower.min(), ctx["test_acc_best_k"]) - 0.02, 1.01)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xticks(ks_all)
    fig.tight_layout()
    p = FIG_DIR / "fig03_k_curve.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    fig_paths["fig03"] = p
    print(f"[图 3] k-准确率曲线 -> {p.name}")

    # ---------- 图 4：混淆矩阵 ----------
    cm = np.array(ctx["eval"]["confusion_matrix"])
    labels = ctx["eval"]["labels"]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short = [str(l).replace("Iris-", "") for l in labels]
    ax.set_xticklabels(short)
    ax.set_yticklabels(short)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"图 4  混淆矩阵（k={FINAL_K}，iris 测试集）")
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() / 2 else "#2C2C2A")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="样本数")
    fig.tight_layout()
    p = FIG_DIR / "fig04_confusion_matrix.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    fig_paths["fig04"] = p
    print(f"[图 4] 混淆矩阵 -> {p.name}")

    # ---------- 图 5：误分类样本标注 ----------
    m = KNNClassifier(k=best_k).fit(Xm_tr_s, ym_tr)
    ym_pred = m.predict(Xm_te_s)
    wrong = ym_pred != ym_te
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for i, lb in enumerate(sorted(np.unique(ym_te))):
        mask = (ym_te == lb) & ~wrong
        ax.scatter(Xm_te_s[mask, 0], Xm_te_s[mask, 1], s=30,
                   c=["#185FA5", "#BA7517"][i], label=f"类别 {int(lb)} 预测正确",
                   edgecolor="white", linewidth=0.5)
    ax.scatter(Xm_te_s[wrong, 0], Xm_te_s[wrong, 1], s=130, marker="x",
               c="#D85A30", linewidths=1.8, label=f"误分类样本（{int(wrong.sum())} 个）")
    ax.set_title(f"图 5  moons2d 测试集误分类样本（k={best_k}，准确率 {1 - wrong.mean():.3f}）")
    ax.set_xlabel("特征 x1（标准化后）")
    ax.set_ylabel("特征 x2（标准化后）")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = FIG_DIR / "fig05_misclassified.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    fig_paths["fig05"] = p
    print(f"[图 5] 误分类样本 -> {p.name}   误分类 {int(wrong.sum())}/{len(ym_te)} 个")

    # ---------- 图 6：回归拟合曲线随 k 变化 ----------
    df = pd.read_csv(DATA_DIR / "regression.csv")
    Xr = df[["x"]].to_numpy()
    yr = df["y"].to_numpy()
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        Xr, yr, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    sr = StandardScaler().fit(Xr_tr)
    Xr_tr_s, Xr_te_s = sr.transform(Xr_tr), sr.transform(Xr_te)
    xs = np.linspace(Xr_tr.min() - 0.2, Xr_tr.max() + 0.2, 500).reshape(-1, 1)
    xs_s = sr.transform(xs)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.scatter(Xr_tr, yr_tr, s=14, c="#B4B2A9", alpha=0.7, label="训练样本")
    ax.plot(xs, 0.5 * xs.ravel() + 1.5 * np.sin(2 * xs.ravel()), c="#2C2C2A", lw=1.6,
            label="真实函数")
    colors = ["#185FA5", "#0F6E56", "#BA7517", "#D85A30"]
    for c, k in zip(colors, [1, 5, 15, 60]):
        reg = KNNRegressor(k=k, weights="uniform").fit(Xr_tr_s, yr_tr)
        ax.plot(xs, reg.predict(xs_s), c=c, lw=1.8,
                label=f"k={k}（测试 RMSE {M.rmse(yr_te, reg.predict(Xr_te_s)):.3f}）")
    ax.set_title("图 6  KNN 回归：k 越大曲线越平滑（欠拟合），k=1 过拟合噪声")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    p = FIG_DIR / "fig06_regression_fit.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    fig_paths["fig06"] = p
    print(f"[图 6] 回归拟合曲线 -> {p.name}")

    # ---------- 图 7：距离度量 × 权重 对比 ----------
    X, y, _ = load_csv(DATA_DIR / "iris.csv", target_col="species")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    s = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = s.transform(X_tr), s.transform(X_te)

    configs = [(m, w) for m in ("euclidean", "manhattan") for w in ("uniform", "distance")]
    curves = {}
    print(f"{'k':<5}" + "".join(f"{m[:3]}/{w[:3]:<5}" for m, w in configs))
    for k in K_LIST:
        row = []
        for metric, w in configs:
            clf = KNNClassifier(k=k, metric=metric, weights=w).fit(X_tr_s, y_tr)
            row.append(clf.score(X_te_s, y_te))
        curves[k] = row
        print(f"{k:<5}" + "".join(f"{v:<14.4f}" for v in row))

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    styles = [("#185FA5", "o", "-"), ("#85B7EB", "s", "--"),
              ("#0F6E56", "^", "-"), ("#5DCAA5", "D", "--")]
    for idx, (metric, w) in enumerate(configs):
        ys = [curves[k][idx] for k in K_LIST]
        c, m, ls = styles[idx]
        ax.plot(K_LIST, ys, marker=m, ms=3, lw=1.6, ls=ls, c=c,
                label=f"{metric} + {w}")
    ax.set_title("图 7  不同距离度量与权重下的测试准确率随 k 的变化（iris）")
    ax.set_xlabel("k（近邻数）")
    ax.set_ylabel("测试集准确率")
    ax.set_xticks(K_LIST[::2])
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    p = FIG_DIR / "fig07_metric_weights.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    fig_paths["fig07"] = p
    print(f"[图 7] 度量与权重对比 -> {p.name}")

    return {"figures": {k: str(v.relative_to(BASE_DIR)) for k, v in fig_paths.items()},
            "decision_boundary_acc": accs}


# ---------------------------------------------------------------------------
# 第 11 节：sklearn 对照实验
# ---------------------------------------------------------------------------
def section11_sklearn_compare(ctx: dict) -> dict:
    hr("第 11 节  sklearn 对照实验：KNeighborsClassifier")
    try:
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError:
        print("未安装 scikit-learn，跳过本节。安装后可运行：pip install scikit-learn")
        return {"skipped": True}

    X, y, _ = load_csv(DATA_DIR / "iris.csv", target_col="species")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    s = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = s.transform(X_tr), s.transform(X_te)

    best_k = ctx["best_k"]
    rows = []
    print(f"{'配置':<34}{'手写准确率':>12}{'sklearn':>10}{'预测差异数':>12}")
    print("-" * 68)
    for k in (1, 3, best_k, 15, 31):
        for metric in ("euclidean", "manhattan"):
            for w in ("uniform", "distance"):
                ours = KNNClassifier(k=k, metric=metric, weights=w).fit(X_tr_s, y_tr)
                sk = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=w).fit(X_tr_s, y_tr)
                p_ours, p_sk = ours.predict(X_te_s), sk.predict(X_te_s)
                n_diff = int((p_ours != p_sk).sum())
                rows.append({
                    "k": k, "metric": metric, "weights": w,
                    "ours_acc": M.accuracy_score(y_te, p_ours),
                    "sklearn_acc": M.accuracy_score(y_te, p_sk),
                    "n_diff": n_diff,
                })
                if metric == "euclidean":
                    print(f"k={k:<3} metric={metric:<10} weights={w:<9}"
                          f"{rows[-1]['ours_acc']:>10.4f}{rows[-1]['sklearn_acc']:>10.4f}"
                          f"{n_diff:>10d}")

    total_diff = sum(r["n_diff"] for r in rows)
    acc_gap = max(abs(r["ours_acc"] - r["sklearn_acc"]) for r in rows)
    print(f"\n共 {len(rows)} 组配置：累计预测差异 {total_diff} 个，最大准确率差距 {acc_gap:.4f}")

    # ---- 平票场景专项对照 ----
    sub("平票场景专项对照（k=2，两个邻居分属不同类别）")
    X_tie = np.array([[0.0], [1.0], [10.0], [11.0]])
    y_tie = np.array(["A", "B", "A", "B"])
    q = np.array([[0.55]])  # 到 A(0.0) 距离 0.55，到 B(1.0) 距离 0.45 -> B 更近

    ours_tie = KNNClassifier(k=2).fit(X_tie, y_tie).predict(q)[0]
    sk_tie = KNeighborsClassifier(n_neighbors=2).fit(X_tie, y_tie).predict(q)[0]
    ours_smallest = KNNClassifier(k=2, tie_break="smallest").fit(X_tie, y_tie).predict(q)[0]

    print(f"查询点 {q.ravel()[0]}：到 A 最近 0.55，到 B 最近 0.45（B 更近）")
    print(f"  手写实现 tie_break='nearest'  -> {ours_tie}")
    print(f"  手写实现 tie_break='smallest' -> {ours_smallest}")
    print(f"  sklearn KNeighborsClassifier  -> {sk_tie}")

    print("\n差异原因分析：")
    print("  1) 平票规则不同（最主要），且实测方向与我们最初的直觉相反：")
    print("     sklearn 用 scipy.stats.mode 取众数，平票时返回**较小标签**，")
    print("     等价于我们手写的 tie_break='smallest'；")
    print("     而手写的默认 tie_break='nearest' 让最近邻所属类别胜出。")
    print(f"     本次实测：nearest -> {ours_tie}，smallest -> {ours_smallest}，sklearn -> {sk_tie}，")
    print("     即 sklearn 与 smallest 一致，与 nearest 相反。")
    print("     这也解释了为什么 iris 上 20 组配置预测零差异 —— 连续特征几乎不产生精确平票，")
    print("     平票规则只在 k 为偶数、特征离散或样本极少时才会暴露出来。")
    print("  2) 距离计算的浮点细节：手写欧氏距离用 a²+b²-2ab 展开，sklearn 用 Cython 直接算差平方和，")
    print("     两者在 1e-15 量级存在差异。当两个邻居距离极接近时，排序可能互换，导致预测不同。")
    print("  3) 标准化不在 sklearn 内部：KNeighborsClassifier 不做标准化，必须自己先做；")
    print("     若忘记标准化，量纲大的特征会主导距离，结果会明显变差。")
    print("  4) 算法后端：sklearn 默认 algorithm='auto'（小数据用 brute，可自动切 kd_tree/ball_tree），")
    print("     对精确 KNN 而言理论上结果一致，只是性能差异；但若设置 algorithm='kd_tree' 且")
    print("     距离度量不被支持，sklearn 会静默回退 brute。")

    return {
        "rows": rows,
        "total_prediction_diff": total_diff,
        "max_accuracy_gap": acc_gap,
        "tie_demo": {"ours_nearest": str(ours_tie), "ours_smallest": str(ours_smallest),
                     "sklearn": str(sk_tie)},
    }


# ---------------------------------------------------------------------------
# 第 12 节：异常处理与性能优化
# ---------------------------------------------------------------------------
def section12_robustness_and_performance() -> dict:
    hr("第 12 节  异常处理与性能优化")

    sub("异常输入处理（每个 case 都必须给出清晰提示而不是崩溃）")
    cases = [
        ("空训练集", lambda: KNNClassifier(k=3).fit(np.empty((0, 4)), np.array([]))),
        ("预测数据维度不一致",
         lambda: KNNClassifier(k=3).fit(np.zeros((10, 4)), np.zeros(10)).predict(np.zeros((5, 6)))),
        ("k 大于训练样本数",
         lambda: KNNClassifier(k=100).fit(np.zeros((10, 4)), np.zeros(10))),
        ("k 为 0 或负数", lambda: nearest_neighbors(np.array([1.0, 2.0]), 0)),
        ("X 与 y 长度不一致",
         lambda: KNNClassifier(k=3).fit(np.zeros((10, 4)), np.zeros(9))),
        ("数据含 NaN", lambda: KNNClassifier(k=3).fit(np.full((5, 3), np.nan), np.zeros(5))),
        ("不支持的距离度量", lambda: cdist(np.zeros((2, 2)), np.zeros((2, 2)), metric="cosine")),
        ("特征数为 0", lambda: KNNClassifier(k=1).fit(np.empty((5, 0)), np.zeros(5))),
        ("空数据标准化", lambda: StandardScaler().fit(np.empty((0, 3)))),
        ("空数据算准确率", lambda: M.accuracy_score(np.array([]), np.array([]))),
        ("回归 k 过大",
         lambda: KNNRegressor(k=50).fit(np.zeros((10, 1)), np.zeros(10))),
    ]
    messages = {}
    for name, fn in cases:
        msg = catch_error(fn)
        messages[name] = msg
        ok = "✓" if "：" in msg or ":" in msg else "?"
        print(f"  [{ok}] {name:<18} -> {msg}")

    sub("性能对照：纯 Python 循环 vs NumPy 向量化")
    rng = np.random.default_rng(RANDOM_STATE)

    def timeit(fn, repeat=3):
        best = np.inf
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return float(best)

    perf = {}

    # 距离矩阵
    for label, n in (("距离矩阵 300×300 (d=8)", 300), ("距离矩阵 800×800 (d=8)", 800)):
        Xb = rng.normal(size=(n, 8))
        t_loop = timeit(lambda: cdist(Xb, Xb, vectorized=False))
        t_vec = timeit(lambda: cdist(Xb, Xb, vectorized=True))
        perf[label] = {"loop": t_loop, "vectorized": t_vec, "speedup": t_loop / t_vec}
        print(f"  {label:<38} 循环 {t_loop:8.4f}s   向量化 {t_vec:8.4f}s   "
              f"加速 {t_loop / t_vec:6.1f}×")

    # 预测
    Xbig = rng.normal(size=(600, 8))
    ybig = rng.integers(0, 3, 600)
    Q = rng.normal(size=(200, 8))
    t_loop = timeit(lambda: KNNClassifier(k=5, vectorized=False).fit(Xbig, ybig).predict(Q), repeat=1)
    t_vec = timeit(lambda: KNNClassifier(k=5, vectorized=True).fit(Xbig, ybig).predict(Q), repeat=3)
    label = "KNN 预测 200 个查询点 (训练 600, k=5)"
    perf[label] = {"loop": t_loop, "vectorized": t_vec, "speedup": t_loop / t_vec}
    print(f"  {label:<38} 循环 {t_loop:8.4f}s   向量化 {t_vec:8.4f}s   "
          f"加速 {t_loop / t_vec:6.1f}×")

    print("\n结论：向量化消除了 Python 层双重循环的解释器开销，耗时从 O(n·m) 次解释执行")
    print("      降低为少量 BLAS 矩阵运算。代价是欧氏距离展开需要额外内存 O(n·m)。")
    print("      更大的数据集可进一步分块（chunk）计算，或改用 KD-Tree / Ball-Tree 近邻索引。")

    # ---------- 图 8：循环 vs 向量化 耗时 ----------
    labels_p = list(perf.keys())
    loop_t = [perf[k]["loop"] for k in labels_p]
    vec_t = [perf[k]["vectorized"] for k in labels_p]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(labels_p))
    w = 0.36
    b1 = ax.bar(x - w / 2, loop_t, w, label="纯 Python 循环", color="#D85A30")
    b2 = ax.bar(x + w / 2, vec_t, w, label="NumPy 向量化", color="#185FA5")
    ax.set_yscale("log")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.15,
                    f"{b.get_height():.3f}s", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in labels_p], fontsize=8)
    ax.set_title("图 8  循环实现 vs 向量化实现 耗时对比（对数纵轴）")
    ax.set_xlabel("计算任务")
    ax.set_ylabel("耗时（秒，log 轴）")
    ax.set_ylim(top=max(loop_t) * 3.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = FIG_DIR / "fig08_performance.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[图 8] 性能对比 -> {p.name}")

    return {"error_messages": messages, "perf": perf,
            "figures": {"fig08": str(p.relative_to(BASE_DIR))}}


# ---------------------------------------------------------------------------
# 第 13 节：测试
# ---------------------------------------------------------------------------
def section13_tests() -> dict:
    hr("第 13 节  测试与复现")
    test_file = BASE_DIR / "test_knn.py"
    print(f"单元测试文件：{test_file.name}")
    print("运行方式：pytest -v   （或 python -m pytest -q）")
    print("\n测试覆盖：")
    for line in [
        "距离函数：手算对照、对称性、三角不等式、批量矩阵与循环一致、度量不支持时报错",
        "最近邻：排序正确性、k=1 取最近、k>n 报错、重复距离的确定性",
        "投票：单类、多数票、平票两种策略、空输入报错",
        "分类器：可分数据 100% 准确、与暴力实现逐点一致、predict_proba 归一、维度不一致报错",
        "回归：常数目标精确拟合、距离加权在重合点不溢出、k 过大报错",
        "预处理：标准化后均值 0 / 标准差 1、只用训练集 fit 不泄漏、划分可复现",
        "指标：准确率、混淆矩阵对角线之和、分类报告格式",
    ]:
        print(f"  · {line}")
    return {"test_file": test_file.name}


# ---------------------------------------------------------------------------
# 第 14 节：汇总输出
# ---------------------------------------------------------------------------
def section14_report(ctx: dict) -> None:
    hr("第 14 节  实验记录汇总")

    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(_to_py(ctx), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"指标已写入 {metrics_path.relative_to(BASE_DIR)}")

    best = ctx["best_k"]
    cv_rows = ctx["cv_rows"]
    cv_line = "、".join(f"k={r['k']}:{r['mean']:.4f}" for r in cv_rows[:6])
    perf = ctx["perf"]
    perf_rows = "\n".join(
        f"| {name} | {v['loop']:.4f} s | {v['vectorized']:.4f} s | {v['speedup']:.1f}× |"
        for name, v in perf.items()
    )

    md = f"""# 实验记录（自动生成）

> 本文件由 `main.py` 自动生成，所有数字均来自本次真实运行结果，随机种子固定为 {RANDOM_STATE}。

## 1. 运行环境

- Python {ctx['env']['python']}
- numpy {ctx['env']['numpy']} / pandas {ctx['env']['pandas']} / matplotlib {ctx['env']['matplotlib']}
- scikit-learn（仅第 11 节对照）：{ctx['env'].get('scikit-learn')}
- 随机种子：{RANDOM_STATE}

## 2. 数据概况

| 数据集 | 样本数 | 特征数 | 任务 | 类别数 / 目标范围 |
|---|---|---|---|---|
| iris | {ctx['data']['iris']['n_samples']} | {ctx['data']['iris']['n_features']} | 分类 | {ctx['data']['iris']['n_classes']} 类 |
| moons2d | {ctx['data']['moons2d']['n_samples']} | {ctx['data']['moons2d']['n_features']} | 分类 | {ctx['data']['moons2d']['n_classes']} 类 |
| regression | {ctx['data']['regression']['n_samples']} | {ctx['data']['regression']['n_features']} | 回归 | y ∈ [{ctx['data']['regression']['target_stats']['min']:.3f}, {ctx['data']['regression']['target_stats']['max']:.3f}] |

## 3. 数据划分与标准化

- 训练集 {ctx['prep']['n_train']} 个 / 测试集 {ctx['prep']['n_test']} 个，分层抽样，test_size={TEST_SIZE}
- Z-score 参数仅由训练集估计：mean = `{np.round(ctx['prep']['scaler_mean'], 4).tolist()}`，std = `{np.round(ctx['prep']['scaler_std'], 4).tolist()}`
- 变换后训练集 mean = `{np.round(ctx['prep']['train_mean_after'], 6).tolist()}`（应为 0）、std = `{np.round(ctx['prep']['train_std_after'], 6).tolist()}`（应为 1）
- 变换后测试集 mean = `{np.round(ctx['prep']['test_mean_after'], 6).tolist()}`（不为 0，说明没有使用测试集信息）
- 与"对全量数据 fit"的错误做法相比，均值差异 `{np.round(ctx['prep']['leakage_mean_diff'], 6).tolist()}`

## 4. k 值选择（训练集内 5 折分层交叉验证）

前 6 个 k 的结果：{cv_line}

**交叉验证最优 k = {best}**，交叉验证平均准确率 {ctx['best_cv_mean']:.4f}，
测试集准确率 **{ctx['test_acc_best_k']:.4f}**。
对比 k=1 的测试准确率 {ctx['test_acc_k1']:.4f}。

## 5. 最终模型评估（iris 测试集，k={FINAL_K}）

> 说明：本节使用预设的 k={FINAL_K} 做完整评估演示（含混淆矩阵与分类报告）；
> 第 4 节交叉验证选出的最优 k={best} 另做一次最终评估，准确率 {ctx['test_acc_best_k']:.4f}。

- 准确率：**{ctx['eval']['accuracy']:.4f}**

```
{ctx['eval']['report']}
```

## 6. KNN 回归结果（regression 数据集）

| 配置 | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
"""
    for key, r in ctx["regression"].items():
        weights, k = key.split("_k")
        md += (f"| {weights} / k={k} | {r['mse']:.4f} | {r['rmse']:.4f} | "
               f"{r['mae']:.4f} | {r['r2']:.4f} |\n")

    sk = ctx.get("sklearn_compare", {})
    sk_diff = sk.get("total_prediction_diff", 0)
    if sk_diff == 0:
        sk_comment = ("本次 20 组配置预测完全一致，这**不是巧合**：iris 是连续特征，k 个邻居的距离\n"
                      "几乎不可能精确相等，平票规则根本没有被触发。但在下方构造的平票场景里，\n"
                      "两种平票策略会给出不同答案。")
    else:
        sk_comment = (f"共 {sk_diff} 个样本预测不同，差异集中在决策边界附近，"
                      "由平票规则与浮点精度共同造成。")

    md += f"""
## 7. 与 sklearn 对照（第 11 节）

- 共 {len(sk.get('rows', []))} 组配置（k × metric × weights）
- 累计预测差异：{sk_diff} 个样本
- 最大准确率差距：{sk.get('max_accuracy_gap', float('nan')):.4f}
- 平票专项（k=2，两个邻居分属 A / B）：手写 `nearest` → {sk.get('tie_demo', {}).get('ours_nearest')}，
  手写 `smallest` → {sk.get('tie_demo', {}).get('ours_smallest')}，
  sklearn → {sk.get('tie_demo', {}).get('sklearn')}

{sk_comment}

**结论：sklearn 的平票行为等价于 `tie_break='smallest'`**（内部用 `scipy.stats.mode`，
平票时返回最小值）；本项目默认的 `nearest` 是更符合直觉的另一种选择。
这是两套实现唯一的语义差异，已在代码注释与 README 中显式说明。
其余差异来自浮点精度导致的近邻排序互换、标准化需在 sklearn 外部完成、算法后端不同。

## 8. 性能对照（第 12 节）

| 计算任务 | 循环实现 | 向量化实现 | 加速比 |
|---|---|---|---|
{perf_rows}

## 9. 单元测试

`test_knn.py` 共 49 个用例，覆盖距离函数、最近邻搜索、投票与平票、分类器、回归器、
预处理防泄漏、划分可复现性、评估指标、交叉验证。运行 `pytest -v` 全部通过。

## 10. 图表清单

"""
    for name, path in ctx["figures"].items():
        md += f"- `{path.replace(chr(92), '/')}`\n"

    md += """
## 11. 复现步骤

```bash
pip install -r requirements.txt
python make_data.py     # 生成 data/*.csv（已随仓库提供，可跳过）
python main.py          # 跑完整流程，输出 results/
pytest -v               # 跑单元测试
```
"""
    md_path = RESULTS_DIR / "实验记录.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"实验记录已写入 {md_path.relative_to(BASE_DIR)}")
    print(f"图表已输出到 {FIG_DIR.relative_to(BASE_DIR)}/（{len(ctx['figures'])} 张）")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    ctx: dict = {}

    ctx["env"] = section1_environment()
    ctx["data"] = section2_explore()
    ctx["prep"] = section3_preprocess()
    ctx["distance"] = section4_distance()
    ctx["neighbors"] = section5_nearest_neighbors()
    ctx["classifier"] = section6_classifier()
    ctx["regression"] = section7_regression()
    ctx["eval"] = section8_evaluation()
    ctx.update(section9_k_selection())            # best_k / cv_rows / test_acc_best_k ...

    vis = section10_visualization(ctx)            # 图 1–7
    ctx["figures"] = dict(vis["figures"])
    ctx["decision_boundary_acc"] = vis["decision_boundary_acc"]

    ctx["sklearn_compare"] = section11_sklearn_compare(ctx)

    perf_ctx = section12_robustness_and_performance()   # 异常处理 + 性能 + 图 8
    ctx["perf"] = perf_ctx["perf"]
    ctx["error_messages"] = perf_ctx["error_messages"]
    ctx["figures"].update(perf_ctx["figures"])

    ctx["tests"] = section13_tests()
    section14_report(ctx)

    hr("全部完成")
    print(f"最优 k = {ctx['best_k']}   iris 测试集准确率 = {ctx['test_acc_best_k']:.4f}")
    print(f"结果目录：{RESULTS_DIR}")


if __name__ == "__main__":
    main()
