"""
make_data.py
============
生成/导出三个数据集到 data/ 目录（CSV），保证实验数据可复现。

1. iris.csv        —— 经典鸢尾花数据（4 特征 × 3 类），来自 sklearn.datasets 内置数据，
                      只用于导出，不参与建模，模型本身完全不依赖 sklearn。
2. moons2d.csv     —— 自造二维双月数据（2 特征 × 2 类），
                      二维便于绘制散点图和决策边界。
3. regression.csv  —— 自造二维非线性回归数据（1 特征 × 连续目标），
                      用于 KNN 回归（对应作业第 7 节）。

用法：
    python make_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parent / "data"


def make_iris(path: Path) -> None:
    """导出 sklearn 内置鸢尾花数据为 CSV。"""
    from sklearn.datasets import load_iris

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=[c.replace(" (cm)", "").strip() for c in iris.feature_names])
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df["species"] = [iris.target_names[t] for t in iris.target]
    df.to_csv(path, index=False)
    print(f"[iris]        {df.shape[0]} 行 × {df.shape[1]} 列 -> {path.name}")
    print(f"              类别分布：{ {str(k): int(v) for k, v in df['species'].value_counts().items()} }")


def make_two_moons(n_samples: int = 300, noise: float = 0.22,
                   random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """自造二维双月数据（两个交错的半月形，线性不可分）。

    noise 取 0.22：噪声太小（0.1）时两个半月几乎可分，k=1 就能 100% 正确，
    k 的作用看不出来；0.22 时 k=1 的边界开始碎裂、k 增大后逐渐平滑，最能体现 k 的影响。
    """
    rng = np.random.default_rng(random_state)
    n_out = n_samples // 2
    n_in = n_samples - n_out

    t1 = np.linspace(0, np.pi, n_out)
    outer = np.column_stack([np.cos(t1), np.sin(t1)])

    t2 = np.linspace(0, np.pi, n_in)
    inner = np.column_stack([1.0 - np.cos(t2), 1.0 - np.sin(t2) - 0.5])

    X = np.vstack([outer, inner])
    y = np.array([0] * n_out + [1] * n_in)
    X = X + rng.normal(scale=noise, size=X.shape)
    return X, y


def make_moons(path: Path) -> None:
    X, y = make_two_moons()
    df = pd.DataFrame({"x1": X[:, 0].round(6), "x2": X[:, 1].round(6), "label": y})
    df.to_csv(path, index=False)
    print(f"[moons2d]     {df.shape[0]} 行 × {df.shape[1]} 列 -> {path.name}")
    print(f"              类别分布：{df['label'].value_counts().sort_index().to_dict()}")


def make_regression(n_samples: int = 220, noise: float = 0.15,
                    random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """自造一维非线性回归数据：y = 0.5x + 1.5·sin(2x) + ε。"""
    rng = np.random.default_rng(random_state)
    x = np.sort(rng.uniform(-3.0, 3.0, n_samples))
    y = 0.5 * x + 1.5 * np.sin(2.0 * x) + rng.normal(scale=noise, size=n_samples)
    return x, y


def make_regression_csv(path: Path) -> None:
    x, y = make_regression()
    df = pd.DataFrame({"x": x.round(6), "y": y.round(6)})
    df.to_csv(path, index=False)
    print(f"[regression]  {df.shape[0]} 行 × {df.shape[1]} 列 -> {path.name}")
    print(f"              目标 y 范围：[{y.min():.3f}, {y.max():.3f}]，均值 {y.mean():.3f}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"数据输出目录：{DATA_DIR}\n")
    make_iris(DATA_DIR / "iris.csv")
    make_moons(DATA_DIR / "moons2d.csv")
    make_regression_csv(DATA_DIR / "regression.csv")
    print("\n全部数据集生成完毕。")


if __name__ == "__main__":
    main()
