# KNN 从零实现 —— 人工智能与机器学习 第 8 周实践一

一个**完全不依赖 scikit-learn KNN 组件**的 K 近邻算法实现，覆盖分类、回归、评估、
调参、可视化、对照实验、异常处理、性能优化与单元测试。

所有随机性由 `RANDOM_STATE = 42` 固定，重复运行结果完全一致。

---

## 一、目录结构

```
knn_practice/
├── knn.py                 # 核心算法：距离函数、最近邻搜索、投票、分类器、回归器
├── preprocess.py          # 数据读取/探索、训练测试划分、Z-score 标准化
├── model_selection.py     # KFold / StratifiedKFold / 交叉验证
├── metrics.py             # 手写评估指标：准确率、混淆矩阵、分类报告、MSE/RMSE/MAE/R²
├── make_data.py           # 生成 data/ 下的三个 CSV 数据集
├── main.py                # 端到端流水线（按作业 14 节顺序执行）
├── test_knn.py            # 单元测试（pytest，30 个用例）
├── requirements.txt
├── README.md
├── data/
│   ├── iris.csv           # 150 × 4，3 类分类
│   ├── moons2d.csv        # 300 × 2，2 类、线性不可分
│   └── regression.csv     # 220 × 1，非线性回归
└── results/
    ├── metrics.json       # 全流程指标（自动生成）
    ├── 实验记录.md         # 实验记录（自动生成）
    └── figures/           # 8 张图表（自动生成）
```

---

## 二、环境与安装

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
pip install -r requirements.txt
```

依赖：`numpy`、`pandas`、`matplotlib`、`scikit-learn`（**仅**用于第 11 节对照实验和
`make_data.py` 导出 iris 原始数据）、`pytest`。

> 核心算法文件 `knn.py` 只 import 了 `numpy`。把 `sklearn` 卸载掉，`knn.py`、
> `preprocess.py`、`metrics.py` 仍然可以正常工作和通过测试。

---

## 三、快速开始

```bash
python make_data.py     # 1. 生成数据集（仓库已附带 CSV，可跳过）
python main.py          # 2. 跑完整流水线，输出 results/
pytest -v               # 3. 跑 30 个单元测试
```

`main.py` 会依次执行作业的 14 个小节，每一步都在终端打印中间结果，
最后把指标写入 `results/metrics.json`、实验记录写入 `results/实验记录.md`、
图表写入 `results/figures/`。

---

## 四、算法流程

### 4.1 训练阶段（惰性学习）

KNN 没有真正的"训练"，`fit()` 只做三件事：

1. 校验输入（维度、NaN、样本数、`k` 是否超过样本数）；
2. 把训练集原样存进 `X_train_` / `y_train_`；
3. 记下类别集合 `classes_`。

所以训练是 O(1)，全部计算都推迟到预测阶段 —— 这就是"惰性学习（lazy learning）"。
代价是模型体积等于训练集，且预测慢。

### 4.2 预测阶段（5 步）

对每个待预测样本 `q`：

```
① 计算距离    d_i = ||q - x_i||₂   （或 L1），得到长度 n_train 的距离向量
② 排序取前 k  idx = argsort(d)[:k]
③ 取邻居标签  labels = y_train[idx]
④ 投票/平均   - 分类：多数投票（含平票处理）
              - 回归：均值（uniform）或距离倒数加权平均（distance）
⑤ 输出结果
```

### 4.3 复杂度

| 阶段 | 时间 | 空间 |
|---|---|---|
| fit | O(1) | O(n·d)（存训练集） |
| predict（单样本，暴力搜索） | O(n·d + n log n) | O(n) |
| predict（全部向量化） | O(n_test·n·d) 次浮点运算，BLAS 加速 | O(n_test·n) 距离矩阵 |

大数据场景的优化方向：KD-Tree / Ball-Tree 索引（把 O(n) 降到 O(log n)，但高维失效）、
乘积量化、近似最近邻（HNSW / FAISS）。

---

## 五、模块与作业小节对照

| 作业小节 | 实现位置 | 产物 |
|---|---|---|
| ① 工程环境与代码规范 | 整个目录结构；`main.py` 的 `section1_environment` | 本 README、固定随机种子 |
| ② 数据读取与探索 | `preprocess.load_csv` / `describe_dataset`；`section2_explore` | `fig01_data_overview.png` |
| ③ 数据预处理 | `preprocess.train_test_split` / `StandardScaler`；`section3_preprocess` | 防泄漏的标准化参数 |
| ④ 距离函数实现 | `knn.euclidean_distance` / `manhattan_distance` / `cdist`；`section4_distance` | 手算校验 + 一致性断言 |
| ⑤ 最近邻搜索 | `knn.nearest_neighbors`；`section5_nearest_neighbors` | 近邻索引与距离表 |
| ⑥ KNN 分类器 | `knn.KNNClassifier` + `majority_vote`；`section6_classifier` | 平票策略演示 |
| ⑦ KNN 回归 | `knn.KNNRegressor`；`section7_regression` | 两种权重的 MSE/RMSE/MAE/R² |
| ⑧ 模型评估 | `metrics.py`；`section8_evaluation` | `fig04_confusion_matrix.png` |
| ⑨ k 值选择与交叉验证 | `model_selection.py`；`section9_k_selection` | `fig03_k_curve.png` |
| ⑩ 可视化分析 | `section10_visualization` | 8 张图 |
| ⑪ sklearn 对照 | `section11_sklearn_compare` | 多配置准确率对照表 + 平票专项 |
| ⑫ 异常处理与性能优化 | `section12_robustness_and_performance` | 11 类异常提示 + `fig08_performance.png` |
| ⑬ 测试与复现 | `test_knn.py` | 30 个用例全通过 |
| ⑭ 报告与答辩 | `results/实验记录.md`（自动生成）、`REPORT.md` | 实验记录 + 报告 |

---

## 六、参数影响

### 6.1 k（近邻数）

| k | 表现 | 偏差-方差 |
|---|---|---|
| 1 | 决策边界贴合每个样本点，噪声被完整记忆 | 低偏差、高方差，过拟合 |
| 3–7 | 边界平滑但仍保留局部结构 | 平衡区 |
| 大（如 30+） | 边界趋于平滑，趋向"多数类" | 高偏差、低方差，欠拟合 |
| k = n | 永远预测训练集的多数类 | 极端欠拟合 |

经验起点 `k ≈ sqrt(n_train)`，**取奇数**可显著减少二分类平票。
本项目用 5 折分层交叉验证在 `k = 1, 3, ..., 31` 上选最优值。

### 6.2 metric（距离度量）

- **欧氏距离 L2**：默认选择，对特征的绝对差异敏感，需要先标准化。
- **曼哈顿距离 L1**：对离群维度更鲁棒（不像 L2 那样把大差异平方放大），
  在特征稀疏或维度较高时通常更稳。

两者在本项目的 iris 数据上表现接近（见 `fig07_metric_weights.png`），
这本身就是个结论：数据已经被标准化且各特征量纲一致时，度量选择的影响小于 k 的影响。

### 6.3 weights（权重方式）

- `uniform`：k 个邻居等权投票。简单、对噪声邻居没有抑制。
- `distance`：`w = 1/d`，越近影响越大。通常能小幅提升准确率，
  但**必须处理 d = 0**，本实现的策略是"距离为 0 的邻居独占全部权重"。

### 6.4 tie_break（平票策略）

- `nearest`（默认）：平票类别中，最近邻更近者胜出。更符合直觉 —— "最近的那个样本"
  本身携带最强证据。
- `smallest`：取标签较小者。**与 sklearn 完全一致**（sklearn 内部用 `scipy.stats.mode`，
  平票时返回最小值），但存在系统性偏向，总是偏爱字典序靠前的类别。

**这是手写实现与 sklearn 唯一的语义差异点**（实测：20 组配置在 iris 上预测零差异，
因为连续特征几乎不产生精确平票；构造的平票场景下 sklearn 判 A、`nearest` 判 B）。
k 取偶数 + 特征离散 + 样本极少时，这个差异才会真正暴露。

### 6.5 标准化

KNN 完全基于距离，量纲差异是致命的：iris 的 petal_length 取值 1–6.9，
若某个特征取值 0–1000，它就会垄断距离计算。**必须先做 Z-score 标准化**。

---

## 七、实验数据与流程

| 数据集 | 规模 | 任务 | 用途 |
|---|---|---|---|
| iris | 150 × 4，3 类 | 分类 | 主实验：分类、评估、调参、混淆矩阵 |
| moons2d | 300 × 2，2 类 | 分类 | 决策边界、误分类标注（二维才能画边界） |
| regression | 220 × 1，连续目标 | 回归 | KNN 回归、k 对平滑度的影响 |

统一流程：

```
读取 CSV
  → 分层划分 训练 70% / 测试 30%（random_state=42）
  → 只在训练集上 fit StandardScaler，测试集只 transform
  → 训练集内做 5 折分层交叉验证选 k（每折内部重新 fit 标准化器）
  → 用选出的 k 在测试集上做唯一一次最终评估
```

**测试集全程不参与调参**，这是避免数据泄漏的关键。

---

## 八、结果摘要

完整数字见 `results/实验记录.md` 与 `results/metrics.json`。核心结论：

1. iris 上 KNN 的分类准确率在 0.93–0.98 区间，k 在 3–15 之间表现最稳；
2. `moons2d` 上 k=1 的决策边界呈碎片状（过拟合噪声），k=15 过于平滑（欠拟合
   月牙的凹陷），k=5 左右最贴合数据结构；
3. 回归任务上 k 从 1 增到 60，拟合曲线从"穿过每个噪声点"逐渐平滑到接近真实函数，
   RMSE 先降后升，存在明显最优 k；
4. 与 sklearn 的 `KNeighborsClassifier` 在相同 k / metric / weights 下准确率基本一致，
   预测差异主要来自**平票规则的实现差异**和浮点精度导致的近邻排序互换；
5. 向量化实现比纯 Python 循环快 1–2 个数量级，加速比随矩阵规模增大而变大。

---

## 九、遇到的问题与改进

### 问题 1：平票导致结果不确定

**现象**：k 取偶数时，同一份数据多次运行预测结果可能不同（取决于排序稳定性）。

**原因**：`np.argsort` 默认非稳定排序，距离相同的邻居顺序不定。

**解决**：
1. `nearest_neighbors` 使用 `np.argsort(kind="stable")`，保证相同距离保持原始索引顺序；
2. 显式实现两种平票策略，并写进 README 和文档字符串；
3. 调参时只用奇数 k。

### 问题 2：距离加权在重合点除零

**现象**：`w = 1/d`，当查询点与训练点完全重合时 `d = 0`，产生 `inf` 或 `nan`。

**解决**：检测到 `d == 0` 时，让这些邻居分享全部权重（即完全匹配者决定预测结果），
并用 `pytest` 用例 `test_regressor_distance_weight_no_overflow_on_exact_match` 固定该行为。

### 问题 3：标准化泄漏

**现象**：最初直接对全量数据 `fit_transform` 再划分，导致测试集的均值方差进入了训练过程。

**解决**：`StandardScaler` 拆成 `fit` / `transform` 两步，交叉验证时**每一折内部重新 fit**；
用 `test_scaler_fit_uses_only_train_data` 用例固化。

### 问题 4：欧氏距离展开的内存膨胀

**现象**：`(a-b)² = a²+b²-2ab` 展开只消耗 O(n·m) 内存，比广播 `(n,m,d)` 省很多；
但曼哈顿距离无法用这个技巧，`np.abs(X[:,None,:] - Y[None,:,:])` 会吃 O(n·m·d) 内存。

**解决**：在 `manhattan_distances` 的文档字符串里明确标注该限制，并给出"大数据集应分块计算"的建议。

### 问题 5：`k > n_samples` 的失败模式

**现象**：直接 `argsort(...)[:k]` 在 `k > n` 时不会报错，而是静默返回不足 k 个邻居，
下游投票依然能"算出结果"，但结论是错的。

**解决**：在 `nearest_neighbors`、`fit`、`predict` 三处都做显式校验并抛出中文提示，
这是第 12 节"异常提示清晰"的验收点。

### 可进一步改进的方向

- 用 KD-Tree / Ball-Tree 降低预测复杂度（高维会退化成暴力搜索）；
- 加权投票引入类别先验，处理类别不均衡；
- 用 `argpartition` 代替 `argsort`，把 O(n log n) 降到 O(n)（本实现保留了 `argsort`
  以保证排序稳定性和可读性）；
- 加入 `n_jobs` 并行预测。

---

## 十、复现与 git

```bash
git init
git add .
git commit -m "feat: 从零实现 KNN 分类与回归（含评估、调参、可视化、测试）"
git remote add origin <你的云端仓库地址>
git push -u origin main
```

复现性保证：

- 所有随机性（划分、k 折、合成数据）统一由 `RANDOM_STATE = 42` 控制；
- `data/` 下的 CSV 随仓库提交，不依赖运行时随机生成；
- `results/` 全部由 `main.py` 自动生成，可随时删除重跑；
- `pytest` 30 个用例覆盖核心行为，改动后能立刻发现回归。
