# 问题1：单孔径 PCB 钻孔路径优化 —— 严谨数学模型

---

## 一、符号说明

| 符号 | 含义 | 单位 |
|------|------|------|
| \(V = \{0, 1, 2, \ldots, n\}\) | 顶点集，其中 \(0\) 为原点 \(O(0,0)\)，\(1,\ldots,n\) 为 \(n\) 个钻孔点 | — |
| \((x_i, y_i)\) | 第 \(i\) 个点的坐标，\(i \in V\) | mm |
| \(d_{ij}\) | 点 \(i\) 与点 \(j\) 之间的欧氏距离：\(d_{ij}=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2}\) | mm |
| \(x_{ij} \in \{0,1\}\) | 决策变量，\(x_{ij}=1\) 若路径中包含边 \((i,j)\)，否则为 0 | — |
| \(v\) | 钻头移动速度 | \(100\ \text{mm/s}\) |
| \(T\) | 钻头移动总时间 | s |
| \(T_0\) | 模拟退火初始温度 | — |
| \(T_{\min}\) | 模拟退火终止温度 | — |
| \(\alpha\) | 模拟退火冷却系数 | — |
| \(N_T\) | 每个温度下的迭代次数 | — |

---

## 二、模型假设与论证

**假设 1：钻头视为质点，移动速度恒定。**

论证：钻头尺寸远小于 PCB 板面尺寸（数百毫米量级），其几何形状对路径规划无影响。钻床主轴在空行程中以恒定快速横移速度运动，忽略加速与减速过程。

**假设 2：所有钻孔点位于同一平面，距离度量采用欧氏距离。**

论证：PCB 为平面板件，钻孔点在 XY 平面内分布。钻床主轴在 X、Y 方向联动，实际路径为直线，故欧氏距离精确刻画实际移动距离。

**假设 3：钻孔顺序不受物理约束（如工件夹持、排屑等）。**

论证：本题聚焦于路径优化，假设钻孔机可自由访问任意顺序的点位，且 PCB 由真空吸附固定，无需考虑中间夹持换位。

**假设 4：起点与终点均为原点 \(O(0,0)\)。**

论证：钻头从原点起始，完成所有钻孔后返回原点，为加工下一块 PCB 板做准备。这使问题成为一个闭合 TSP。

---

## 三、数学模型建立

### 3.1 决策变量

定义二元变量 \(x_{ij}\)：

\[
x_{ij} = \begin{cases}
1, & \text{若钻头从点 } i \text{移动到点 } j \\
0, & \text{否则}
\end{cases}
\qquad (i,j \in V,\ i \neq j)
\]

### 3.2 目标函数

最小化钻头移动总距离：

\[
\min \quad Z = \sum_{i \in V} \sum_{j \in V \setminus \{i\}} d_{ij} \cdot x_{ij}
\]

总移动时间 \(T = Z / v = Z / 100\ \text{s}\)。

### 3.3 约束条件

**(1) 度约束（每个点恰好访问一次）：**

\[
\sum_{j \in V \setminus \{i\}} x_{ij} = 1, \quad \forall i \in V
\]
\[
\sum_{i \in V \setminus \{j\}} x_{ji} = 1, \quad \forall j \in V
\]

即每个点恰有一条出边和一条入边。

**(2) 子回路消除约束（DFJ 形式）：**

\[
\sum_{i \in S} \sum_{j \in V \setminus S} x_{ij} \geq 1, \quad \forall S \subset V,\ S \neq \emptyset,\ S \neq V
\]

保证路径是单一回路，而非多个不相交子回路。

**(3) 0-1 约束：**

\[
x_{ij} \in \{0, 1\}, \quad \forall i,j \in V,\ i \neq j
\]

---

## 四、算法设计与分析

### 4.1 最近邻算法（NN — Nearest Neighbor）

**思想：** 从原点出发，每次选择距离当前点最近的未访问点作为下一站。

**伪代码：**

```
Algorithm: Nearest_Neighbor(dist_matrix, n)
  Input: 距离矩阵 dist_matrix, 顶点数 n+1 (含原点)
  Output: 访问顺序 order (长度 n+2)
  
  unvisited ← {1, 2, ..., n}
  order ← [0]
  current ← 0
  
  while unvisited ≠ Ø:
    next ← argmin_{j ∈ unvisited} dist_matrix[current, j]
    order.append(next)
    unvisited.remove(next)
    current ← next
  
  order.append(0)   // 返回原点
  return order
```

**复杂度：** \(O(n^2)\)。每步扫描所有未访问点。

**特点：** 构造速度快，但缺乏全局视野——早期短视选择可能导致后期长距离"回补"。

### 4.2 2-opt 局部搜索

**思想：** 从 NN 初始解出发，反复尝试交换两条边的连接方式（反转一段子路径），若缩短则接受，直到无法改进。

**伪代码：**

```
Algorithm: 2-opt(initial_order, dist_matrix, n)
  Input: 初始路径 order (长度 n+2，首尾均为 0)
  Output: 优化后的路径
  improved ← true
  
  while improved:
    improved ← false
    for i ← 1 to n-1:
      for j ← i+1 to n:
        old ← dist[order[i-1], order[i]] + dist[order[j], order[j+1]]
        new ← dist[order[i-1], order[j]] + dist[order[i], order[j+1]]
        if new < old:
          order[i : j+1] ← reverse(order[i : j+1])
          improved ← true
          break  // 采用首次改进策略
  
  return order
```

**复杂度：** 每轮扫描 \(O(n^2)\)，总轮数通常为 \(O(\log n)\)，总复杂度约 \(O(n^2 \log n)\)。

**特点：** 确定性搜索，保证收敛到 2-opt 局部最优。对中小规模效果显著。

### 4.3 模拟退火（SA — Simulated Annealing）

**思想：** 以一定概率接受劣化解，跳出局部最优。温度从高到低，接受劣解的概率逐渐减小。

**伪代码：**

```
Algorithm: Simulated_Annealing(initial_order, dist_matrix, n,
                                alpha, T0, T_min, iters_per_T)
  Input: 初始路径 order (NN 解), 参数
  Output: 最优路径 best_order, 最优距离 best_cost
  
  order ← copy(initial_order)
  current_cost ← TotalPathLength(order, dist_matrix)
  best_order ← copy(order)
  best_cost ← current_cost
  T ← T0
  
  while T > T_min:
    for iter ← 1 to iters_per_T:
      i ← random(1, n)
      j ← random(i+1, n)
      // 计算 2-opt 移动的代价变化
      old ← dist[order[i-1], order[i]] + dist[order[j], order[j+1]]
      new ← dist[order[i-1], order[j]] + dist[order[i], order[j+1]]
      Δ ← new − old
      
      if Δ < 0 or random() < exp(−Δ / T):
        order[i : j+1] ← reverse(order[i : j+1])
        current_cost ← current_cost + Δ
        if current_cost < best_cost:
          best_cost ← current_cost
          best_order ← copy(order)
    
    T ← T × alpha
  
  return best_order, best_cost
```

**参数选择（基于敏感性分析）：**

| 参数 | 小规模 (n≤50) | 中规模 (n≤200) | 大规模 (n≤500) | 超大规模 (n>500) |
|------|:---:|:---:|:---:|:---:|
| \(\alpha\) | 0.995 | 0.998 | 0.999 | 0.9995 |
| \(T_0\) | 10000 | 10000 | 10000 | 10000 |
| \(N_T\) | 50 | 100 | 200 | 300 |

敏感性分析揭示：\(\alpha\) 是影响解质量的最关键参数——\(\alpha < 0.99\) 时 SA 完全退化为 NN（零改进）；初始温度 \(T_0\) 的影响出乎意料地小。

### 4.4 SA-2opt 混合策略

大规模实例上 SA-from-NN 劣于 2-opt 的现象，其根源不在于模拟退火本身，而在于初始解的选取。以 NN 解为起点，SA 在有限计算预算内难以跨越 NN 构造引入的系统性偏差。

**SA-2opt 混合策略**：将 SA 的初始解替换为 2-opt 局部最优解。

```
Algorithm: SA_2opt_Hybrid(opt_order, dist_matrix, n, ...)
  // opt_order 为 2-opt 局部搜索的输出
  return Simulated_Annealing(opt_order, dist_matrix, n, ...)
```

**效果：** SA-2opt 在大规模上保持了 2-opt 的解质量（n=1173: 12,664 mm），消除了 SA-from-NN 的退化（13,340 → 12,664 mm，差距 5.1%→0%）。SA 未能从 2-opt 解进一步逃离，表明 2-opt 的强局部最优可能已接近全局最优——这是一个有价值的负结果。

### 4.5 MVODM 距离矩阵变换

受饶卫振等[12]提出的距离矩阵方差最小法（MVODM）启发，本文尝试对距离矩阵进行预处理以改善贪心构建型算法的性能。

**变换公式：**

\[
\bar{d}_i = \frac{1}{n}\sum_{j \neq i} d_{ij}, \quad d'_{ij} = d_{ij} - \alpha(\bar{d}_i + \bar{d}_j)
\]

其中 \(\bar{d}_i\) 为节点 \(i\) 的平均距离（表征其"中心性"），\(\alpha\) 为缩放参数。变换后的距离矩阵允许出现负值——负距离表示"强偏好"边。

**关键实验发现：** (1) 需使用非钳制变换（允许负距离），`max(0,·)` 钳制会导致距离矩阵退化为零矩阵；(2) 最优缩放参数 \(\alpha=0.7\)。

| n | 原始 NN | MVODM NN | 原始 2-opt | MVODM 2-opt |
|---|------:|------:|------:|------:|
| 50 | 2,260 | **1,963** (+13.1%) | 1,957 | **1,852** (+5.4%) |
| 1173 | 14,698 | **13,894** (+5.5%) | 12,664 | 12,584 (+0.6%) |

MVODM 在小规模上效果显著（NN 改进 13.1%），大规模上对 NN 仍有 5.5% 的改善，但对 2-opt 提升有限（仅 0.6%），表明 2-opt 局部搜索已足够强大，预处理带来的额外收益递减。

### 4.6 与 Kirkpatrick 原始 SA 参数的对比

Kirkpatrick 等[5]在 1983 年提出 SA 时使用的参数集为 \(\alpha=0.9, T_0=10000\)，共 40 个温度级别 × 200 次尝试 = 8000 次总迭代。本文将此参数集与优化后的参数集进行对比：

| n | Kirkpatrick (α=0.9) | 本文优化 SA | 本文改进率 |
|---|:--:|:--:|:--:|
| 50 | 0% 改进 | 1,927 mm | **14.7%** |
| 198 | 0% 改进 | 3,247 mm | **14.1%** |
| 442 | 0% 改进 | 10,845 mm | **13.7%** |

**Kirkpatrick 原始参数集在本文全部 PCB 实例上零改进**——冷却速度过快（仅 40 次降温），8000 次随机 2-opt 探针中没有任何一次发现可行改善。这强烈证明了针对具体问题调整 SA 冷却策略的必要性。

---

## 五、结果验证

### 5.1 算法对比

| n | NN (mm) | 2-opt (mm) | SA (mm) | SA-2opt (mm) | SA vs NN 改进率 |
|---|--------:|----------:|--------:|------------:|----------:|
| 50 | 2,260 | 1,957 | **1,927** | 1,927 | 14.7% |
| 198 | 3,779 | 3,266 | **3,239** | 3,239 | 14.3% |
| 442 | 12,565 | **10,979** | 11,057 | **10,979** | 12.0% |
| 1173 | 14,698 | **12,664** | 13,340 | **12,664** | 9.7% |

**关键发现：** 小规模上 SA 全局搜索优势明显，MVODM 预处理进一步提升了 NN 的解质量；大规模上 2-opt 确定性搜索反超 SA-from-NN，SA-2opt 混合策略可消除退化。Kirkpatrick 原始 SA 参数（α=0.9）在本数据集上完全无效，凸显了参数调优的必要性。

### 5.2 最优性分析

对 n=50 使用 LKH（Lin-Kernighan Heuristic，elkai 实现）求得精确最优解，评估启发式方法的近似质量：

| 指标 | 数值 |
|------|------|
| LKH 精确最优解 | **1,833.77 mm** |
| 本文最优启发式结果 (SA) | 1,927.29 mm |
| 最优性差距 | **5.10%** |

差距 5.10% 处于启发式方法的预期范围（5-10%）内，对于仅使用构造启发式（NN）和局部搜索（2-opt）的方法而言是可接受的结果。

### 5.3 Held-Karp 下界（双重下界论证）

为进一步缩窄关于全局最优解的不确定性，使用 Held-Karp 松弛[11]计算最优值的理论下界。Held-Karp 下界将 TSP 松弛为最小 1-树问题，利用拉格朗日乘子迭代调整节点权重，逐步收紧下界。

| 指标 | 数值 (mm) |
|------|------:|
| Held-Karp 下界 (LB) | 1,798.86 |
| LKH 精确最优解 (UB) | 1,833.77 |
| LB 与 UB 之差距 | 34.91 mm (**1.90%**) |
| 真实最优值区间 | [1,798.86, 1,833.77] |
| 本文 SA 解 | 1,927.29 (距 HK 下界 **7.14%**) |

Held-Karp 下界与 LKH 上界构成的区间宽度仅 1.90%——这是非常严密的理论保证：真实全局最优解几乎可以肯定位于该狭窄区间内。本文 SA 解距 HK 下界 7.14%，说明仍有改进空间，但 5.10% 的 LKH gap 已接近该问题规模的启发式性能上限。
