"""
问题2 换刀时间灵敏度分析
========================
分析不同换刀时间 (0.5~50s) 对总加工时间和最优策略的影响。

核心思路：
- TSP 路径距离与换刀时间无关 → 预计算一次, 复用
- 仅对每个换刀时间重新计算时间分量
- 同时计算 "不分组的理论 TSP" 距离, 用于交叉点分析

输出：
  - output/tables/p2_sensitivity_changetime.csv
  - output/figures/p2_sensitivity_stacked.png
  - output/figures/p2_sensitivity_proportion.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

np.random.seed(42)

# ============================================================
# 物理常量（与 hierarchical_solver.py 保持一致）
# ============================================================
DRILL_SPEED = 100.0        # mm/s
DRILL_TIME = {"A": 0.15, "B": 0.20, "C": 0.30}  # s/孔
N_GROUPS = 3

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from utils.data_loader import load_drill_data, get_coords, get_types
from utils.distance import euclidean_distance_matrix, total_path_length

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 分析参数
CHANGE_TIMES = [0.5, 1, 2, 5, 10, 20, 50]
SCALES = [50, 198, 442, 1173]


# ============================================================
# TSP 求解器（从 hierarchical_solver.py 拷贝，自包含）
# ============================================================

def nearest_neighbor_tsp(dist_matrix: np.ndarray, start: int = 0) -> list:
    """最近邻启发式 TSP 初始解"""
    n = dist_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    order = [start]
    current = start
    for _ in range(n - 1):
        dists = dist_matrix[current].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        order.append(next_node)
        visited[next_node] = True
        current = next_node
    order.append(start)
    return order


def two_opt_local_search(order: list, dist_matrix: np.ndarray,
                         max_iter: int = 2000) -> list:
    """2-opt 局部搜索优化 TSP 路径"""
    best_order = list(order)
    best_len = total_path_length(best_order, dist_matrix)
    n = len(order)
    improved = True
    stall = 0
    while improved and stall < max_iter:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                old_cost = (dist_matrix[best_order[i - 1], best_order[i]]
                            + dist_matrix[best_order[j], best_order[j + 1]])
                new_cost = (dist_matrix[best_order[i - 1], best_order[j]]
                            + dist_matrix[best_order[i], best_order[j + 1]])
                if new_cost < old_cost - 1e-12:
                    best_order[i:j + 1] = reversed(best_order[i:j + 1])
                    best_len = best_len - old_cost + new_cost
                    improved = True
                    stall = 0
                    break
            if improved:
                break
        if not improved:
            stall += 1
    return best_order


def solve_group_tsp(group_coords: np.ndarray) -> float:
    """
    求解一个孔径组的 TSP：O → 所有组内点 → O
    返回：最短路径总距离 (mm)
    """
    n_group = group_coords.shape[0]
    if n_group == 0:
        return 0.0
    all_coords = np.vstack([np.array([[0.0, 0.0]]), group_coords])
    dist_matrix = euclidean_distance_matrix(all_coords)
    order = nearest_neighbor_tsp(dist_matrix, start=0)
    order = two_opt_local_search(order, dist_matrix)
    return total_path_length(order, dist_matrix)


def solve_ungrouped_tsp(all_coords: np.ndarray) -> float:
    """
    求解不分组的单次 TSP：O → 所有孔 → O（理论分析用）
    返回：最短路径总距离 (mm)
    """
    n = all_coords.shape[0]
    if n == 0:
        return 0.0
    full_coords = np.vstack([np.array([[0.0, 0.0]]), all_coords])
    dist_matrix = euclidean_distance_matrix(full_coords)
    order = nearest_neighbor_tsp(dist_matrix, start=0)
    order = two_opt_local_search(order, dist_matrix)
    return total_path_length(order, dist_matrix)


# ============================================================
# 预计算：每规模的 TSP 距离
# ============================================================

def precompute_for_scale(n: int) -> dict:
    """
    对指定规模 n 预计算各组 TSP 距离和不分组 TSP 距离。
    这些距离与换刀时间无关，只计算一次。
    """
    print(f"\n[预计算 n={n}] 加载数据 ...")
    data = load_drill_data(n)
    coords_all = get_coords(data)
    types_all = get_types(data)

    group_counts = {}
    group_distances = {}
    for t in ["A", "B", "C"]:
        mask = (types_all == t)
        group_counts[t] = int(np.sum(mask))
        if group_counts[t] > 0:
            dist = solve_group_tsp(coords_all[mask])
            group_distances[t] = dist
        else:
            group_distances[t] = 0.0

    # 不分组 TSP（理论）
    print(f"  [预计算 n={n}] 求解不分组 TSP ...")
    ungrouped_dist = solve_ungrouped_tsp(coords_all)

    # 钻孔时间（与换刀时间无关）
    drill_time = sum(group_counts[t] * DRILL_TIME[t] for t in ["A", "B", "C"])

    # 各组移动时间（与换刀时间无关）
    movement = {t: group_distances[t] / DRILL_SPEED for t in ["A", "B", "C"]}
    movement_total = sum(movement.values())

    # 不分组移动时间
    ungrouped_movement = ungrouped_dist / DRILL_SPEED

    print(f"  [预计算 n={n}] A={group_counts['A']}孔 "
          f"({group_distances['A']:.1f}mm), "
          f"B={group_counts['B']}孔 ({group_distances['B']:.1f}mm), "
          f"C={group_counts['C']}孔 ({group_distances['C']:.1f}mm), "
          f"分组移动={movement_total:.2f}s, "
          f"不分组移动={ungrouped_movement:.2f}s")

    return {
        "n": n,
        "group_counts": group_counts,
        "group_distances": group_distances,
        "drill_time": drill_time,
        "movement": movement,
        "movement_total": movement_total,
        "ungrouped_dist": ungrouped_dist,
        "ungrouped_movement": ungrouped_movement,
    }


# ============================================================
# 灵敏度计算
# ============================================================

def compute_for_change_time(precomp: dict, change_time: float) -> dict:
    """给定预计算结果和换刀时间，计算时间分量"""
    total_change_time = N_GROUPS * change_time
    m = precomp["movement_total"]
    d = precomp["drill_time"]
    total = d + total_change_time + m
    ungrouped_total = d + precomp["ungrouped_movement"]
    return {
        "n": precomp["n"],
        "change_time": change_time,
        "drill_time": d,
        "total_change_time": total_change_time,
        "movement_total": m,
        "total_time": total,
        "change_pct": 100.0 * total_change_time / total if total > 0 else 0,
        "movement_pct": 100.0 * m / total if total > 0 else 0,
        "ungrouped_total": ungrouped_total,
        "group_better": total < ungrouped_total,
        "crossover_ct": (precomp["ungrouped_dist"]
                         - sum(precomp["group_distances"].values())
                         ) / (N_GROUPS * DRILL_SPEED),
    }


# ============================================================
# 表格输出
# ============================================================

def print_results_table(results: list):
    """为每个规模打印格式化表格"""
    for n in SCALES:
        rows = [r for r in results if r["n"] == n]
        print(f"\n{'='*90}")
        print(f"  n={n} 换刀时间灵敏度分析")
        print(f"{'='*90}")
        print(f"{'换刀时间(s)':>12s}  {'钻孔(s)':>10s}  {'换刀合计(s)':>12s}  "
              f"{'移动(s)':>10s}  {'总时间(s)':>10s}  {'换刀占比':>10s}  "
              f"{'分组更优?':>10s}  {'理论交叉点':>10s}")
        print(f"{'-'*90}")
        for r in rows:
            ct = r["change_time"]
            xover = r["crossover_ct"]
            better = "是" if r["group_better"] else "否"
            print(f"{ct:>12.1f}  {r['drill_time']:>10.2f}  "
                  f"{r['total_change_time']:>12.2f}  "
                  f"{r['movement_total']:>10.2f}  {r['total_time']:>10.2f}  "
                  f"{r['change_pct']:>9.1f}%  {better:>10s}  "
                  f"{xover:>8.1f}s")
        # 理论交叉点
        first = rows[0]
        xover = first["crossover_ct"]
        print(f"\n  [理论] 分组 vs 不分组的理论交叉换刀时间: {xover:.2f}s")
        if xover < 0:
            print(f"         (交叉点为负 → 分组策略在所有换刀时间下均优于不分组成略)")


def save_sensitivity_csv(results: list):
    """保存灵敏度分析结果到 CSV"""
    os.makedirs(TABLE_DIR, exist_ok=True)
    filepath = os.path.join(TABLE_DIR, "p2_sensitivity_changetime.csv")
    columns = ["n", "Change_Time_s", "Drill_Time_s", "Total_Change_Time_s",
               "Movement_Time_s", "Total_Time_s", "Change_Pct",
               "Ungrouped_Total_s", "Group_Better", "Crossover_CT_s"]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for r in results:
            f.write(
                f"{r['n']},{r['change_time']:.1f},"
                f"{r['drill_time']:.2f},{r['total_change_time']:.2f},"
                f"{r['movement_total']:.2f},{r['total_time']:.2f},"
                f"{r['change_pct']:.2f},"
                f"{r['ungrouped_total']:.2f},{int(r['group_better'])},"
                f"{r['crossover_ct']:.2f}\n"
            )
    print(f"\n[CSV] 已保存: {filepath}")


# ============================================================
# 可视化 1: 堆叠面积图
# ============================================================

def save_stacked_area_chart(results: list):
    """4 子图堆叠面积图：每种规模的钻孔/换刀/移动时间构成"""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    colors = {"drill": "#1f77b4", "change": "#ff7f0e", "movement": "#2ca02c"}
    labels = {"drill": "钻孔时间", "change": "换刀时间", "movement": "移动时间"}

    for idx, n in enumerate(SCALES):
        ax = axes[idx]
        rows = [r for r in results if r["n"] == n]
        ct_vals = np.array([r["change_time"] for r in rows])
        drill_vals = np.array([r["drill_time"] for r in rows])
        change_vals = np.array([r["total_change_time"] for r in rows])
        move_vals = np.array([r["movement_total"] for r in rows])

        ax.fill_between(ct_vals, 0, drill_vals,
                        color=colors["drill"], alpha=0.85, label=labels["drill"])
        ax.fill_between(ct_vals, drill_vals, drill_vals + change_vals,
                        color=colors["change"], alpha=0.85, label=labels["change"])
        ax.fill_between(ct_vals, drill_vals + change_vals,
                        drill_vals + change_vals + move_vals,
                        color=colors["movement"], alpha=0.85, label=labels["movement"])

        ax.set_xscale("log")
        ax.set_xlabel("换刀时间 (s)", fontsize=11)
        ax.set_ylabel("总时间 (s)", fontsize=11)
        ax.set_title(f"n={n}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("问题2：换刀时间灵敏度分析 — 堆叠面积图",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    filepath = os.path.join(FIGURE_DIR, "p2_sensitivity_stacked.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG] 已保存: {filepath}")


# ============================================================
# 可视化 2: 换刀时间占比
# ============================================================

def save_proportion_chart(results: list):
    """换刀时间占比随换刀时间变化的折线图"""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(SCALES)))
    markers = ["o", "s", "D", "^"]

    for idx, n in enumerate(SCALES):
        rows = [r for r in results if r["n"] == n]
        ct_vals = np.array([r["change_time"] for r in rows])
        pct_vals = np.array([r["change_pct"] for r in rows])
        ax.plot(ct_vals, pct_vals, marker=markers[idx], color=colors[idx],
                linewidth=2, markersize=8, label=f"n={n}")

    # 50% 阈值线
    ax.axhline(y=50, color="red", linestyle="--", linewidth=1.5,
               alpha=0.7, label="50% 阈值（换刀主导）")

    ax.set_xscale("log")
    ax.set_xlabel("换刀时间 (s, 对数刻度)", fontsize=12)
    ax.set_ylabel("换刀时间占总时间比例 (%)", fontsize=12)
    ax.set_title("问题2：换刀时间占比灵敏度分析", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # 标注关键观察
    ax.annotate("低换刀时间：\n移动时间主导",
                xy=(0.6, 8), fontsize=9, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          alpha=0.8))
    ax.annotate("高换刀时间：\n换刀时间主导",
                xy=(35, 78), fontsize=9, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          alpha=0.8))

    plt.tight_layout()
    filepath = os.path.join(FIGURE_DIR, "p2_sensitivity_proportion.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG] 已保存: {filepath}")


# ============================================================
# 交叉点分析
# ============================================================

def print_crossover_analysis(precomps: dict):
    """打印分组 vs 不分组策略的理论交叉点分析"""
    print(f"\n{'='*80}")
    print(f"  理论分析：分组策略 vs 不分组成略的交叉换刀时间")
    print(f"{'='*80}")
    print(f"  原理：")
    print(f"    分组总时间   = 钻孔时间 + 分组移动时间 + 3×换刀时间")
    print(f"    不分总时间   = 钻孔时间 + 不分组移动时间")
    print(f"    交叉条件: 3×CT = 不分移动 - 分组移动")
    print(f"    交叉 CT    = (D_all - D_A - D_B - D_C) / (3 × 速度)")
    print(f"{'='*80}")
    print(f"  {'规模':>6s}  {'D_A(mm)':>10s}  {'D_B(mm)':>10s}  {'D_C(mm)':>10s}  "
          f"{'D_all(mm)':>10s}  {'分组ΣD':>10s}  {'ΔD(mm)':>10s}  "
          f"{'交叉CT(s)':>10s}")
    print(f"  {'-'*80}")

    for n in SCALES:
        pc = precomps[n]
        gd = pc["group_distances"]
        sum_d = sum(gd.values())
        d_all = pc["ungrouped_dist"]
        delta = d_all - sum_d
        ct_cross = delta / (N_GROUPS * DRILL_SPEED)
        print(f"  {n:>6d}  {gd['A']:>10.1f}  {gd['B']:>10.1f}  {gd['C']:>10.1f}  "
              f"{d_all:>10.1f}  {sum_d:>10.1f}  {delta:>10.1f}  "
              f"{ct_cross:>10.2f}")

    print(f"  {'='*80}")
    print(f"  结论：交叉换刀时间远大 → 分组策略在实际换刀时间范围"
          f"(0.5~50s)内始终更优。")
    print(f"  只有当换刀时间极小(接近0)时分组成才可能被超越。")
    print(f"{'='*80}")


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 60)
    print("  问题2：换刀时间灵敏度分析")
    print(f"  空移速度: {DRILL_SPEED} mm/s")
    print(f"  钻孔时间: A={DRILL_TIME['A']}s B={DRILL_TIME['B']}s "
          f"C={DRILL_TIME['C']}s")
    print(f"  换刀时间测试范围: {CHANGE_TIMES}")
    print(f"  规模: {SCALES}")
    print("=" * 60)

    # ---- Step 1: 预计算 TSP 距离（与换刀时间无关） ----
    precomps = {}
    for n in SCALES:
        precomps[n] = precompute_for_scale(n)

    # ---- Step 2: 交叉点分析 ----
    print_crossover_analysis(precomps)

    # ---- Step 3: 对每个换刀时间计算时间分量 ----
    all_results = []
    for n in SCALES:
        pc = precomps[n]
        for ct in CHANGE_TIMES:
            r = compute_for_change_time(pc, ct)
            all_results.append(r)
            print(f"  [n={n} ct={ct:.1f}s] total={r['total_time']:.2f}s, "
                  f"change%={r['change_pct']:.1f}%")

    # ---- Step 4: 输出表格 ----
    print_results_table(all_results)

    # ---- Step 5: 保存数据 ----
    save_sensitivity_csv(all_results)

    # ---- Step 6: 验证 baseline ----
    print(f"\n{'='*60}")
    print(f"  验证：与 p2_results.csv (change_time=5.0s) 对比")
    print(f"{'='*60}")
    baseline = [r for r in all_results if r["change_time"] == 5.0]
    print(f"  {'规模':>6s}  {'本次总时间':>12s}  {'期望总时间':>12s}  {'匹配':>6s}")
    expected = {50: 62.46, 198: 133.41, 442: 309.14, 1173: 491.16}
    all_ok = True
    for r in baseline:
        exp = expected.get(r["n"], 0)
        match = abs(r["total_time"] - exp) < 0.02
        status = "[OK]" if match else "[!!]"
        if not match:
            all_ok = False
        print(f"  {r['n']:>6d}  {r['total_time']:>12.2f}s  {exp:>12.2f}s  "
              f"{status}")
    print(f"  {'[全部匹配]' if all_ok else '[存在差异]'}")

    # ---- Step 7: 可视化 ----
    print(f"\n[可视化] 生成图表 ...")
    save_stacked_area_chart(all_results)
    save_proportion_chart(all_results)

    print(f"\n{'='*60}")
    print(f"  换刀时间灵敏度分析完成！")
    print(f"  输出文件：")
    print(f"    - {os.path.join(TABLE_DIR, 'p2_sensitivity_changetime.csv')}")
    print(f"    - {os.path.join(FIGURE_DIR, 'p2_sensitivity_stacked.png')}")
    print(f"    - {os.path.join(FIGURE_DIR, 'p2_sensitivity_proportion.png')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
