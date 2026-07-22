"""
问题一：3-opt 验证 + Held-Karp 下界 + Greedy Insertion + MVODM
对 n=50 运行 3-opt，对比 2-opt 结果，计算 Held-Karp 下界
同时对 Greedy Insertion 跑 MVODM 预处理

产出:
  - output/tables/p1_3opt_check.csv
  - output/tables/p1_hk_bound.csv
  - output/tables/p1_mvodm_gi.csv
"""
import os
import sys
import io
import time
import numpy as np
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import load_drill_data, get_coords
from src.utils.distance import euclidean_distance_matrix, total_path_length
from src.problem1.solve_all import nearest_neighbor, two_opt

np.random.seed(42)
SPEED = 100.0
N = 50

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 72)
print("  Problem 1: 3-opt + Held-Karp + Greedy Insertion + MVODM")
print("=" * 72)

# ── Load data ──
data = load_drill_data(N)
drill_coords = get_coords(data)
origin = np.array([[0.0, 0.0]])
coords = np.vstack([origin, drill_coords])
dist_matrix = euclidean_distance_matrix(coords)

# ═══════════════════════════════════════════════════════════════════
# 3-opt 局部搜索
# ═══════════════════════════════════════════════════════════════════

def three_opt(order: list, dist_matrix: np.ndarray, n: int, max_passes: int = 30) -> list:
    """
    3-opt 局部搜索：尝试移除三条边并用其他方式重新连接。
    限制 passes 数量以避免无限循环。
    """
    order = list(order)
    improved = True
    iteration = 0
    
    while improved and iteration < max_passes:
        improved = False
        iteration += 1
        
        # i < j < k，三个切割点
        # 发现改进后立即 break 出所有 for 循环，下个 pass 从头开始
        for i in range(1, n - 1):
            if improved:
                break
            for j in range(i + 1, n):
                if improved:
                    break
                for k in range(j + 1, n + 1):
                    # 当前路径段: [0..i-1] [i..j] [j+1..k] [k+1..n] 0
                    a, b = order[i - 1], order[i]
                    c, d = order[j], order[j + 1] if j + 1 <= n else order[0]
                    e, f = order[k], order[k + 1] if k + 1 <= n else order[0]
                    
                    # 当前三条边
                    old_edges = dist_matrix[a, b] + dist_matrix[c, d] + dist_matrix[e, f]
                    
                    # 只检查主要的 3 种 non-trivial 3-opt 重连方式
                    new1 = dist_matrix[a, c] + dist_matrix[b, d] + dist_matrix[e, f]  # reverse [i..j]
                    new2 = dist_matrix[a, b] + dist_matrix[c, e] + dist_matrix[d, f]  # reverse [j+1..k]
                    new3 = dist_matrix[a, c] + dist_matrix[b, e] + dist_matrix[d, f]  # reverse both
                    
                    best_new = min(new1, new2, new3)
                    
                    if best_new < old_edges - 1e-9:
                        if new1 == best_new:
                            order[i:j + 1] = list(reversed(order[i:j + 1]))
                        elif new2 == best_new:
                            order[j + 1:k + 1] = list(reversed(order[j + 1:k + 1]))
                        elif new3 == best_new:
                            order[i:j + 1] = list(reversed(order[i:j + 1]))
                            order[j + 1:k + 1] = list(reversed(order[j + 1:k + 1]))
                        improved = True
                        break  # 立即 break k 循环
    
    return order


# ═══════════════════════════════════════════════════════════════════
# Greedy Insertion
# ═══════════════════════════════════════════════════════════════════

def greedy_insertion(dist_matrix: np.ndarray, n: int) -> list:
    """
    Greedy Insertion: 从原点出发，每次选择使路径增量最小的插入位置。
    """
    # 初始路径: 0 -> 0
    order = [0, 0]
    unvisited = set(range(1, n + 1))
    
    while unvisited:
        best_node = -1
        best_pos = -1
        best_delta = float('inf')
        
        for node in unvisited:
            # 尝试将 node 插入到路径的每个可能位置
            for pos in range(1, len(order)):
                prev = order[pos - 1]
                nxt = order[pos]
                delta = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                
                if delta < best_delta:
                    best_delta = delta
                    best_node = node
                    best_pos = pos
        
        # 插入最优节点到最优位置
        if best_node >= 0:
            order.insert(best_pos, best_node)
            unvisited.remove(best_node)
    
    return order


# ═══════════════════════════════════════════════════════════════════
# Held-Karp Lower Bound (from optimality_check.py)
# ═══════════════════════════════════════════════════════════════════

def compute_held_karp_lower_bound(dist_matrix: np.ndarray, n: int):
    """
    Held-Karp lower bound using 1-tree relaxation with subgradient optimization.
    """
    V = n + 1
    pi = np.zeros(V)
    best_lower_bound = 0.0
    step_size = 10.0
    alpha = 2.0
    max_iter = 200

    for iter_count in range(max_iter):
        modified = dist_matrix.copy()
        for i in range(V):
            modified[i, :] += pi[i]
            modified[:, i] += pi[i]

        in_tree = {1}
        not_in_tree = set(range(2, V))
        mst_cost = 0.0
        mst_degree = np.zeros(V, dtype=int)

        while not_in_tree:
            best_edge_cost = float('inf')
            best_target = -1
            best_source = -1
            for u in in_tree:
                for v in not_in_tree:
                    if modified[u, v] < best_edge_cost:
                        best_edge_cost = modified[u, v]
                        best_target = v
                        best_source = u

            in_tree.add(best_target)
            not_in_tree.remove(best_target)
            mst_cost += best_edge_cost
            mst_degree[best_source] += 1
            mst_degree[best_target] += 1

        edges_to_0 = [(modified[0, j], j) for j in range(1, V)]
        edges_to_0.sort()
        one_tree_cost = mst_cost + edges_to_0[0][0] + edges_to_0[1][0]

        one_tree_degree = mst_degree.copy()
        one_tree_degree[0] = 2
        one_tree_degree[edges_to_0[0][1]] += 1
        one_tree_degree[edges_to_0[1][1]] += 1

        subgradient = one_tree_degree - 2
        lb = one_tree_cost - 2.0 * np.sum(pi)

        if lb > best_lower_bound:
            best_lower_bound = lb

        sg_norm = np.sqrt(np.sum(subgradient ** 2))
        if sg_norm < 1e-8:
            break

        current_step = alpha * step_size / sg_norm
        pi += current_step * subgradient
        step_size *= 0.95

    return best_lower_bound


# ═══════════════════════════════════════════════════════════════════
# MVODM transform (from mvodm_check.py)
# ═══════════════════════════════════════════════════════════════════

def mvodm_transform(dist_matrix: np.ndarray, alpha: float) -> np.ndarray:
    n_plus_1 = dist_matrix.shape[0]
    avg_dist = np.sum(dist_matrix, axis=1) / (n_plus_1 - 1)
    D_new = dist_matrix - alpha * (avg_dist[:, np.newaxis] + avg_dist[np.newaxis, :])
    np.fill_diagonal(D_new, 0.0)
    return D_new


# ═══════════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'─' * 60}")
print(f"  Part 1: 3-opt validation (n={N})")
print(f"{'─' * 60}")

# Baseline: 2-opt from NN
nn_order = nearest_neighbor(dist_matrix, N)
nn_dist = total_path_length(nn_order, dist_matrix)
opt2_order = two_opt(nn_order, dist_matrix, N)
opt2_dist = total_path_length(opt2_order, dist_matrix)

print(f"  [NN]      distance = {nn_dist:.4f} mm")
print(f"  [2-opt]   distance = {opt2_dist:.4f} mm")

# 3-opt from 2-opt
print(f"  [3-opt]   running from 2-opt solution...", end="", flush=True)
t0 = time.perf_counter()
opt3_order = three_opt(opt2_order, dist_matrix, N)
opt3_dist = total_path_length(opt3_order, dist_matrix)
t1 = time.perf_counter()
print(f"\r  [3-opt]   distance = {opt3_dist:.4f} mm  CPU={t1-t0:.2f}s")

improvement_3opt = (opt2_dist - opt3_dist) / opt2_dist * 100.0
print(f"  3-opt vs 2-opt improvement: {improvement_3opt:+.4f}%")

# Save 3-opt results
tables_dir = os.path.join(PROJECT_ROOT, "output", "tables")
os.makedirs(tables_dir, exist_ok=True)
csv_3opt = os.path.join(tables_dir, "p1_3opt_check.csv")
with open(csv_3opt, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "Algorithm", "Distance_mm", "Time_s"])
    writer.writerow([N, "NN", f"{nn_dist:.4f}", f"{nn_dist/SPEED:.4f}"])
    writer.writerow([N, "2-opt", f"{opt2_dist:.4f}", f"{opt2_dist/SPEED:.4f}"])
    writer.writerow([N, "3-opt", f"{opt3_dist:.4f}", f"{opt3_dist/SPEED:.4f}"])
    writer.writerow([N, "3opt_vs_2opt_improvement_pct", f"{improvement_3opt:.4f}", ""])
print(f"  [CSV] saved: {csv_3opt}")


print(f"\n{'─' * 60}")
print(f"  Part 2: Held-Karp lower bound (n={N})")
print(f"{'─' * 60}")

# Our best from SA
our_best = 1927.2947
our_algo = "SA"

print(f"  Our best: {our_best:.4f} mm ({our_algo})")
print(f"  [Held-Karp] computing lower bound...", end="", flush=True)
t0 = time.perf_counter()
hk_lb = compute_held_karp_lower_bound(dist_matrix, N)
t1 = time.perf_counter()
print(f"\r  [Held-Karp] lower bound = {hk_lb:.4f} mm  CPU={t1-t0:.2f}s")

# LKH optimal from previous run
lkh_opt = 1833.7724
gap_vs_lkh = (our_best - lkh_opt) / lkh_opt * 100.0
gap_vs_hk = (our_best - hk_lb) / hk_lb * 100.0

print(f"  Gap vs LKH optimal:  {gap_vs_lkh:.4f}%")
print(f"  Gap vs Held-Karp LB: {gap_vs_hk:.4f}%")

# Save HK results
csv_hk = os.path.join(tables_dir, "p1_hk_bound.csv")
with open(csv_hk, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "Our_Best_mm", "Our_Algo", "LKH_Optimal_mm", "HeldKarp_LB_mm",
                      "Gap_vs_LKH_pct", "Gap_vs_HK_pct"])
    writer.writerow([N, f"{our_best:.4f}", our_algo, f"{lkh_opt:.4f}", f"{hk_lb:.4f}",
                      f"{gap_vs_lkh:.4f}", f"{gap_vs_hk:.4f}"])
print(f"  [CSV] saved: {csv_hk}")


print(f"\n{'─' * 60}")
print(f"  Part 3: Greedy Insertion + MVODM (n={N})")
print(f"{'─' * 60}")

# Baseline GI
t0 = time.perf_counter()
gi_order = greedy_insertion(dist_matrix, N)
gi_dist = total_path_length(gi_order, dist_matrix)
t1 = time.perf_counter()
print(f"  [GI]      distance = {gi_dist:.4f} mm  CPU={t1-t0:.2f}s")

# GI + 2-opt
gi2_order = two_opt(gi_order, dist_matrix, N)
gi2_dist = total_path_length(gi2_order, dist_matrix)
print(f"  [GI+2opt] distance = {gi2_dist:.4f} mm")

# GI + MVODM (best alpha from mvodm_check = 0.70)
best_alpha = 0.70
D_trans = mvodm_transform(dist_matrix, best_alpha)
gi_trans_order = greedy_insertion(D_trans, N)
gi_trans_dist = total_path_length(gi_trans_order, dist_matrix)
gi_trans2_order = two_opt(gi_trans_order, dist_matrix, N)
gi_trans2_dist = total_path_length(gi_trans2_order, dist_matrix)
print(f"  [GI+MVODM(α={best_alpha})] distance = {gi_trans_dist:.4f} mm")
print(f"  [GI+MVODM+2opt]      distance = {gi_trans2_dist:.4f} mm")

gi_impr = (nn_dist - gi_dist) / nn_dist * 100.0
gi2_impr = (opt2_dist - gi2_dist) / opt2_dist * 100.0
gi_mv_impr = (gi_dist - gi_trans_dist) / gi_dist * 100.0

print(f"  GI vs NN improvement: {gi_impr:+.2f}%")
print(f"  GI+2opt vs 2opt improvement: {gi2_impr:+.2f}%")
print(f"  GI+MVODM vs GI improvement: {gi_mv_impr:+.2f}%")

# Save GI results
csv_gi = os.path.join(tables_dir, "p1_mvodm_gi.csv")
with open(csv_gi, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "Distance_mm", "Time_s", "vs_NN_pct", "vs_2opt_pct"])
    writer.writerow(["NN", f"{nn_dist:.4f}", f"{nn_dist/SPEED:.4f}", "0.00", "—"])
    writer.writerow(["2-opt", f"{opt2_dist:.4f}", f"{opt2_dist/SPEED:.4f}", "—", "0.00"])
    writer.writerow(["GI", f"{gi_dist:.4f}", f"{gi_dist/SPEED:.4f}", f"{gi_impr:.2f}", "—"])
    writer.writerow(["GI+2opt", f"{gi2_dist:.4f}", f"{gi2_dist/SPEED:.4f}", "—", f"{gi2_impr:.2f}"])
    writer.writerow(["GI+MVODM", f"{gi_trans_dist:.4f}", f"{gi_trans_dist/SPEED:.4f}", "—", "—"])
    writer.writerow(["GI+MVODM+2opt", f"{gi_trans2_dist:.4f}", f"{gi_trans2_dist/SPEED:.4f}", "—", "—"])
print(f"  [CSV] saved: {csv_gi}")


print(f"\n{'=' * 72}")
print(f"  Summary")
print(f"{'=' * 72}")
print(f"  n={N} results:")
print(f"    NN:           {nn_dist:.4f} mm")
print(f"    2-opt:        {opt2_dist:.4f} mm")
print(f"    3-opt:        {opt3_dist:.4f} mm  (vs 2-opt: {improvement_3opt:+.4f}%)")
print(f"    GI:           {gi_dist:.4f} mm")
print(f"    GI+2opt:      {gi2_dist:.4f} mm")
print(f"    GI+MVODM:     {gi_trans_dist:.4f} mm")
print(f"    SA (our best): {our_best:.4f} mm")
print(f"    LKH optimal:  {lkh_opt:.4f} mm")
print(f"    Held-Karp LB: {hk_lb:.4f} mm")
print(f"    Gap vs LKH:   {gap_vs_lkh:.4f}%")
print(f"    Gap vs HK:    {gap_vs_hk:.4f}%")
print(f"{'=' * 72}")
print(f"  Done!")
