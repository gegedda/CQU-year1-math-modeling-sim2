"""
问题一：最优性下界检查 (n=50)
使用 LKH (Lin-Kernighan Heuristic) 计算精确/近似最优 TSP 解，
并与启发式算法结果比较，给出最优性差距。

方法优先级:
  Option A (首选): elkai (LKH Python binding) — 对 n=50 通常找到精确最优
  Option B (回退): 多起点 2-opt (20 次随机初始化解, 取最优)
  Option C (最后): Held-Karp 下界 (1-tree bound)
"""
import os
import sys
import io
import csv
import time
import numpy as np

# 强制 stdout UTF-8（必须在导入其他模块前设置，因为 solve_all 也会重设 stdout）
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except (ValueError, AttributeError):
        pass  # stdout already wrapped by imported module

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import load_drill_data, get_coords
from src.utils.distance import euclidean_distance_matrix, total_path_length
from src.problem1.solve_all import nearest_neighbor, two_opt

np.random.seed(42)

N = 50
OUR_BEST_MM = 1927.2947     # SA (known result for n=50 from p1_results.csv)
OUR_BEST_ALGO = "SA"

# ── Load data ──
print("=" * 60)
print("  Problem 1: Optimality Lower Bound Check (n=50)")
print("=" * 60)

data = load_drill_data(N)
drill_coords = get_coords(data)
origin = np.array([[0.0, 0.0]])
coords = np.vstack([origin, drill_coords])          # (51, 2), index 0 = origin
dist_matrix = euclidean_distance_matrix(coords)

print(f"  Loaded n={N}, coords shape={coords.shape}")

# ═══════════════════════════════════════════════════════════════════
# Option A: LKH via elkai
# ═══════════════════════════════════════════════════════════════════

def compute_lkh_optimal(dist_matrix: np.ndarray, n: int):
    """
    使用 elkai (LKH) 计算精确或极度近似最优解。
    对于 n=50，LKH 几乎总是找到全局最优。
    使用新版 elkai API: DistanceMatrix 类。
    """
    try:
        from elkai import DistanceMatrix
    except ImportError:
        return None, "elkai not installed"

    try:
        t0 = time.perf_counter()
        # New elkai API: DistanceMatrix takes list-of-lists, solve_tsp(runs=N)
        dm = DistanceMatrix(dist_matrix.tolist())
        solution = dm.solve_tsp(runs=10)
        t1 = time.perf_counter()

        # solution is a list of node indices e.g. [0, 3, 1, 2, ...]
        # Doesn't include return to origin — we need to close the loop
        # But note: the solution may not start at 0. Rotate.
        if solution[0] != 0:
            origin_idx = solution.index(0)
            solution = solution[origin_idx:] + solution[:origin_idx]
        full_order = solution + [0]  # return to origin
        opt_dist = total_path_length(full_order, dist_matrix)

        print(f"  [LKH] order (first 10): {full_order[:10]}")
        print(f"  [LKH] distance = {opt_dist:.4f} mm, CPU = {t1 - t0:.2f}s")
        return opt_dist, "LKH (elkai, 10 runs)"
    except Exception as e:
        print(f"  [LKH] Error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"LKH error: {e}"


# ═══════════════════════════════════════════════════════════════════
# Option B: Multi-start 2-opt (20 random restarts)
# ═══════════════════════════════════════════════════════════════════

def compute_multistart_2opt(dist_matrix: np.ndarray, n: int, num_starts: int = 30):
    """从多个随机初始化解出发运行 2-opt，返回最优结果。"""
    best_order = None
    best_dist = float('inf')

    t0 = time.perf_counter()
    for run in range(num_starts):
        # Random permutation starting and ending at origin
        perm = [0] + list(np.random.permutation(np.arange(1, n + 1))) + [0]
        opt_order = two_opt(perm, dist_matrix, n)
        opt_dist = total_path_length(opt_order, dist_matrix)

        if opt_dist < best_dist:
            best_dist = opt_dist
            best_order = opt_order

    t1 = time.perf_counter()
    print(f"  [Multi-start 2-opt] {num_starts} runs, best={best_dist:.4f} mm, CPU={t1 - t0:.2f}s")
    return best_dist, f"Multi-start 2-opt ({num_starts} restarts)"


# ═══════════════════════════════════════════════════════════════════
# Option C: Held-Karp lower bound (1-tree relaxation)
# ═══════════════════════════════════════════════════════════════════

def compute_held_karp_lower_bound(dist_matrix: np.ndarray, n: int):
    """
    Held-Karp lower bound using 1-tree relaxation with subgradient optimization.
    This is a PROVEN lower bound — the true optimum cannot be below this.
    """
    V = n + 1  # vertices including origin
    # Exclude node 0 from the MST computation
    # 1-tree = MST on V\{0} + two cheapest edges incident to 0

    # Step 0: Initialize node penalties (dual variables)
    pi = np.zeros(V)
    best_lower_bound = 0.0
    step_size = 10.0
    alpha = 2.0
    max_iter = 200

    for iter_count in range(max_iter):
        # Compute modified distances: d'(i,j) = d(i,j) + pi[i] + pi[j]
        modified = dist_matrix.copy()
        for i in range(V):
            modified[i, :] += pi[i]
            modified[:, i] += pi[i]

        # Build MST on vertices {1, ..., n} (exclude vertex 0)
        in_tree = {1}
        not_in_tree = set(range(2, V))
        mst_cost = 0.0
        mst_degree = np.zeros(V, dtype=int)  # degrees in the MST

        while not_in_tree:
            # Find cheapest edge from tree to any vertex not in tree
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

        # Add two cheapest edges incident to vertex 0 (to complete the 1-tree)
        edges_to_0 = [(modified[0, j], j) for j in range(1, V)]
        edges_to_0.sort()
        one_tree_cost = mst_cost + edges_to_0[0][0] + edges_to_0[1][0]

        # Track vertices used for edge 0
        one_tree_degree = mst_degree.copy()
        one_tree_degree[0] = 2
        one_tree_degree[edges_to_0[0][1]] += 1
        one_tree_degree[edges_to_0[1][1]] += 1

        # Compute subgradient: degree[i] - 2 for all i
        subgradient = one_tree_degree - 2

        # The true lower bound with penalties
        lb = one_tree_cost - 2.0 * np.sum(pi)

        if lb > best_lower_bound:
            best_lower_bound = lb

        # Subgradient update
        sg_norm = np.sqrt(np.sum(subgradient ** 2))
        if sg_norm < 1e-8:
            break

        # Step size rule: alpha^k * step_size / ||subgradient||
        current_step = alpha * step_size / sg_norm
        pi += current_step * subgradient

        # Diminish step size
        step_size *= 0.95

    print(f"  [Held-Karp] {max_iter} iterations, lower bound={best_lower_bound:.4f} mm")
    return best_lower_bound, f"Held-Karp 1-tree ({max_iter} iterations)"


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

optimal_dist = None
optimal_method = ""
gap_pct = None

print(f"\n{'─' * 60}")
print(f"  Our best result: {OUR_BEST_MM:.4f} mm ({OUR_BEST_ALGO})")
print(f"{'─' * 60}")

# --- Option A: LKH ---
print("\n  --- Option A: LKH (elkai) ---")
lkh_dist, lkh_method = compute_lkh_optimal(dist_matrix, N)

if lkh_dist is not None:
    optimal_dist = lkh_dist
    optimal_method = lkh_method

if optimal_dist is None:
    # --- Option B: Multi-start 2-opt ---
    print("\n  --- Option B: Multi-start 2-opt fallback ---")
    ms_dist, ms_method = compute_multistart_2opt(dist_matrix, N, num_starts=30)
    optimal_dist = ms_dist
    optimal_method = ms_method

if optimal_dist is None:
    # --- Option C: Held-Karp ---
    print("\n  --- Option C: Held-Karp lower bound fallback ---")
    hk_dist, hk_method = compute_held_karp_lower_bound(dist_matrix, N)
    optimal_dist = hk_dist
    optimal_method = hk_method

# ═══════════════════════════════════════════════════════════════════
# Compute gap
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"  Results")
print(f"{'=' * 60}")

if optimal_dist is not None:
    gap_mm = OUR_BEST_MM - optimal_dist
    gap_pct = (OUR_BEST_MM - optimal_dist) / optimal_dist * 100.0

    print(f"  Our best:           {OUR_BEST_MM:>12.4f} mm  ({OUR_BEST_ALGO})")
    print(f"  Optimal (approx):   {optimal_dist:>12.4f} mm  ({optimal_method})")
    print(f"  Absolute gap:       {gap_mm:>12.4f} mm")
    print(f"  Relative gap:       {gap_pct:>12.4f} %")
    print(f"")

    if gap_pct < 1.0:
        print(f"  ✓  Gap < 1%: 2-opt/SA is nearly optimal for this scale.")
        print(f"     This is a strong claim for the paper.")
    elif gap_pct < 5.0:
        print(f"  ○  Gap 1-5%: within expected range for heuristic methods.")
    else:
        print(f"  ⚠ Gap > 5%: investigate — possibly the SA parameters need tuning.")
else:
    print("  ERROR: Could not compute any optimality reference.")
    gap_pct = None
    optimal_dist = None

# ═══════════════════════════════════════════════════════════════════
# Save to CSV
# ═══════════════════════════════════════════════════════════════════
tables_dir = os.path.join(PROJECT_ROOT, "output", "tables")
os.makedirs(tables_dir, exist_ok=True)
csv_path = os.path.join(tables_dir, "p1_optimality_gap.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "Our_Best_mm", "Our_Best_Algo", "Optimal_mm", "Optimal_Method",
                      "Gap_pct"])
    writer.writerow([
        N,
        f"{OUR_BEST_MM:.4f}",
        OUR_BEST_ALGO,
        f"{optimal_dist:.4f}" if optimal_dist is not None else "N/A",
        optimal_method,
        f"{gap_pct:.4f}" if gap_pct is not None else "N/A"
    ])

print(f"  [CSV] saved: {csv_path}")
print(f"\n{'=' * 60}")
print(f"  Done!")
print(f"{'=' * 60}")
