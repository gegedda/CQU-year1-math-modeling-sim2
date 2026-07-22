"""
问题一：PCB 钻孔路径优化求解器
实现 NN（最近邻）、2-opt 局部搜索、SA（模拟退火）三种算法
在 n=50, 198, 442, 1173 四个规模下运行
"""
import os
import sys
import io
import time
import numpy as np
import csv

# 将项目根目录加入 sys.path，以便导入 utils 模块
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import load_drill_data, get_coords
from src.utils.distance import euclidean_distance_matrix, total_path_length
from src.utils.visualization import plot_path, plot_convergence, plot_comparison

# ── 随机种子 ──
np.random.seed(42)

# ── 常数 ──
SPEED = 100.0          # mm/s
SCALES = [50, 198, 442, 1173]


# ═══════════════════════════════════════════════════════════════════
# 算法 A: 最近邻 (Nearest Neighbor)
# ═══════════════════════════════════════════════════════════════════
def nearest_neighbor(dist_matrix: np.ndarray, n: int) -> list:
    """
    贪心构造 TSP 路径，从原点（索引 0）出发，每次选最近未访问点。
    
    参数:
        dist_matrix: (n+1)×(n+1) 距离矩阵，索引 0 为原点
        n: 钻孔数量
    返回:
        order: 访问顺序 [0, ..., 0]，长度为 n+2
    """
    unvisited = set(range(1, n + 1))
    order = [0]
    cur = 0
    
    while unvisited:
        # 找到距离 cur 最近的未访问点
        next_pt = min(unvisited, key=lambda j: dist_matrix[cur, j])
        order.append(next_pt)
        unvisited.remove(next_pt)
        cur = next_pt
    
    order.append(0)  # 返回原点
    return order


# ═══════════════════════════════════════════════════════════════════
# 算法 B: 2-opt 局部搜索
# ═══════════════════════════════════════════════════════════════════
def two_opt(order: list, dist_matrix: np.ndarray, n: int) -> list:
    """
    从给定解出发，反复进行 2-opt 交换直到无法改进。
    反转区间 order[i:j+1] 以消除交叉边。

    参数:
        order: 初始访问顺序 [0, ..., 0]
        dist_matrix: 距离矩阵
        n: 钻孔数量
    返回:
        order: 改进后的访问顺序
    """
    order = list(order)  # 不修改原列表
    improved = True
    iteration = 0
    
    while improved:
        improved = False
        iteration += 1
        
        # 遍历所有可能的 (i, j) 对
        # i 从 1 到 n-1, j 从 i+1 到 n
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                old_edges = (
                    dist_matrix[order[i - 1], order[i]]
                    + dist_matrix[order[j], order[j + 1]]
                )
                new_edges = (
                    dist_matrix[order[i - 1], order[j]]
                    + dist_matrix[order[i], order[j + 1]]
                )
                
                if new_edges < old_edges:
                    # 反转区间 [i, j]
                    order[i:j + 1] = reversed(order[i:j + 1])
                    improved = True
    
    return order


# ═══════════════════════════════════════════════════════════════════
# 算法 C: 模拟退火 (Simulated Annealing)
# ═══════════════════════════════════════════════════════════════════
def simulated_annealing(order: list, dist_matrix: np.ndarray, n: int,
                        T0: float = 10000.0, T_min: float = 0.1,
                        alpha: float = 0.995, iters_per_T: int = 50,
                        verbose: bool = False):
    """
    模拟退火求解 TSP，使用随机 2-opt 作为邻域移动。

    参数:
        order: 初始访问顺序（来自 NN）
        dist_matrix: 距离矩阵
        n: 钻孔数量
        T0: 初始温度
        T_min: 终止温度
        alpha: 冷却因子
        iters_per_T: 每个温度步的迭代次数
        verbose: 是否输出进度信息
    返回:
        best_order: 最优访问顺序
        best_cost: 最优路径长度
        history: 各温度步的最优距离历史
    """
    order = list(order)
    current_cost = total_path_length(order, dist_matrix)
    
    best_order = list(order)
    best_cost = current_cost
    
    history = []
    T = T0
    step = 0
    accept_count = 0
    improve_count = 0
    total_moves = 0
    
    while T > T_min:
        for _ in range(iters_per_T):
            total_moves += 1
            # 随机选择 2-opt 区间
            i = np.random.randint(1, n)           # 1 .. n-1
            j = np.random.randint(i + 1, n + 1)   # i+1 .. n
            
            old_edges = (
                dist_matrix[order[i - 1], order[i]]
                + dist_matrix[order[j], order[j + 1]]
            )
            new_edges = (
                dist_matrix[order[i - 1], order[j]]
                + dist_matrix[order[i], order[j + 1]]
            )
            delta = new_edges - old_edges
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                # 接受该移动
                order[i:j + 1] = list(reversed(order[i:j + 1]))
                current_cost += delta
                accept_count += 1
                
                if delta < 0:
                    improve_count += 1
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_order = list(order)
        
        history.append(best_cost)
        
        if verbose and step % 200 == 0:
            acc_rate = accept_count / max(total_moves, 1) * 100
            imp_rate = improve_count / max(accept_count, 1) * 100
            print(f"    SA T={T:.1f} step={step:>5d} best={best_cost:.1f} "
                  f"acc_rate={acc_rate:.1f}% imp_rate={imp_rate:.1f}%", flush=True)
        
        step += 1
        T *= alpha
    
    # Sanity check: recompute best_cost from actual order
    verified_cost = total_path_length(best_order, dist_matrix)
    if abs(verified_cost - best_cost) > 0.01:
        print(f"    WARNING: cost tracking drifted! tracked={best_cost:.2f} actual={verified_cost:.2f}")
        best_cost = verified_cost
    
    return best_order, best_cost, history


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════
def build_problem(n: int):
    """加载数据、构建坐标数组（含原点）和距离矩阵"""
    data = load_drill_data(n)
    drill_coords = get_coords(data)                     # (n, 2)
    origin = np.array([[0.0, 0.0]])
    coords = np.vstack([origin, drill_coords])          # (n+1, 2)，索引 0 为原点
    dist_matrix = euclidean_distance_matrix(coords)     # (n+1, n+1)
    return coords, dist_matrix


def print_result(n: int, algo: str, distance: float, elapsed: float = 0.0):
    """格式化打印单次运行结果"""
    tt = distance / SPEED
    timing = f", CPU={elapsed:.2f}s" if elapsed > 0 else ""
    print(f"  [{algo:>5s}  n={n:>4d}]  distance={distance:>12.2f} mm  time={tt:>8.2f} s{timing}")


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 强制 stdout 使用 UTF-8，避免 Windows GBK 编码错误（仅在直接运行时）
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("=" * 72)
    print("  Problem 1: PCB Drill Path Optimization -- TSP Solver")
    print("  Algorithms: Nearest Neighbor | 2-opt | Simulated Annealing")
    print("  Scales: n = 50, 198, 442, 1173")
    print("=" * 72)
    
    # 存储所有结果:  {(n, algo): (distance, order), ...}
    all_results = {}
    # 存储 SA 收敛历史:  {n: history}
    sa_histories = {}
    # 存储各规模坐标（用于绘图）
    all_coords = {}
    # 存储对比数据:  {algo: [dist50, dist198, dist442, dist1173]}
    comparison_data = {"NN": [], "2-opt": [], "SA": [], "SA-2opt": []}
    
    for n in SCALES:
        print(f"\n{'-' * 60}")
        print(f"  >> Scale n = {n}")
        print(f"{'-' * 60}")
        
        coords, dist_matrix = build_problem(n)
        all_coords[n] = coords
        
        # --- A: Nearest Neighbor ---
        print(f"  [NN n={n:>4d}] running...", end="", flush=True)
        t0 = time.perf_counter()
        nn_order = nearest_neighbor(dist_matrix, n)
        nn_dist = total_path_length(nn_order, dist_matrix)
        t1 = time.perf_counter()
        print(f"\r  ", end="")
        print_result(n, "NN", nn_dist, t1 - t0)
        all_results[(n, "NN")] = (nn_dist, nn_order)
        
        # --- B: 2-opt ---
        print(f"  [2-opt n={n:>4d}] running...", end="", flush=True)
        t0 = time.perf_counter()
        opt_order = two_opt(nn_order, dist_matrix, n)
        opt_dist = total_path_length(opt_order, dist_matrix)
        t1 = time.perf_counter()
        print(f"\r  ", end="")
        print_result(n, "2-opt", opt_dist, t1 - t0)
        all_results[(n, "2-opt")] = (opt_dist, opt_order)
        
        # --- C: SA ---
        # Scale parameters with problem size for better convergence
        if n <= 50:
            sa_alpha, sa_iters = 0.995, 50
        elif n <= 200:
            sa_alpha, sa_iters = 0.998, 100
        elif n <= 500:
            sa_alpha, sa_iters = 0.999, 200
        else:
            sa_alpha, sa_iters = 0.9995, 300
        
        print(f"  [SA  n={n:>4d}] running (alpha={sa_alpha}, iters/T={sa_iters})...", end="", flush=True)
        t0 = time.perf_counter()
        sa_order, sa_dist, history = simulated_annealing(
            nn_order, dist_matrix, n,
            alpha=sa_alpha, iters_per_T=sa_iters, verbose=False
        )
        t1 = time.perf_counter()
        print(f"\r  ", end="")
        print_result(n, "SA", sa_dist, t1 - t0)
        all_results[(n, "SA")] = (sa_dist, sa_order)
        sa_histories[n] = history
        
        # --- D: SA from 2-opt (hybrid) --- only for large n where SA-from-NN loses to 2-opt
        if n >= 442:
            print(f"  [SA2 n={n:>4d}] running (starting from 2-opt, alpha={sa_alpha}, iters/T={sa_iters})...",
                  end="", flush=True)
            t0 = time.perf_counter()
            sa2_order, sa2_dist, sa2_history = simulated_annealing(
                opt_order, dist_matrix, n,
                alpha=sa_alpha, iters_per_T=sa_iters, verbose=False
            )
            t1 = time.perf_counter()
            print(f"\r  ", end="")
            print_result(n, "SA2", sa2_dist, t1 - t0)
            all_results[(n, "SA-2opt")] = (sa2_dist, sa2_order)
        else:
            # For small n, SA already beats 2-opt; copy SA result for consistent tables
            all_results[(n, "SA-2opt")] = (sa_dist, sa_order)
    
    # Build comparison data from all_results (after loop)
    comparison_data = {"NN": [], "2-opt": [], "SA": [], "SA-2opt": []}
    for n in SCALES:
        for algo in ["NN", "2-opt", "SA", "SA-2opt"]:
            comparison_data[algo].append(all_results[(n, algo)][0])
    
    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 72}")
    print(f"  Summary Results")
    print(f"{'=' * 72}")
    header = f"  {'n':>5s}  {'Algorithm':>6s}  {'Distance (mm)':>16s}  {'Time (s)':>10s}"
    print(header)
    print(f"  {'─'*5}  {'─'*6}  {'─'*16}  {'─'*10}")
    
    for n in SCALES:
        for algo in ["NN", "2-opt", "SA", "SA-2opt"]:
            dist, _ = all_results[(n, algo)]
            tt = dist / SPEED
            print(f"  {n:>5d}  {algo:>6s}  {dist:>16.2f}  {tt:>10.2f}")
    
    # ================================================================
    # Save CSV
    # ================================================================
    tables_dir = os.path.join(PROJECT_ROOT, "output", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, "p1_results.csv")
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "Algorithm", "Total_Distance_mm", "Total_Time_s"])
        for n in SCALES:
            for algo in ["NN", "2-opt", "SA", "SA-2opt"]:
                dist, _ = all_results[(n, algo)]
                tt = dist / SPEED
                writer.writerow([n, algo, f"{dist:.4f}", f"{tt:.4f}"])
    
    print(f"\n  [CSV] saved: {csv_path}")
    
    # ================================================================
    # Visualizations
    # ================================================================
    print(f"\n{'=' * 72}")
    print(f"  Generating figures...")
    print(f"{'=' * 72}")
    
    # 1) Best path for each scale (use the algorithm with minimum distance)
    for n in SCALES:
        coords = all_coords[n]
        # Find the best algorithm for this scale
        best_algo = min(["NN", "2-opt", "SA", "SA-2opt"], key=lambda algo: all_results[(n, algo)][0])
        best_dist, best_order = all_results[(n, best_algo)]
        tt = best_dist / SPEED
        plot_path(
            coords, best_order,
            title=f"Problem 1 Optimal Path ({best_algo}, n={n})\nTotal distance: {best_dist:.1f} mm  Total time: {tt:.2f} s",
            filename=f"p1_path_n{n}.png"
        )
    
    # 2) SA convergence curves
    for n in SCALES:
        history = sa_histories[n]
        plot_convergence(
            history,
            title=f"Problem 1 SA Convergence (n={n})",
            filename=f"p1_convergence_n{n}.png"
        )
    
    # 3) Algorithm comparison bar chart
    plot_comparison(
        comparison_data,
        title="Problem 1 Algorithm Comparison Across Scales",
        filename="p1_algorithm_comparison.png"
    )
    
    # ================================================================
    # Verification
    # ================================================================
    print(f"\n{'=' * 72}")
    print(f"  Verification")
    print(f"{'=' * 72}")
    for n in SCALES:
        nn_dist, _ = all_results[(n, "NN")]
        sa_dist, _ = all_results[(n, "SA")]
        improvement = (nn_dist - sa_dist) / nn_dist * 100.0
        status = "OK" if improvement >= 5.0 else "FAIL"
        print(f"  n={n:>4d}:  SA vs NN  improvement {improvement:.2f}%  {status}  (target: >=5%)")
    
    print(f"\n{'=' * 72}")
    print(f"  Done! All results saved to output/")
    print(f"{'=' * 72}")
