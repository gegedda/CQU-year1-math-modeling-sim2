"""
Problem 1: Multi-Algorithm Cross-Comparison
Algorithms: NN, Greedy Insertion, 2-opt, SA
Scales: n = 50, 198, 442, 1173

Output:
  - Formatted summary table (stdout)
  - output/tables/p1_full_comparison.csv
  - output/figures/p1_full_quality.png     (bar chart)
  - output/figures/p1_tradeoff.png          (scatter: time vs quality)
  - output/figures/p1_scalability.png       (line: n vs CPU time)
"""
import os
import sys
import io
import time
import csv
import numpy as np
import warnings

# ── Force UTF-8 stdout ──
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Project paths ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import load_drill_data, get_coords
from src.utils.distance import euclidean_distance_matrix, total_path_length

# ── Plotting setup ──
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Try CJK font
for font_name in ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']:
    try:
        plt.rcParams['font.sans-serif'] = [font_name]
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

# ── Constants ──
SCALES = [50, 198, 442, 1173]
SEED = 42
ALGORITHMS = ["NN", "Greedy", "2-opt", "SA"]

# SA parameter scaling (same as solve_all.py)
SA_CONFIG = {
    50:  dict(alpha=0.995,  iters_per_T=50),
    198: dict(alpha=0.998,  iters_per_T=100),
    442: dict(alpha=0.999,  iters_per_T=200),
    1173:dict(alpha=0.9995, iters_per_T=300),
}


# ═══════════════════════════════════════════════════════════════════
# ALGORITHM A: Nearest Neighbor (copied from solve_all.py)
# ═══════════════════════════════════════════════════════════════════
def nearest_neighbor(dist_matrix: np.ndarray, n: int) -> list:
    unvisited = set(range(1, n + 1))
    order = [0]
    cur = 0
    while unvisited:
        next_pt = min(unvisited, key=lambda j: dist_matrix[cur, j])
        order.append(next_pt)
        unvisited.remove(next_pt)
        cur = next_pt
    order.append(0)
    return order


# ═══════════════════════════════════════════════════════════════════
# ALGORITHM B: Greedy Insertion — NEW / MUST-IMPLEMENT
# ═══════════════════════════════════════════════════════════════════
def greedy_insertion(dist_matrix: np.ndarray, n: int) -> list:
    """
    Standard greedy insertion TSP heuristic.

    1. Start with subtour: origin 0 → nearest point → origin 0
    2. While unvisited points remain:
       - For each unvisited point k, find the insertion edge (i, j)
         that minimizes: dist[i,k] + dist[k,j] - dist[i,j]
       - Insert k between i and j
    3. Return complete tour [0, ..., 0]

    Complexity: O(n³)
    """
    unvisited = set(range(1, n + 1))

    # Step 1: initial subtour (origin → nearest → origin)
    nearest_to_origin = min(unvisited, key=lambda j: dist_matrix[0, j])
    unvisited.remove(nearest_to_origin)
    tour = [0, nearest_to_origin, 0]

    # Step 2: iteratively insert remaining points
    while unvisited:
        best_k = None
        best_pos = None
        best_cost = float('inf')

        for k in unvisited:
            # For each insertion position (i = tour[pos], j = tour[pos+1])
            for pos in range(len(tour) - 1):
                i = tour[pos]
                j = tour[pos + 1]
                # Insertion cost: how much longer the path becomes
                cost = dist_matrix[i, k] + dist_matrix[k, j] - dist_matrix[i, j]
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
                    best_pos = pos

        # Insert best_k between best_pos and best_pos+1
        tour.insert(best_pos + 1, best_k)
        unvisited.remove(best_k)

    return tour


# ═══════════════════════════════════════════════════════════════════
# ALGORITHM C: 2-opt Local Search (copied from solve_all.py)
# ═══════════════════════════════════════════════════════════════════
def two_opt(order: list, dist_matrix: np.ndarray, n: int) -> list:
    order = list(order)
    improved = True
    while improved:
        improved = False
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
                    order[i:j + 1] = reversed(order[i:j + 1])
                    improved = True
    return order


# ═══════════════════════════════════════════════════════════════════
# ALGORITHM D: Simulated Annealing (copied from solve_all.py)
# ═══════════════════════════════════════════════════════════════════
def simulated_annealing(order: list, dist_matrix: np.ndarray, n: int,
                        T0: float = 10000.0, T_min: float = 0.1,
                        alpha: float = 0.995, iters_per_T: int = 50):
    order = list(order)
    current_cost = total_path_length(order, dist_matrix)
    best_order = list(order)
    best_cost = current_cost
    T = T0
    total_moves = 0
    accept_count = 0
    improve_count = 0

    while T > T_min:
        for _ in range(iters_per_T):
            total_moves += 1
            i = np.random.randint(1, n)
            j = np.random.randint(i + 1, n + 1)
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
                order[i:j + 1] = list(reversed(order[i:j + 1]))
                current_cost += delta
                accept_count += 1
                if delta < 0:
                    improve_count += 1
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_order = list(order)
        T *= alpha

    # Sanity check
    verified_cost = total_path_length(best_order, dist_matrix)
    if abs(verified_cost - best_cost) > 0.01:
        best_cost = verified_cost

    return best_order, best_cost


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def build_problem(n: int):
    """Load data, build coordinate array (with origin) and distance matrix."""
    data = load_drill_data(n)
    drill_coords = get_coords(data)                     # (n, 2)
    origin = np.array([[0.0, 0.0]])
    coords = np.vstack([origin, drill_coords])          # (n+1, 2), index 0 = origin
    dist_matrix = euclidean_distance_matrix(coords)
    return coords, dist_matrix


def fmt_dist(d: float) -> str:
    return f"{d:>15.2f}"


def fmt_time(t: float) -> str:
    return f"{t:>10.3f}"


def fmt_tpp(t: float) -> str:
    return f"{t:>14.3f}"


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 90)
    print("  Problem 1: Multi-Algorithm Cross-Comparison")
    print("  Algorithms: NN | Greedy Insertion | 2-opt | SA")
    print("  Scales: n = 50, 198, 442, 1173")
    print("=" * 90)

    # results: list of dicts
    results = []

    for n in SCALES:
        print(f"\n{'─' * 70}")
        print(f"  >> Scale n = {n}")
        print(f"{'─' * 70}")

        coords, dist_matrix = build_problem(n)
        sa_cfg = SA_CONFIG[n]

        # ── NN ──
        np.random.seed(SEED)
        print(f"  [NN  n={n:>4d}] running...", end="", flush=True)
        t0 = time.time()
        nn_order = nearest_neighbor(dist_matrix, n)
        nn_dist = total_path_length(nn_order, dist_matrix)
        t1 = time.time()
        cpu_nn = t1 - t0
        print(f"\r  [NN  n={n:>4d}] distance={nn_dist:>12.2f}, CPU={cpu_nn:.3f}s")

        results.append(dict(n=n, algo="NN", dist=nn_dist, cpu=cpu_nn, tpp=cpu_nn / n * 1000))

        # ── Greedy Insertion ──
        np.random.seed(SEED)
        print(f"  [GI  n={n:>4d}] running...", end="", flush=True)
        t0 = time.time()
        gi_order = greedy_insertion(dist_matrix, n)
        gi_dist = total_path_length(gi_order, dist_matrix)
        t1 = time.time()
        cpu_gi = t1 - t0
        print(f"\r  [GI  n={n:>4d}] distance={gi_dist:>12.2f}, CPU={cpu_gi:.3f}s")

        results.append(dict(n=n, algo="Greedy", dist=gi_dist, cpu=cpu_gi, tpp=cpu_gi / n * 1000))

        # ── 2-opt (from NN) ──
        np.random.seed(SEED)
        print(f"  [2-opt n={n:>4d}] running...", end="", flush=True)
        t0 = time.time()
        opt_order = two_opt(nn_order, dist_matrix, n)
        opt_dist = total_path_length(opt_order, dist_matrix)
        t1 = time.time()
        cpu_opt = t1 - t0
        print(f"\r  [2-opt n={n:>4d}] distance={opt_dist:>12.2f}, CPU={cpu_opt:.3f}s")

        results.append(dict(n=n, algo="2-opt", dist=opt_dist, cpu=cpu_opt, tpp=cpu_opt / n * 1000))

        # ── SA (from NN) ──
        np.random.seed(SEED)
        print(f"  [SA  n={n:>4d}] running (alpha={sa_cfg['alpha']}, iters/T={sa_cfg['iters_per_T']})...",
              end="", flush=True)
        t0 = time.time()
        sa_order, sa_dist = simulated_annealing(
            nn_order, dist_matrix, n,
            alpha=sa_cfg['alpha'], iters_per_T=sa_cfg['iters_per_T']
        )
        t1 = time.time()
        cpu_sa = t1 - t0
        print(f"\r  [SA  n={n:>4d}] distance={sa_dist:>12.2f}, CPU={cpu_sa:.3f}s")

        results.append(dict(n=n, algo="SA", dist=sa_dist, cpu=cpu_sa, tpp=cpu_sa / n * 1000))

    # ═══════════════════════════════════════════════════════════════
    # PRINT SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  SUMMARY — All 16 Results")
    print("=" * 90)
    header = (f"  {'n':>5s}  {'Algorithm':>8s}  "
              f"{'Distance (mm)':>16s}  {'CPU (s)':>10s}  {'Time/pt (ms)':>14s}")
    print(header)
    print(f"  {'─'*5}  {'─'*8}  {'─'*16}  {'─'*10}  {'─'*14}")

    for r in results:
        print(f"  {r['n']:>5d}  {r['algo']:>8s}  {r['dist']:>16.2f}  "
              f"{r['cpu']:>10.3f}  {r['tpp']:>14.3f}")

    # ═══════════════════════════════════════════════════════════════
    # SANITY CHECK
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  SANITY CHECKS")
    print("=" * 90)
    for n in SCALES:
        nn_d = [r for r in results if r['n'] == n and r['algo'] == 'NN'][0]['dist']
        gi_d = [r for r in results if r['n'] == n and r['algo'] == 'Greedy'][0]['dist']
        opt_d = [r for r in results if r['n'] == n and r['algo'] == '2-opt'][0]['dist']
        sa_d = [r for r in results if r['n'] == n and r['algo'] == 'SA'][0]['dist']

        gi_vs_nn = "OK" if gi_d < nn_d else "FAIL"
        opt_vs_gi = "OK" if opt_d < gi_d else "FAIL"
        sa_vs_opt = "OK" if sa_d <= opt_d else "OK (SA>=2-opt is acceptable)"
        print(f"  n={n:>4d}:  GI < NN? {gi_vs_nn}  "
              f"2-opt < GI? {opt_vs_gi}  SA <= 2-opt? {sa_vs_opt}")

    # ═══════════════════════════════════════════════════════════════
    # SAVE CSV
    # ═══════════════════════════════════════════════════════════════
    tables_dir = os.path.join(PROJECT_ROOT, "output", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, "p1_full_comparison.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "Algorithm", "Total_Distance_mm", "CPU_Time_s", "Time_Per_Point_ms"])
        for r in results:
            writer.writerow([r['n'], r['algo'], f"{r['dist']:.4f}",
                             f"{r['cpu']:.6f}", f"{r['tpp']:.4f}"])

    print(f"\n  [CSV] saved: {csv_path}")

    # ═══════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════
    figs_dir = os.path.join(PROJECT_ROOT, "output", "figures")
    os.makedirs(figs_dir, exist_ok=True)

    colors = {"NN": "#4472C4", "Greedy": "#ED7D31", "2-opt": "#A5A5A5", "SA": "#FFC000"}
    markers = {50: "s", 198: "^", 442: "o", 1173: "D"}
    sizes = {50: 80, 198: 120, 442: 160, 1173: 200}

    x_labels = [str(s) for s in SCALES]
    bar_width = 0.18
    x = np.arange(len(SCALES))

    # ── Plot 1: Quality Comparison (Grouped Bar) ──
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for idx, algo in enumerate(ALGORITHMS):
        dists = [r['dist'] for r in results if r['algo'] == algo]
        offset = (idx - 1.5) * bar_width
        bars = ax1.bar(x + offset, dists, bar_width, label=algo,
                       color=colors[algo], edgecolor='white', linewidth=0.5)
        # Value labels on bars
        for bar, val in zip(bars, dists):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(dists) * 0.005,
                     f"{val:,.0f}", ha='center', va='bottom', fontsize=7, rotation=90)

    ax1.set_xlabel("Number of Drill Points (n)", fontsize=13)
    ax1.set_ylabel("Total Path Distance (mm)", fontsize=13)
    ax1.set_title("Problem 1: Algorithm Quality Comparison Across Scales", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=12)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f"{val:,.0f}"))
    fig1.tight_layout()
    fig1.savefig(os.path.join(figs_dir, "p1_full_quality.png"), dpi=150)
    plt.close(fig1)
    print(f"  [FIG] saved: p1_full_quality.png")

    # ── Plot 2: Time-Quality Tradeoff (Scatter) ──
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    for algo in ALGORITHMS:
        pts = [r for r in results if r['algo'] == algo]
        xs = [r['cpu'] for r in pts]
        ys = [r['dist'] for r in pts]
        ss = [sizes[r['n']] for r in pts]
        ax2.scatter(xs, ys, s=ss, c=colors[algo], alpha=0.85, edgecolors='white',
                    linewidth=0.5, label=algo, zorder=3)

    # Add algorithm labels near points
    for algo in ALGORITHMS:
        for r in results:
            if r['algo'] == algo:
                ax2.annotate(f"{algo}\nn={r['n']}",
                             (r['cpu'], r['dist']),
                             textcoords="offset points", xytext=(10, 5),
                             fontsize=8, color=colors[algo], fontweight='bold',
                             alpha=0.9)

    ax2.set_xscale('log')
    ax2.set_xlabel("CPU Time (seconds, log scale)", fontsize=13)
    ax2.set_ylabel("Total Path Distance (mm)", fontsize=13)
    ax2.set_title("Problem 1: Time-Quality Tradeoff", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower left')
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f"{val:,.0f}"))
    # Scale marker legend
    for n_val in SCALES:
        ax2.scatter([], [], s=sizes[n_val], c='gray', alpha=0.6, edgecolors='white',
                    linewidth=0.5, label=f"n={n_val}")
    fig2.tight_layout()
    fig2.savefig(os.path.join(figs_dir, "p1_tradeoff.png"), dpi=150)
    plt.close(fig2)
    print(f"  [FIG] saved: p1_tradeoff.png")

    # ── Plot 3: Scalability (Line Plot) ──
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for algo in ALGORITHMS:
        pts = sorted([r for r in results if r['algo'] == algo], key=lambda r: r['n'])
        xs = [r['n'] for r in pts]
        ys = [r['cpu'] for r in pts]
        ax3.plot(xs, ys, 'o-', color=colors[algo], linewidth=2, markersize=8,
                 label=algo, zorder=3)
        # Label end points
        ax3.annotate(f"{algo}", (xs[-1], ys[-1]),
                     textcoords="offset points", xytext=(10, 0),
                     fontsize=10, color=colors[algo], fontweight='bold')

    ax3.set_xlabel("Number of Drill Points (n)", fontsize=13)
    ax3.set_ylabel("CPU Time (seconds, log scale)", fontsize=13)
    ax3.set_title("Problem 1: Algorithm Scalability", fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend(fontsize=11)
    ax3.grid(True, which='both', ls='--', alpha=0.4)
    fig3.tight_layout()
    fig3.savefig(os.path.join(figs_dir, "p1_scalability.png"), dpi=150)
    plt.close(fig3)
    print(f"  [FIG] saved: p1_scalability.png")

    # ═══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print(f"  COMPLETE — 4 algorithms × 4 scales = {len(results)} results")
    print(f"  Outputs:")
    print(f"    {csv_path}")
    print(f"    {os.path.join(figs_dir, 'p1_full_quality.png')}")
    print(f"    {os.path.join(figs_dir, 'p1_tradeoff.png')}")
    print(f"    {os.path.join(figs_dir, 'p1_scalability.png')}")
    print("=" * 90)
