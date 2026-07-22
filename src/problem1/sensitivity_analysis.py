"""
Problem 1: Simulated Annealing Parameter Sensitivity Analysis
Sweeps α (cooling rate), T0 (initial temperature), and iters_per_T
for n=198 as a representative scale.
"""
import os
import sys
import io
import time
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import load_drill_data, get_coords
from src.utils.distance import euclidean_distance_matrix, total_path_length

# ═══════════════════════════════════════════════════════════════════
# Output dirs
# ═══════════════════════════════════════════════════════════════════
TABLES_DIR = os.path.join(PROJECT_ROOT, "output", "tables")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "output", "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# Matplotlib style
# ═══════════════════════════════════════════════════════════════════
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


# ═══════════════════════════════════════════════════════════════════
# Copied SA function (adapted from solve_all.py, standalone)
# ═══════════════════════════════════════════════════════════════════
def simulated_annealing(order, dist_matrix, n,
                        T0=10000.0, T_min=0.1,
                        alpha=0.995, iters_per_T=50,
                        verbose=False):
    """Simulated annealing for TSP using random 2-opt moves."""
    order = list(order)
    current_cost = total_path_length(order, dist_matrix)

    best_order = list(order)
    best_cost = current_cost

    T = T0
    step = 0

    while T > T_min:
        for _ in range(iters_per_T):
            i = np.random.randint(1, n)
            j = np.random.randint(i + 1, n + 1)

            old_edges = (dist_matrix[order[i - 1], order[i]]
                         + dist_matrix[order[j], order[j + 1]])
            new_edges = (dist_matrix[order[i - 1], order[j]]
                         + dist_matrix[order[i], order[j + 1]])
            delta = new_edges - old_edges

            if delta < 0 or np.random.random() < np.exp(-delta / T):
                order[i:j + 1] = list(reversed(order[i:j + 1]))
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_order = list(order)

        step += 1
        T *= alpha

    # Sanity check
    verified_cost = total_path_length(best_order, dist_matrix)
    if abs(verified_cost - best_cost) > 0.01:
        best_cost = verified_cost

    return best_order, best_cost, step


# ═══════════════════════════════════════════════════════════════════
# Nearest Neighbor (copied from solve_all.py)
# ═══════════════════════════════════════════════════════════════════
def nearest_neighbor(dist_matrix, n):
    """Greedy TSP path starting from origin (index 0)."""
    unvisited = set(range(1, n + 1))
    order = [0]
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist_matrix[cur, j])
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    order.append(0)
    return order


# ═══════════════════════════════════════════════════════════════════
# Build problem for n=198
# ═══════════════════════════════════════════════════════════════════
def build_problem(n):
    data = load_drill_data(n)
    drill_coords = get_coords(data)
    origin = np.array([[0.0, 0.0]])
    coords = np.vstack([origin, drill_coords])
    dist_matrix = euclidean_distance_matrix(coords)
    return coords, dist_matrix


# ═══════════════════════════════════════════════════════════════════
# Sweep runner helpers
# ═══════════════════════════════════════════════════════════════════
def sweep_alpha(dist_matrix, nn_order, n):
    """Sweep 1: Cooling rate α."""
    print("\n" + "=" * 70)
    print("  Sweep 1: Cooling Rate α  (T0=10000, iters_per_T=50, n=198)")
    print("=" * 70)

    alphas = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995]
    results = []  # (alpha, distance, cpu_time, n_steps)

    for a in alphas:
        np.random.seed(42)
        t0 = time.perf_counter()
        _, dist, steps = simulated_annealing(
            nn_order, dist_matrix, n,
            alpha=a, T0=10000.0, iters_per_T=50, verbose=False
        )
        cpu_t = time.perf_counter() - t0
        results.append((a, dist, cpu_t, steps))
        print(f"  [α={a:<7}] distance={dist:>12.2f} mm  CPU={cpu_t:>6.2f}s  steps={steps:>6d}")

    # Console table
    print(f"\n  {'α':>8s}  {'Distance (mm)':>16s}  {'CPU (s)':>10s}  {'Steps':>8s}")
    print(f"  {'─'*8}  {'─'*16}  {'─'*10}  {'─'*8}")
    for row in results:
        print(f"  {row[0]:>8.4f}  {row[1]:>16.2f}  {row[2]:>10.2f}  {row[3]:>8d}")

    # CSV
    csv_path = os.path.join(TABLES_DIR, "p1_sensitivity_alpha.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "distance_mm", "cpu_time_s", "n_steps"])
        w.writerows(results)
    print(f"  [CSV] saved: {csv_path}")

    return results


def sweep_T0(dist_matrix, nn_order, n):
    """Sweep 2: Initial temperature T0."""
    print("\n" + "=" * 70)
    print("  Sweep 2: Initial Temperature T0  (α=0.998, iters_per_T=100, n=198)")
    print("=" * 70)

    T0_vals = [100, 500, 1000, 5000, 10000, 50000, 100000]
    results = []

    for t0_val in T0_vals:
        np.random.seed(42)
        t_start = time.perf_counter()
        _, dist, _ = simulated_annealing(
            nn_order, dist_matrix, n,
            alpha=0.998, T0=t0_val, iters_per_T=100, verbose=False
        )
        cpu_t = time.perf_counter() - t_start
        results.append((t0_val, dist, cpu_t))
        print(f"  [T0={t0_val:<7}] distance={dist:>12.2f} mm  CPU={cpu_t:>6.2f}s")

    print(f"\n  {'T0':>8s}  {'Distance (mm)':>16s}  {'CPU (s)':>10s}")
    print(f"  {'─'*8}  {'─'*16}  {'─'*10}")
    for row in results:
        print(f"  {row[0]:>8d}  {row[1]:>16.2f}  {row[2]:>10.2f}")

    csv_path = os.path.join(TABLES_DIR, "p1_sensitivity_T0.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["T0", "distance_mm", "cpu_time_s"])
        w.writerows(results)
    print(f"  [CSV] saved: {csv_path}")

    return results


def sweep_iters(dist_matrix, nn_order, n):
    """Sweep 3: Iterations per temperature."""
    print("\n" + "=" * 70)
    print("  Sweep 3: Iterations per Temperature  (α=0.998, T0=10000, n=198)")
    print("=" * 70)

    iters_vals = [10, 25, 50, 100, 200, 500, 1000]
    results = []

    for it in iters_vals:
        np.random.seed(42)
        t_start = time.perf_counter()
        _, dist, _ = simulated_annealing(
            nn_order, dist_matrix, n,
            alpha=0.998, T0=10000.0, iters_per_T=it, verbose=False
        )
        cpu_t = time.perf_counter() - t_start
        results.append((it, dist, cpu_t))
        print(f"  [iters/T={it:<5}] distance={dist:>12.2f} mm  CPU={cpu_t:>6.2f}s")

    print(f"\n  {'iters/T':>8s}  {'Distance (mm)':>16s}  {'CPU (s)':>10s}")
    print(f"  {'─'*8}  {'─'*16}  {'─'*10}")
    for row in results:
        print(f"  {row[0]:>8d}  {row[1]:>16.2f}  {row[2]:>10.2f}")

    csv_path = os.path.join(TABLES_DIR, "p1_sensitivity_iters.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iters_per_T", "distance_mm", "cpu_time_s"])
        w.writerows(results)
    print(f"  [CSV] saved: {csv_path}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════

def plot_alpha_sweep(results):
    """Line plot: α vs final distance."""
    alphas = np.array([r[0] for r in results])
    dists = np.array([r[1] for r in results])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, dists, 'o-', color='#2c7bb6', linewidth=1.5, markersize=8)
    ax.set_xlabel(r'Cooling Rate $\alpha$')
    ax.set_ylabel('Final Path Length (mm)')
    ax.set_title('SA Sensitivity: Cooling Rate α vs Solution Quality (n=198)')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Annotate points
    for a, d in zip(alphas, dists):
        ax.annotate(f'{a}', (a, d), textcoords="offset points", xytext=(0, 10),
                    fontsize=8, ha='center')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "p1_sensitivity_alpha.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Figure] saved: {path}")


def plot_T0_sweep(results):
    """Line plot: T0 (log scale) vs final distance."""
    T0s = np.array([r[0] for r in results])
    dists = np.array([r[1] for r in results])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T0s, dists, 'o-', color='#d7191c', linewidth=1.5, markersize=8)
    ax.set_xlabel(r'Initial Temperature $T_0$')
    ax.set_ylabel('Final Path Length (mm)')
    ax.set_title('SA Sensitivity: Initial Temperature T₀ vs Solution Quality (n=198)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    for t0, d in zip(T0s, dists):
        ax.annotate(f'{t0}', (t0, d), textcoords="offset points", xytext=(0, 10),
                    fontsize=8, ha='center')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "p1_sensitivity_T0.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Figure] saved: {path}")


def plot_iters_sweep(results):
    """Line plot: iters_per_T (log scale) vs final distance."""
    iters = np.array([r[0] for r in results])
    dists = np.array([r[1] for r in results])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, dists, 'o-', color='#2c7bb6', linewidth=1.5, markersize=8)
    ax.set_xlabel('Iterations per Temperature Step')
    ax.set_ylabel('Final Path Length (mm)')
    ax.set_title('SA Sensitivity: Iterations/T vs Solution Quality (n=198)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    for it, d in zip(iters, dists):
        ax.annotate(f'{it}', (it, d), textcoords="offset points", xytext=(0, 10),
                    fontsize=8, ha='center')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "p1_sensitivity_iters.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Figure] saved: {path}")


def plot_tradeoff(alpha_res, T0_res, iters_res):
    """Combined tradeoff: CPU time vs Solution Quality, color-coded by parameter sweep."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Sweep 1: α — color by alpha value (normalized)
    alphas = np.array([r[0] for r in alpha_res])
    a_dists = np.array([r[1] for r in alpha_res])
    a_times = np.array([r[2] for r in alpha_res])
    norm_a = (alphas - alphas.min()) / (alphas.max() - alphas.min() + 1e-12)
    sc1 = ax.scatter(a_times, a_dists, c=norm_a, cmap='Blues', s=80, marker='o',
                     edgecolors='black', linewidth=0.5, zorder=5, label='α sweep')
    for i, (x, y, a) in enumerate(zip(a_times, a_dists, alphas)):
        ax.annotate(f'α={a}', (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)

    # Sweep 2: T0
    t0s = np.array([r[0] for r in T0_res])
    t0_dists = np.array([r[1] for r in T0_res])
    t0_times = np.array([r[2] for r in T0_res])
    sc2 = ax.scatter(t0_times, t0_dists, c='#d7191c', s=80, marker='s',
                     edgecolors='black', linewidth=0.5, zorder=5, label='T₀ sweep')
    for x, y, t0v in zip(t0_times, t0_dists, t0s):
        ax.annotate(f'T0={t0v}', (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)

    # Sweep 3: iters
    iters = np.array([r[0] for r in iters_res])
    i_dists = np.array([r[1] for r in iters_res])
    i_times = np.array([r[2] for r in iters_res])
    sc3 = ax.scatter(i_times, i_dists, c='#2c7bb6', s=80, marker='^',
                     edgecolors='black', linewidth=0.5, zorder=5, label='iters/T sweep')
    for x, y, iv in zip(i_times, i_dists, iters):
        ax.annotate(f'iters={iv}', (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_xlabel('CPU Time (s)')
    ax.set_ylabel('Final Path Length (mm)')
    ax.set_title('SA Parameter Sensitivity: Time-Quality Tradeoff (n=198)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc1, ax=ax, label='Cooling Rate α (normalized)')
    cbar.set_label('Cooling Rate α', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "p1_sensitivity_tradeoff.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Figure] saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    SCALE_N = 198
    print("=" * 70)
    print(f"  Problem 1: SA Parameter Sensitivity Analysis  (n={SCALE_N})")
    print("=" * 70)

    # Build problem once
    print(f"\n  Building problem for n={SCALE_N}...")
    coords, dist_matrix = build_problem(SCALE_N)
    print(f"  Done. {SCALE_N} drill points + origin. Distance matrix shape: {dist_matrix.shape}")

    # NN initial solution (shared)
    print(f"  Computing NN initial solution...")
    np.random.seed(42)
    nn_order = nearest_neighbor(dist_matrix, SCALE_N)
    nn_dist = total_path_length(nn_order, dist_matrix)
    print(f"  NN initial distance: {nn_dist:.2f} mm")

    # Run sweeps
    alpha_results = sweep_alpha(dist_matrix, nn_order, SCALE_N)
    T0_results = sweep_T0(dist_matrix, nn_order, SCALE_N)
    iters_results = sweep_iters(dist_matrix, nn_order, SCALE_N)

    # Visualizations
    print(f"\n{'=' * 70}")
    print(f"  Generating Figures...")
    print(f"{'=' * 70}")
    plot_alpha_sweep(alpha_results)
    plot_T0_sweep(T0_results)
    plot_iters_sweep(iters_results)
    plot_tradeoff(alpha_results, T0_results, iters_results)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Sensitivity Analysis Complete!")
    print(f"{'=' * 70}")
    print(f"  Output tables: {TABLES_DIR}")
    print(f"  Output figures: {FIGURES_DIR}")
    print(f"  Files created:")
    print(f"    p1_sensitivity_alpha.csv")
    print(f"    p1_sensitivity_T0.csv")
    print(f"    p1_sensitivity_iters.csv")
    print(f"    p1_sensitivity_alpha.png")
    print(f"    p1_sensitivity_T0.png")
    print(f"    p1_sensitivity_iters.png")
    print(f"    p1_sensitivity_tradeoff.png")
