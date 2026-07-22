"""
MVODM (Minimizing Variance of Distance Matrix) preprocessing for TSP.

Algorithm (Rao et al. 2015, 中国科学: 信息科学):
  1. Compute the original distance matrix D (n+1 × n+1, including origin).
  2. For each node i, compute its average distance to all other nodes: avg_dist[i].
  3. Transform: D_new[i][j] = D[i][j] - α * (avg_dist[i] + avg_dist[j]).
     NOTE: We do NOT clamp to max(0, ...) — negative transformed distances are
     allowed and indicate "preferred" edges (per the Rao paper).
  4. Run NN / 2-opt on the TRANSFORMED distance matrix to obtain the order.
  5. Evaluate the order against the ORIGINAL distance matrix.

Idea: reducing distance variance makes greedy choices (NN) less myopic.
"""
import os
import sys
import io
import time
import numpy as np
import csv

# ── Path setup (mirrors solve_all.py) ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import load_drill_data, get_coords
from src.utils.distance import euclidean_distance_matrix, total_path_length

np.random.seed(42)

SPEED = 100.0  # mm/s
# Original α values from the specification, plus smaller calibrated values
ALPHAS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]


# ═══════════════════════════════════════════════════════════════════
# Algorithm A: Nearest Neighbor (copied from solve_all.py)
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
# Algorithm B: 2-opt local search (copied from solve_all.py)
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
# MVODM transformation
# ═══════════════════════════════════════════════════════════════════
def mvodm_transform(dist_matrix: np.ndarray, alpha: float, clamp: bool = False) -> np.ndarray:
    """
    Transform the distance matrix using MVODM.

    D_new[i][j] = D[i][j] - α * (avg_dist[i] + avg_dist[j])

    If clamp=True: D_new[i][j] = max(0, D_new[i][j])
    If clamp=False: negative values are allowed (per Rao et al. 2015).

    avg_dist[i] = mean distance from i to all OTHER nodes (excluding self).
    """
    n_plus_1 = dist_matrix.shape[0]  # n + 1 (including origin)
    avg_dist = np.sum(dist_matrix, axis=1) / (n_plus_1 - 1)
    D_new = dist_matrix - alpha * (avg_dist[:, np.newaxis] + avg_dist[np.newaxis, :])
    if clamp:
        D_new = np.maximum(D_new, 0.0)
    np.fill_diagonal(D_new, 0.0)
    return D_new


# ═══════════════════════════════════════════════════════════════════
# Experiment runner for one n and one alpha
# ═══════════════════════════════════════════════════════════════════
def run_experiment(n: int, alpha: float, dist_orig: np.ndarray,
                   clamp: bool = False, verbose: bool = True):
    """Run MVODM → NN → 2-opt pipeline, evaluate against original matrix."""
    D_trans = mvodm_transform(dist_orig, alpha, clamp=clamp)
    nn_order_trans = nearest_neighbor(D_trans, n)
    nn_dist = total_path_length(nn_order_trans, dist_orig)
    opt_order_trans = two_opt(nn_order_trans, D_trans, n)
    opt_dist = total_path_length(opt_order_trans, dist_orig)
    if verbose:
        print(f"    NN  = {nn_dist:>12.2f} mm  ({nn_dist / SPEED:.2f} s)")
        print(f"    2-opt = {opt_dist:>10.2f} mm  ({opt_dist / SPEED:.2f} s)")
    return nn_dist, opt_dist


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 72)
    print("  MVODM Check — Variance-Reducing Distance Matrix for TSP")
    print("  Based on Rao et al. (2015), 中国科学: 信息科学")
    print("=" * 72)

    # ── Phase 1: n = 50 ─────────────────────────────────────────────
    n = 50
    print(f"\n{'─' * 60}")
    print(f"  Phase 1: n = {n}")
    print(f"{'─' * 60}")

    data = load_drill_data(n)
    drill_coords = get_coords(data)
    origin = np.array([[0.0, 0.0]])
    coords = np.vstack([origin, drill_coords])
    dist_orig = euclidean_distance_matrix(coords)

    # --- Baseline: NN + 2-opt on original matrix ---
    print(f"  [Baseline] Original distance matrix:")
    t0 = time.perf_counter()
    nn_order_base = nearest_neighbor(dist_orig, n)
    nn_base = total_path_length(nn_order_base, dist_orig)
    opt_order_base = two_opt(nn_order_base, dist_orig, n)
    opt_base = total_path_length(opt_order_base, dist_orig)
    t1 = time.perf_counter()
    orig_var = np.var(dist_orig[dist_orig > 0])
    print(f"    NN  = {nn_base:>12.2f} mm  ({nn_base / SPEED:.2f} s)")
    print(f"    2-opt = {opt_base:>10.2f} mm  ({opt_base / SPEED:.2f} s)")
    print(f"    Distance variance = {orig_var:.1f}")
    print(f"    Time: {t1 - t0:.3f}s")

    # --- Test both clamped and unclamped for the specified α values ---
    print(f"\n  ---- Unclamped MVODM (negative distances allowed) ----")

    results = {}  # key: (alpha, clamped) -> (nn_dist, opt_dist, nn_imp, opt_imp)
    best_nn_result = (ALPHAS[0], False, -float('inf'), 0.0)
    best_opt_result = (ALPHAS[0], False, 0.0, -float('inf'))

    for alpha in ALPHAS:
        label = f"MVODM(α={alpha:.2f}, unclamped)"
        print(f"\n  [{label}] Transformed distance matrix:")
        D_trans = mvodm_transform(dist_orig, alpha, clamp=False)
        var_trans = np.var(D_trans[D_trans != 0]) if np.any(D_trans != 0) else 0
        neg_frac = np.sum(D_trans < 0) / (D_trans.size - D_trans.shape[0]) * 100
        print(f"    Variance: {orig_var:.0f} → {var_trans:.0f}  ({var_trans/orig_var*100:.1f}% of original)")
        print(f"    Negative entries: {neg_frac:.1f}%")

        t0 = time.perf_counter()
        nn_dist, opt_dist = run_experiment(n, alpha, dist_orig, clamp=False)
        t1 = time.perf_counter()

        nn_impr = (nn_base - nn_dist) / nn_base * 100.0
        opt_impr = (opt_base - opt_dist) / opt_base * 100.0
        key = (alpha, False)
        results[key] = (nn_dist, opt_dist, nn_impr, opt_impr)
        print(f"    NN  improvement: {nn_impr:+.2f}%")
        print(f"    2-opt improvement: {opt_impr:+.2f}%")
        print(f"    Time: {t1 - t0:.3f}s")

        if nn_impr > best_nn_result[2]:
            best_nn_result = (alpha, False, nn_impr, nn_dist)
        if opt_impr > best_opt_result[3]:
            best_opt_result = (alpha, False, opt_dist, opt_impr)

    # ── Print comparison table ──────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  Comparison Table (n = {n}, unclamped MVODM)")
    print(f"{'=' * 72}")
    print(f"  {'Method':<22s}  {'NN (mm)':>12s}  {'2-opt (mm)':>12s}  {'NN imp%':>8s}  {'2-opt imp%':>8s}")
    print(f"  {'─'*22}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}")
    print(f"  {'Original':<22s}  {nn_base:>12.2f}  {opt_base:>12.2f}  {'—':>8s}  {'—':>8s}")
    for alpha in ALPHAS:
        key = (alpha, False)
        if key in results:
            nn_d, opt_d, nn_i, opt_i = results[key]
            marker = " *" if nn_i > 0 or opt_i > 0 else ""
            print(f"  {f'MVODM(α={alpha:.2f})':<22s}  {nn_d:>12.2f}  {opt_d:>12.2f}  {nn_i:>+7.2f}%  {opt_i:>+7.2f}%{marker}")

    print(f"\n  NOTE: * = improvement over original")

    # ── Phase 2: n = 1173 (only if improvement found) ───────────────
    any_improvement = any(impr_nn > 0 for _, _, impr_nn, _ in results.values()) or \
                      any(impr_opt > 0 for _, _, _, impr_opt in results.values())

    n1173_result = None
    if any_improvement:
        best_alpha_val, best_clamped_val, best_nn_impr_val, _ = best_nn_result
        best_oa_val, best_oc_val, _, best_opt_impr_val = best_opt_result
        # Use the alpha that helps NN most (MVODM targets greedy heuristics)
        use_alpha = best_alpha_val if best_nn_impr_val >= best_opt_impr_val else best_oa_val
        use_clamped = best_clamped_val if best_nn_impr_val >= best_opt_impr_val else best_oc_val

        print(f"\n{'─' * 60}")
        print(f"  Phase 2: n = 1173 (best α = {use_alpha}, {'clamped' if use_clamped else 'unclamped'})")
        print(f"{'─' * 60}")

        n_big = 1173
        data_big = load_drill_data(n_big)
        coords_big = np.vstack([[[0.0, 0.0]], get_coords(data_big)])
        dist_big = euclidean_distance_matrix(coords_big)

        # Baseline
        print(f"  [Baseline] Original distance matrix (n={n_big}):")
        t0 = time.perf_counter()
        nn_big_base_order = nearest_neighbor(dist_big, n_big)
        nn_big_base = total_path_length(nn_big_base_order, dist_big)
        opt_big_base = total_path_length(two_opt(nn_big_base_order, dist_big, n_big), dist_big)
        t1 = time.perf_counter()
        print(f"    NN  = {nn_big_base:>12.2f} mm  ({nn_big_base / SPEED:.2f} s)")
        print(f"    2-opt = {opt_big_base:>10.2f} mm  ({opt_big_base / SPEED:.2f} s)")

        # MVODM
        print(f"  [MVODM(α={use_alpha}, {'clamped' if use_clamped else 'unclamped'})]:")
        t0 = time.perf_counter()
        D_big_trans = mvodm_transform(dist_big, use_alpha, clamp=use_clamped)
        nn_big_order = nearest_neighbor(D_big_trans, n_big)
        nn_big_dist = total_path_length(nn_big_order, dist_big)
        opt_big_order = two_opt(nn_big_order, D_big_trans, n_big)
        opt_big_dist = total_path_length(opt_big_order, dist_big)
        t1 = time.perf_counter()
        nn_big_impr = (nn_big_base - nn_big_dist) / nn_big_base * 100.0
        opt_big_impr = (opt_big_base - opt_big_dist) / opt_big_base * 100.0
        print(f"    NN  = {nn_big_dist:>12.2f} mm  ({nn_big_dist / SPEED:.2f} s)  improvement: {nn_big_impr:+.2f}%")
        print(f"    2-opt = {opt_big_dist:>10.2f} mm  ({opt_big_dist / SPEED:.2f} s)  improvement: {opt_big_impr:+.2f}%")
        print(f"    Time: {t1 - t0:.3f}s")
        n1173_result = (n_big, use_alpha, nn_big_base, nn_big_dist, opt_big_base, opt_big_dist,
                        nn_big_impr, opt_big_impr, use_clamped)
    else:
        print(f"\n  No improvement found for n={n}. Skipping n=1173.")

    # ── Save CSV ────────────────────────────────────────────────────
    tables_dir = os.path.join(PROJECT_ROOT, "output", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, "p1_mvodm_results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "alpha", "NN_Distance_mm", "2opt_Distance_mm",
                          "NN_Improvement_pct", "2opt_Improvement_pct"])
        writer.writerow(["Original", "", f"{nn_base:.4f}", f"{opt_base:.4f}", "0.00", "0.00"])
        for alpha in ALPHAS:
            key = (alpha, False)
            if key in results:
                nn_d, opt_d, nn_i, opt_i = results[key]
                writer.writerow([f"MVODM(n=50)", f"{alpha:.2f}", f"{nn_d:.4f}", f"{opt_d:.4f}",
                                  f"{nn_i:.2f}", f"{opt_i:.2f}"])
        if n1173_result:
            _, ba, nnb, nnd, optb, optd, nni, oi, _ = n1173_result
            writer.writerow([f"MVODM(n=1173)", f"{ba:.2f}", f"{nnd:.4f}", f"{optd:.4f}",
                              f"{nni:.2f}", f"{oi:.2f}"])

    print(f"\n  [CSV] saved: {csv_path}")

    # ── Summary ─────────────────────────────────────────────────────
    best_nn_a, _, best_nn_i, _ = best_nn_result
    best_opt_a, _, _, best_opt_i = best_opt_result
    print(f"\n{'=' * 72}")
    print(f"  Summary")
    print(f"{'=' * 72}")
    print(f"  Best NN  α: {best_nn_a}  ({best_nn_i:+.2f}% vs original)")
    print(f"  Best 2-opt α: {best_opt_a}  ({best_opt_i:+.2f}% vs original)")
    print(f"{'=' * 72}")
    print(f"  Done!")
