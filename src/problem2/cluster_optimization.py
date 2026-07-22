"""
问题2 空间聚类优化模块
===================
Inspired by Yang et al. (2026)'s zone-optimization strategy:
Decompose large hole-diameter groups into spatial sub-clusters,
solve TSP within each cluster, then connect clusters via centroid TSP.

For groups with > THRESHOLD points (e.g. 100):
  1. Cluster points spatially (k-means with k = ceil(n/50))
  2. Intra-cluster TSP (NN + 2-opt) via geometric center
  3. Inter-cluster TSP on centroids
  4. Rewire connections through closest point pairs

Only applies to n=1173 scale (other scales too small for clustering benefit).
Outputs comparison CSV and cluster visualization.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Physical Constants (matching hierarchical_solver.py)
# ============================================================
DRILL_SPEED = 100.0          # mm/s
DRILL_TIME = {"A": 0.15, "B": 0.20, "C": 0.30}
CHANGE_TIME = 5.0            # s per tool change
N_GROUPS = 3

# ============================================================
# Path Configuration
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from utils.data_loader import load_drill_data, get_coords, get_types

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

GROUP_COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
GROUP_NAMES = {"A": "A型孔 (0.3mm)", "B": "B型孔 (0.5mm)", "C": "C型孔 (1.0mm)"}

# ============================================================
# TSP Functions (copied from hierarchical_solver.py — self-contained)
# ============================================================

def euclidean_dist_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix, shape (n, n)."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def total_path_length(order: list, dist_matrix: np.ndarray) -> float:
    """Total path length for given visiting order."""
    total = 0.0
    for i in range(len(order) - 1):
        total += dist_matrix[order[i], order[i + 1]]
    return total


def nearest_neighbor_tsp(dist_matrix: np.ndarray, start: int = 0) -> list:
    """Nearest-neighbor heuristic to construct initial TSP solution."""
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
    """2-opt local search to improve TSP route."""
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


def solve_closed_tsp(coords: np.ndarray) -> tuple:
    """
    Solve closed TSP on given coords (no origin inserted).
    Returns (total_distance, order) where order is [0, ..., 0].
    """
    n = coords.shape[0]
    if n <= 1:
        return 0.0, [0, 0]
    dist_mat = euclidean_dist_matrix(coords)
    order = nearest_neighbor_tsp(dist_mat, start=0)
    order = two_opt_local_search(order, dist_mat)
    total_dist = total_path_length(order, dist_mat)
    return total_dist, order


# ============================================================
# K-Means Clustering (manual implementation, no sklearn dependency)
# ============================================================

def kmeans_cluster(coords: np.ndarray, k: int,
                   max_iters: int = 100) -> tuple:
    """
    Manual Lloyd's k-means clustering.
    Returns (labels: array of int, centroids: (k, 2) array).
    """
    n = len(coords)
    k = min(k, n)  # cannot have more clusters than points

    # Initialize centroids using k-means++ seeding
    centroids = np.zeros((k, 2))
    # First centroid: random point
    centroids[0] = coords[np.random.randint(n)]
    for c in range(1, k):
        dists = np.linalg.norm(coords[:, None, :] - centroids[None, :c, :], axis=2)
        min_dists = np.min(dists, axis=1)
        # Weighted random selection (k-means++)
        probs = min_dists ** 2
        probs /= probs.sum()
        centroids[c] = coords[np.random.choice(n, p=probs)]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iters):
        # Assign points to nearest centroid
        dists_all = np.linalg.norm(
            coords[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists_all, axis=1)

        # Update centroids
        new_centroids = np.array([
            coords[labels == i].mean(axis=0) if np.any(labels == i)
            else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids, rtol=1e-6):
            break
        centroids = new_centroids

    return labels, centroids


# ============================================================
# Cluster-based TSP Solver
# ============================================================

def clustered_tsp_solve(group_coords: np.ndarray,
                        threshold: int = 100,
                        target_cluster_size: int = 50) -> dict:
    """
    Solve TSP for a diameter group using spatial clustering.

    Parameters:
        group_coords: (n, 2) coordinates of points in this group
        threshold: min points to trigger clustering
        target_cluster_size: approx points per cluster

    Returns:
        dict with keys: total_dist, cluster_count, intra_dists, inter_cost,
                        labels, centroids, cluster_visit_order, visit_orders
    """
    n = group_coords.shape[0]
    origin = np.array([0.0, 0.0])

    if n <= threshold:
        # No clustering: solve directly (O → all points → O)
        all_coords = np.vstack([origin.reshape(1, 2), group_coords])
        dist_mat = euclidean_dist_matrix(all_coords)
        order = nearest_neighbor_tsp(dist_mat, start=0)
        order = two_opt_local_search(order, dist_mat)
        total_dist = total_path_length(order, dist_mat)
        return {
            "total_dist": total_dist,
            "cluster_count": 1,
            "intra_dists": [total_dist],
            "inter_cost": 0.0,
            "labels": np.zeros(n, dtype=int),
            "centroids": np.array([group_coords.mean(axis=0)]),
            "cluster_visit_order": [0],
            "visit_orders": {0: [idx - 1 for idx in order[1:-1]]},
        }

    # ---- Step 1: Spatial clustering ----
    k = max(2, int(np.ceil(n / target_cluster_size)))
    labels, centroids = kmeans_cluster(group_coords, k)
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)

    print(f"    [Clustering] {n} points → {n_clusters} clusters "
          f"(k={k}, target ~{target_cluster_size}/cluster)")

    # ---- Step 2: Intra-cluster TSP ----
    # Solve centroid-based TSP for each cluster to determine internal
    # point visiting order (p1 → p2 → ... → pm). The actual path within
    # the cluster will be: entry → p1 → p2 → ... → pm → exit,
    # where entry/exit are dynamically chosen for inter-cluster connection.
    cluster_data = {}
    for label in unique_labels:
        mask = labels == label
        pts = group_coords[mask]
        n_pts = len(pts)
        if n_pts == 0:
            continue

        # Build coordinates: centroid + cluster points
        centroid = centroids[label]
        pts_with_center = np.vstack([centroid.reshape(1, 2), pts])

        # Solve TSP: centroid → all points → centroid
        dist_mat = euclidean_dist_matrix(pts_with_center)
        order = nearest_neighbor_tsp(dist_mat, start=0)
        order = two_opt_local_search(order, dist_mat)

        # Extract visit order (excluding centroid at indices 0 and -1)
        # order = [0, i1, i2, ..., im, 0]  where indices are into pts_with_center
        # visit_order = [i1-1, i2-1, ..., im-1]  (indices into pts)
        visit_order = [idx - 1 for idx in order[1:-1]]

        # Precompute internal edge distances: d(p1, p2) + ... + d(p_{m-1}, pm)
        # These are the TSP edges between consecutive points (excluding centroid edges)
        internal_edges = 0.0
        for i in range(len(visit_order) - 1):
            internal_edges += np.linalg.norm(
                pts[visit_order[i]] - pts[visit_order[i + 1]])

        cluster_data[label] = {
            "points": pts,
            "centroid": centroid,
            "visit_order": visit_order,     # indices into pts
            "internal_edges": internal_edges,  # sum of consecutive TSP edges
            "n_pts": n_pts,
        }

    # ---- Step 3: TSP on centroids (cluster visit order) ----
    centroid_array = np.array([cluster_data[l]["centroid"]
                                for l in unique_labels])
    centroid_dist_mat = euclidean_dist_matrix(centroid_array)
    centroid_order = nearest_neighbor_tsp(centroid_dist_mat, start=0)
    centroid_order = two_opt_local_search(centroid_order, centroid_dist_mat)

    # Map centroid TSP order to cluster labels
    cluster_visit_order = [unique_labels[i] for i in centroid_order[:-1]]

    # ---- Step 4: Connect clusters (rewire) ----
    total_dist = 0.0
    _next_entry_idx = 0  # initialized for linter; set by previous iteration

    for order_idx, label in enumerate(cluster_visit_order):
        cd = cluster_data[label]
        pts = cd["points"]
        visit_order = cd["visit_order"]
        internal_edges = cd["internal_edges"]
        n_pts = cd["n_pts"]

        # p1 and pm are the first and last points in the TSP visit order
        p1_idx = visit_order[0]
        pm_idx = visit_order[-1] if len(visit_order) > 0 else p1_idx

        # Determine entry point into this cluster
        if order_idx == 0:
            # First cluster: entry is nearest point to origin
            dists_to_origin = np.linalg.norm(pts - origin, axis=1)
            entry_idx = int(np.argmin(dists_to_origin))
            total_dist += dists_to_origin[entry_idx]
        else:
            # Entry was set by previous cluster's exit
            entry_idx = _next_entry_idx

        # Determine exit point from this cluster
        if order_idx < len(cluster_visit_order) - 1:
            # Exit: nearest point to next cluster
            next_label = cluster_visit_order[order_idx + 1]
            next_pts = cluster_data[next_label]["points"]

            # Compute all pairwise distances
            diff = pts[:, None, :] - next_pts[None, :, :]
            dists_between = np.sqrt(np.sum(diff ** 2, axis=2))
            flat_min_idx = int(np.argmin(dists_between))
            exit_idx_local, next_entry_idx_local = np.unravel_index(
                flat_min_idx, dists_between.shape)
            exit_idx = int(exit_idx_local)
            _next_entry_idx = int(next_entry_idx_local)
            inter_dist = float(dists_between[exit_idx, _next_entry_idx])
            total_dist += inter_dist
        else:
            # Last cluster: exit is nearest point to origin
            dists_to_origin = np.linalg.norm(pts - origin, axis=1)
            exit_idx = int(np.argmin(dists_to_origin))
            total_dist += dists_to_origin[exit_idx]

        # Intra-cluster path: entry → p1 → p2 → ... → pm → exit
        # = internal_edges (TSP consecutive edges, excl. centroid)
        #   + d(entry, p1) + d(pm, exit)
        intra_path = internal_edges
        intra_path += np.linalg.norm(pts[entry_idx] - pts[p1_idx])
        if len(visit_order) > 0:
            intra_path += np.linalg.norm(pts[pm_idx] - pts[exit_idx])
        total_dist += intra_path

    return {
        "total_dist": total_dist,
        "cluster_count": n_clusters,
        "labels": labels,
        "centroids": centroids,
        "cluster_visit_order": cluster_visit_order,
        "visit_orders": {l: cluster_data[l]["visit_order"] for l in unique_labels},
        "cluster_data": cluster_data,
    }


# ============================================================
# Direct TSP (no clustering) for comparison
# ============================================================

def direct_tsp_solve(group_coords: np.ndarray) -> float:
    """
    Solve TSP directly: O → all points → O (no clustering).
    Returns total distance in mm.
    """
    origin = np.array([0.0, 0.0])
    all_coords = np.vstack([origin.reshape(1, 2), group_coords])
    dist_mat = euclidean_dist_matrix(all_coords)
    order = nearest_neighbor_tsp(dist_mat, start=0)
    order = two_opt_local_search(order, dist_mat)
    return total_path_length(order, dist_mat)


# ============================================================
# Main Analysis
# ============================================================

def run_cluster_analysis():
    """Main function: run clustering optimization and compare with baseline."""
    print("=" * 70)
    print("  问题2 — 空间聚类优化分析 (Yang et al., 2026 启发)")
    print(f"  规模: n=1173 孔")
    print("=" * 70)

    # ---- Step 1: Load data ----
    data = load_drill_data(1173)
    coords_all = get_coords(data)
    types_all = get_types(data)

    groups = {}
    group_counts = {}
    for t in ["A", "B", "C"]:
        mask = (types_all == t)
        groups[t] = coords_all[mask]
        group_counts[t] = mask.sum()

    print(f"\n  [数据] A型={group_counts['A']}孔, "
          f"B型={group_counts['B']}孔, C型={group_counts['C']}孔")

    # Baseline distances from p2_results.csv (n=1173):
    # Movement_A=78.33s, Movement_B=70.17s, Movement_C=75.27s @ 100mm/s
    baseline_dist = {
        "A": 78.33 * DRILL_SPEED,   # 7833.0 mm
        "B": 70.17 * DRILL_SPEED,   # 7017.0 mm
        "C": 75.27 * DRILL_SPEED,   # 7527.0 mm
    }

    print(f"\n  [基线] 原始TSP距离:")
    for t in ["A", "B", "C"]:
        print(f"    {t}型 ({group_counts[t]}孔): {baseline_dist[t]:.2f} mm "
              f"= {baseline_dist[t] / DRILL_SPEED:.2f} s")

    # ---- Step 2: Direct TSP verification (re-run to confirm) ----
    print(f"\n  [验证] 重新计算直接TSP (用于公平对比) ...")
    direct_dist = {}
    direct_times = {}
    for t in ["A", "B", "C"]:
        t_start = time.perf_counter()
        dist = direct_tsp_solve(groups[t])
        elapsed = time.perf_counter() - t_start
        direct_dist[t] = dist
        direct_times[t] = elapsed
        print(f"    {t}型: {dist:.2f} mm = {dist/DRILL_SPEED:.2f}s  "
              f"(计算 {elapsed:.3f}s)")

    # ---- Step 3: Clustered TSP ----
    print(f"\n  [聚类TSP] 应用空间聚类优化 ...")
    clustered_dist = {}
    clustered_times = {}
    cluster_details = {}
    cluster_counts = {}

    for t in ["A", "B", "C"]:
        print(f"\n  --- {GROUP_NAMES[t]} ({group_counts[t]}孔) ---")
        t_start = time.perf_counter()
        result = clustered_tsp_solve(
            groups[t], threshold=100, target_cluster_size=50)
        elapsed = time.perf_counter() - t_start

        clustered_dist[t] = result["total_dist"]
        clustered_times[t] = elapsed
        cluster_counts[t] = result["cluster_count"]
        cluster_details[t] = result

        improv = (direct_dist[t] - result["total_dist"]) / direct_dist[t] * 100
        print(f"    聚类后距离: {result['total_dist']:.2f} mm "
              f"= {result['total_dist']/DRILL_SPEED:.2f}s  "
              f"(计算 {elapsed:.3f}s)")
        print(f"    分{result['cluster_count']}个簇, "
              f"vs直接: 改善 {improv:+.2f}%")

    # ---- Step 4: Summary comparison ----
    print(f"\n{'=' * 70}")
    print(f"  结果对比")
    print(f"{'=' * 70}")
    print(f"{'组别':>6s}  {'原始距离mm':>12s}  {'聚类距离mm':>12s}  "
          f"{'改善%':>10s}  {'原始用时s':>10s}  {'聚类用时s':>10s}  {'簇数':>5s}")
    print(f"{'-' * 70}")

    results = []
    for t in ["A", "B", "C"]:
        improv = (direct_dist[t] - clustered_dist[t]) / direct_dist[t] * 100
        print(f"{t:>6s}  {direct_dist[t]:>12.2f}  {clustered_dist[t]:>12.2f}  "
              f"{improv:>+9.2f}%  {direct_times[t]:>10.3f}  "
              f"{clustered_times[t]:>10.3f}  {cluster_counts[t]:>5d}")
        results.append({
            "Group": t,
            "Direct_Dist_mm": direct_dist[t],
            "Clustered_Dist_mm": clustered_dist[t],
            "Improvement_pct": improv,
            "Direct_Time_s": direct_times[t],
            "Clustered_Time_s": clustered_times[t],
            "Cluster_Count": cluster_counts[t],
        })

    # Total comparison
    total_direct = sum(direct_dist.values())
    total_clustered = sum(clustered_dist.values())
    total_improv = (total_direct - total_clustered) / total_direct * 100
    print(f"{'-' * 70}")
    print(f"{'合计':>6s}  {total_direct:>12.2f}  {total_clustered:>12.2f}  "
          f"{total_improv:>+9.2f}%  {sum(direct_times.values()):>10.3f}  "
          f"{sum(clustered_times.values()):>10.3f}")
    print(f"{'=' * 70}")

    # ---- Step 5: Save comparison CSV ----
    os.makedirs(TABLE_DIR, exist_ok=True)
    csv_path = os.path.join(TABLE_DIR, "p2_cluster_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Group,Original_Dist_mm,Clustered_Dist_mm,Improvement_pct,"
                "Original_Time_s,Clustered_Time_s,Cluster_Count\n")
        for r in results:
            f.write(f"{r['Group']},{r['Direct_Dist_mm']:.2f},"
                    f"{r['Clustered_Dist_mm']:.2f},"
                    f"{r['Improvement_pct']:.2f},"
                    f"{r['Direct_Time_s']:.3f},{r['Clustered_Time_s']:.3f},"
                    f"{r['Cluster_Count']}\n")
        f.write(f"TOTAL,{total_direct:.2f},{total_clustered:.2f},"
                f"{total_improv:.2f},{sum(direct_times.values()):.3f},"
                f"{sum(clustered_times.values()):.3f},\n")
    print(f"\n[输出] 对比表已保存: {csv_path}")

    # ---- Step 6: Compute total completion time with clustering ----
    # Using clustered distances instead of original group distances
    move_A = clustered_dist["A"] / DRILL_SPEED
    move_B = clustered_dist["B"] / DRILL_SPEED
    move_C = clustered_dist["C"] / DRILL_SPEED
    drill_time = sum(group_counts[t] * DRILL_TIME[t] for t in ["A", "B", "C"])
    change_time = N_GROUPS * CHANGE_TIME
    total_clustered_time = drill_time + change_time + move_A + move_B + move_C

    move_A_orig = baseline_dist["A"] / DRILL_SPEED
    move_B_orig = baseline_dist["B"] / DRILL_SPEED
    move_C_orig = baseline_dist["C"] / DRILL_SPEED
    total_original_time = drill_time + change_time + move_A_orig + move_B_orig + move_C_orig

    print(f"\n  [总完成时间]  （钻孔={drill_time:.2f}s 换刀={change_time:.0f}s）")
    print(f"    原始移动: A={move_A_orig:.2f}s B={move_B_orig:.2f}s C={move_C_orig:.2f}s")
    print(f"    聚类移动: A={move_A:.2f}s B={move_B:.2f}s C={move_C:.2f}s")
    print(f"    原始总时间: {total_original_time:.2f}s")
    print(f"    聚类总时间: {total_clustered_time:.2f}s")
    print(f"    总节省: {total_original_time - total_clustered_time:.2f}s "
          f"({(total_original_time - total_clustered_time) / total_original_time * 100:.2f}%)")

    # ---- Step 7: Visualization ----
    # Generate cluster scatter plots for each group
    for t in ["A", "B", "C"]:
        if cluster_counts.get(t, 1) > 1:
            _plot_clustered_group(groups[t], cluster_details[t],
                                  t, group_counts[t])

    return results


def _plot_clustered_group(coords: np.ndarray, detail: dict,
                          group_label: str, n_total: int):
    """Generate scatter plot: points color-coded by cluster."""
    labels = detail["labels"]
    centroids = detail["centroids"]

    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)
    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "tab20b", n_clusters)

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        color = cmap(i)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], s=8, alpha=0.7, edgecolors='none',
                   label=f"簇 {label + 1} ({mask.sum()}点)")
        # Mark centroid
        ax.scatter(centroids[label, 0], centroids[label, 1],
                   c=[color], s=80, marker='X', edgecolors='black',
                   linewidths=1, zorder=5)

    # Mark origin
    ax.scatter(0, 0, c='red', s=120, marker='*', zorder=6, label='原点 O')

    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_title(f"空间聚类结果 — {GROUP_NAMES[group_label]} "
                 f"(n={n_total}, {n_clusters}簇)",
                 fontsize=14)
    ax.legend(loc="upper left", fontsize=8, ncol=2,
              bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    filepath = os.path.join(FIGURE_DIR, f"p2_cluster_{group_label}_group.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [可视化] 已保存: p2_cluster_{group_label}_group.png")


# ============================================================
# Entry
# ============================================================

def main():
    """Main entry point."""
    run_cluster_analysis()
    print(f"\n{'=' * 70}")
    print(f"  聚类优化分析完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
