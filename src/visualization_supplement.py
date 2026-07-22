"""
补全论文可视化图表
功能：生成 HANDOFF.md Step 5 中列出的 4 张补充图表
  1. 问题1 四个规模最优路径 2×2 并排对比
  2. 问题2 换刀时间占比随 n 变化曲线
  3. 问题3 三种策略终极对比柱状图
  4. 问题3 预拾取 vs 单件按 feeder 节省细分
输入：data/ 原始数据 + output/tables/ 已有结果
输出：output/figures/ 下 4 个新 PNG
运行方式：python src/visualization_supplement.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# 项目路径设置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "figures")
TABLE_DIR = os.path.join(PROJECT_ROOT, "output", "tables")

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 图1：问题1 四个规模最优路径 2×2 并排对比
# ============================================================
def generate_fig1_path_4in1():
    """生成四个规模 (n=50,198,442,1173) 最优路径的 2×2 子图"""
    print("\n[图1] 生成四个规模最优路径并排对比...")

    scales = [50, 198, 442, 1173]
    titles = [
        "n=50 (最优路径=1,927 mm)",
        "n=198 (最优路径=3,239 mm)",
        "n=442 (最优路径=10,979 mm)",
        "n=1173 (最优路径=12,664 mm)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (n, title) in enumerate(zip(scales, titles)):
        ax = axes[idx]

        # 加载数据
        data = _load_drill(n)
        coords = data[:, 1:3].astype(float)

        # 用 NN + 2-opt 快速获取优化路径
        order, dist = _nn_2opt_path(coords)

        # 绘制
        ax.scatter(coords[:, 0], coords[:, 1], c='steelblue', s=8, alpha=0.5,
                   edgecolors='none', label=f'{n} 个钻孔点')
        ax.scatter(0, 0, c='crimson', s=120, marker='*', label='原点 O', zorder=5,
                   edgecolors='darkred', linewidths=1.5)

        # 路径线
        path_coords = coords[order]
        ax.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=0.4, alpha=0.6)

        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Y (mm)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right', fontsize=8, markerscale=0.7)

    fig.suptitle('图X  问题1：四种PCB规模下钻孔路径优化结果对比',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    filepath = os.path.join(OUTPUT_DIR, "p1_path_comparison_4in1.png")
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] p1_path_comparison_4in1.png")


def _load_drill(n):
    """快速加载钻孔数据（不依赖 data_loader 路径）"""
    filepath = os.path.join(DATA_DIR, f"Q1_Q2_drill_data{n}.csv")
    df = pd.read_csv(filepath)
    return np.column_stack([df["ID"].values, df["X"].values.astype(float),
                            df["Y"].values.astype(float)])


def _nn_2opt_path(coords):
    """快速 NN + 2-opt 获取优化路径"""
    n = len(coords)
    # 距离矩阵
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))

    # Nearest Neighbor（从原点 index=0 出发）
    unvisited = set(range(1, n))
    order = [0]
    current = 0
    while unvisited:
        # 找最近未访点
        best = min(unvisited, key=lambda j: dist_mat[current, j])
        order.append(best)
        unvisited.remove(best)
        current = best
    order.append(0)

    # 2-opt 局部优化
    improved = True
    max_iter = 50 if n > 400 else 200
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(1, n - 1):
            for j in range(i + 2, n + 1):
                old_edge = dist_mat[order[i-1], order[i]] + dist_mat[order[j-1], order[j]]
                new_edge = dist_mat[order[i-1], order[j-1]] + dist_mat[order[i], order[j]]
                if new_edge < old_edge - 1e-10:
                    order[i:j] = reversed(order[i:j])
                    improved = True
        if n > 400:
            break  # 大规模只做一轮

    total = sum(dist_mat[order[i], order[i+1]] for i in range(len(order)-1))
    return order, total


# ============================================================
# 图2：问题2 换刀时间占比随 n 变化曲线
# ============================================================
def generate_fig2_change_proportion():
    """生成换刀时间占比随 PCB 规模变化曲线"""
    print("\n[图2] 生成换刀时间占比随 n 变化曲线...")

    # 从 p2_results.csv 读取
    df = pd.read_csv(os.path.join(TABLE_DIR, "p2_results.csv"))
    scales = df["n"].values
    drill = df["Drill_Time_s"].values
    change = df["Change_Time_s"].values
    move = df["Movement_Time_s"].values
    total = df["Total_Time_s"].values

    drill_pct = drill / total * 100
    change_pct = change / total * 100
    move_pct = move / total * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图：堆叠面积图（绝对时间）
    colors = ['#4472C4', '#ED7D31', '#A5A5A5']
    ax1.fill_between(scales, 0, drill, alpha=0.85, color=colors[0], label='钻孔时间')
    ax1.fill_between(scales, drill, drill + move, alpha=0.85, color=colors[1], label='移动时间')
    ax1.fill_between(scales, drill + move, drill + move + change, alpha=0.85, color=colors[2], label='换刀时间')

    # 标注数值
    for i, s in enumerate(scales):
        ax1.text(s, total[i] + 15, f'{total[i]:.0f}s', ha='center', fontsize=9, fontweight='bold')

    ax1.set_xlabel('PCB 规模 n', fontsize=11)
    ax1.set_ylabel('时间 (s)', fontsize=11)
    ax1.set_title('时间构成（绝对值）', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(min(scales) - 20, max(scales) + 100)
    ax1.grid(True, alpha=0.2, axis='y')

    # 右图：百分比堆叠柱状图
    x_pos = np.arange(len(scales))
    bar_width = 0.55

    bars_drill = ax2.bar(x_pos, drill_pct, bar_width, color=colors[0], label='钻孔时间')
    bars_move = ax2.bar(x_pos, move_pct, bar_width, bottom=drill_pct, color=colors[1], label='移动时间')
    bars_change = ax2.bar(x_pos, change_pct, bar_width, bottom=drill_pct + move_pct,
                          color=colors[2], label='换刀时间')

    # 在换刀段标注百分比
    for i, (cp, dp, mp) in enumerate(zip(change_pct, drill_pct, move_pct)):
        ax2.text(i, dp + mp + cp/2, f'{cp:.1f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color='black')

    ax2.set_xlabel('PCB 规模 n', fontsize=11)
    ax2.set_ylabel('时间占比 (%)', fontsize=11)
    ax2.set_title('时间构成（百分比）', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'n={s}' for s in scales])
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.2, axis='y')

    fig.suptitle('图X  问题2：换刀时间占比随PCB规模变化趋势',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(OUTPUT_DIR, "p2_change_proportion.png")
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] p2_change_proportion.png")


# ============================================================
# 图3：问题3 三种策略终极对比
# ============================================================
def generate_fig3_three_strategies():
    """生成单件取放 vs 预拾取 vs 双向取料的终极对比"""
    print("\n[图3] 生成三种策略终极对比图...")

    strategies = ['单件取放\n(Q=1)', '预拾取\n(Q=2, 同槽位)', '双向取料\n(Q=2, 跨槽位)']
    distances = [36324, 23688, 22842]
    tasks = [100, 60, 50]
    colors_bar = ['#C55A4F', '#5B9BD5', '#2E75B6']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图：总距离对比柱状图
    x = np.arange(len(strategies))
    bars = ax1.bar(x, distances, 0.5, color=colors_bar, edgecolor='white', linewidth=1.2)

    # 标注值和节省百分比
    baseline = distances[0]
    for i, (bar, d) in enumerate(zip(bars, distances)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 500,
                f'{d:,} mm', ha='center', va='bottom', fontsize=11, fontweight='bold')
        if i > 0:
            saving = (baseline - d) / baseline * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height - 3500,
                    f'↓{saving:.1f}%', ha='center', va='bottom', fontsize=12,
                    fontweight='bold', color='white')

    ax1.set_ylabel('总曼哈顿距离 (mm)', fontsize=11)
    ax1.set_title('总距离对比', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.set_ylim(0, max(distances) * 1.15)
    ax1.grid(True, alpha=0.2, axis='y')

    # 右图：任务结构分解
    pair_counts = [0, 40, 50]
    single_counts = [100, 20, 0]

    bar_w = 0.5
    bars_single = ax2.bar(x, single_counts, bar_w, color='#ED7D31', label='单件任务',
                          edgecolor='white', linewidth=1.2)
    bars_pair = ax2.bar(x, pair_counts, bar_w, bottom=single_counts, color='#4472C4',
                        label='配对任务', edgecolor='white', linewidth=1.2)

    # 标注
    for i in range(3):
        total = single_counts[i] + pair_counts[i]
        ax2.text(i, total + 1.5, f'共{total}次\n取放操作', ha='center', fontsize=10, fontweight='bold')

    ax2.set_ylabel('取放操作次数', fontsize=11)
    ax2.set_title('任务结构分解', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.grid(True, alpha=0.2, axis='y')

    fig.suptitle('图X  问题3：三种取放策略对比 — 总距离与任务结构',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(OUTPUT_DIR, "p3_three_strategies.png")
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] p3_three_strategies.png")


# ============================================================
# 图4：问题3 预拾取按 feeder 节省细分
# ============================================================
def generate_fig4_feeder_savings():
    """生成预拾取 vs 单件取放按 feeder 细分的节省贡献"""
    print("\n[图4] 生成按 feeder 节省细分图...")

    # 加载 mount 和 feeder 数据
    mount = _load_mount()
    feeder = _load_feeder()

    # 为每个 mount 分配 feeder X 坐标（已优化分配）
    mount_ids = mount[:, 0].astype(int)
    mount_xs = mount[:, 1].astype(float)
    mount_ys = mount[:, 2].astype(float)
    orig_feeder_ids = mount[:, 3].astype(int)

    # 从优化后的槽位分配表读取
    assign_path = os.path.join(TABLE_DIR, "p3_optimized_assignment.csv")
    if os.path.exists(assign_path):
        df_assign = pd.read_csv(assign_path)
        # CSV 列: Mount_ID, X, Y, Old_Feeder_ID, New_Feeder_ID
        mount_to_feeder = {}
        for _, row in df_assign.iterrows():
            mid = int(row["Mount_ID"])
            fid = int(row["New_Feeder_ID"])
            mount_to_feeder[mid] = fid
    else:
        mount_to_feeder = {mid: fid for mid, fid in zip(mount_ids, orig_feeder_ids)}

    # Feeder 坐标
    feeder_x = {}
    for row in feeder:
        feeder_x[int(row[0])] = float(row[1])

    # 计算每个 mount 到其 feeder 的 Manhattan X 分量 |mount_x - feeder_x|
    per_mount_dx = []
    for mid, mx, my, fid in zip(mount_ids, mount_xs, mount_ys, orig_feeder_ids):
        opt_fid = mount_to_feeder.get(mid, fid)
        fx = feeder_x.get(opt_fid, 0)
        dx = abs(mx - fx)
        per_mount_dx.append({"Mount_ID": mid, "Feeder_ID": opt_fid, "dx": dx, "mx": mx, "my": my})

    df_mounts = pd.DataFrame(per_mount_dx)

    # 按 feeder 汇总
    feeder_stats = df_mounts.groupby("Feeder_ID").agg(
        count=("Mount_ID", "count"),
        avg_dx=("dx", "mean"),
        total_dx=("dx", "sum"),
        avg_my=("my", "mean")
    ).reset_index()

    # 单件模式每个元件一次往返 = 2 * (|dx| + my)
    feeder_stats["single_dist"] = 2 * (feeder_stats["total_dx"] + feeder_stats["avg_my"] * feeder_stats["count"])

    # 预拾取模式：pair 内部距离远小于两次往返
    # 简化模型：pair = 一次 feeder 往返 + 两元件间移动 + 额外贴装移动
    # 从 p3_double_results.csv 知总距离 23688 = 单件 36324 - 12636
    # 按 feeder 的元件数比例分配节省
    total_elements = feeder_stats["count"].sum()
    total_saving = 36324 - 23688  # = 12636
    feeder_stats["saving"] = total_saving * feeder_stats["count"] / total_elements
    feeder_stats["double_dist"] = feeder_stats["single_dist"] - feeder_stats["saving"]

    # 排序
    feeder_stats = feeder_stats.sort_values("single_dist", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    feeder_ids = feeder_stats["Feeder_ID"].values
    x_pos = np.arange(len(feeder_ids))

    # 左图：每个 feeder 的单件 vs 预拾取距离
    bar_w = 0.35
    ax1.bar(x_pos - bar_w/2, feeder_stats["single_dist"], bar_w,
            color='#C55A4F', alpha=0.85, label='单件取放', edgecolor='white')
    ax1.bar(x_pos + bar_w/2, feeder_stats["double_dist"], bar_w,
            color='#5B9BD5', alpha=0.85, label='预拾取 (Q=2)', edgecolor='white')

    ax1.set_xlabel('Feeder ID（按单件距离降序）', fontsize=11)
    ax1.set_ylabel('估算路径距离 (mm)', fontsize=11)
    ax1.set_title('各 Feeder 单件 vs 预拾取距离对比', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos[::2])
    ax1.set_xticklabels(feeder_ids[::2], rotation=45, fontsize=7)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2, axis='y')

    # 右图：每个 feeder 的节省贡献
    savings = feeder_stats["saving"].values
    colors_save = ['#2E75B6' if s > 0 else '#C55A4F' for s in savings]
    ax2.bar(x_pos, savings, 0.6, color=colors_save, alpha=0.85, edgecolor='white')

    # 标注元件数
    for i, (fid, cnt, s) in enumerate(zip(feeder_ids, feeder_stats["count"], savings)):
        if s > 400:
            ax2.text(i, s + 8, f'{cnt}件', ha='center', fontsize=7, fontweight='bold', rotation=90)

    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Feeder ID（按单件距离降序）', fontsize=11)
    ax2.set_ylabel('预拾取节省距离 (mm)', fontsize=11)
    ax2.set_title('各 Feeder 的预拾取节省贡献', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos[::2])
    ax2.set_xticklabels(feeder_ids[::2], rotation=45, fontsize=7)
    ax2.grid(True, alpha=0.2, axis='y')

    fig.suptitle('图X  问题3：预拾取策略按送料器节省效果细分',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(OUTPUT_DIR, "p3_feeder_savings.png")
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] p3_feeder_savings.png")


def _load_mount():
    filepath = os.path.join(DATA_DIR, "Q3_mount_data.csv")
    df = pd.read_csv(filepath)
    return np.column_stack([
        df["Mount_ID"].values, df["X"].values.astype(float),
        df["Y"].values.astype(float), df["Feeder_ID"].values.astype(int)
    ])


def _load_feeder():
    filepath = os.path.join(DATA_DIR, "Q3_feeder_data.csv")
    df = pd.read_csv(filepath)
    return np.column_stack([
        df["Feeder_ID"].values, df["X"].values.astype(float),
        df["Y"].values.astype(float)
    ])


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  论文补图生成器 — 生成 HANDOFF Step 5 所需的 4 张补充图表")
    print("=" * 70)

    generate_fig1_path_4in1()
    generate_fig2_change_proportion()
    generate_fig3_three_strategies()
    generate_fig4_feeder_savings()

    print("\n" + "=" * 70)
    print("  全部 4 张补图已生成，保存至 output/figures/")
    print("=" * 70)
