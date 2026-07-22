"""
问题3(1) — 单次拾放路径优化 (容量=1)
算法：交替拾放, 容量=1, 贪心送料器访问顺序 + 组内最近邻排列
"""
import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

np.random.seed(42)

# ── 路径设置 ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.data_loader import load_mount_data, load_feeder_data
from src.utils.distance import manhattan_distance

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output')
TABLE_DIR = os.path.join(OUTPUT_DIR, 'tables')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# 优化的送料器分配：按 X 坐标排序后分配
# feeder 1-10: 各 4 个, feeder 11-30: 各 3 个
FEEDER_CAPACITIES = [4]*10 + [3]*20   # feeder_id 1-30


def build_optimized_assignment(mount_data, feeder_data):
    """构建基于 X 坐标排序的优化送料器分配"""
    n_mounts = mount_data.shape[0]
    n_feeders = feeder_data.shape[0]

    # 送料器位置
    feeder_pos = {}
    for i in range(n_feeders):
        fid = int(feeder_data[i, 0])
        fx = feeder_data[i, 1]
        fy = feeder_data[i, 2]
        feeder_pos[fid] = np.array([fx, fy])

    # 贴装点按 X 排序
    mounts_sorted = []
    for i in range(n_mounts):
        mounts_sorted.append({
            'mid': int(mount_data[i, 0]),
            'x':   mount_data[i, 1],
            'y':   mount_data[i, 2]
        })
    mounts_sorted.sort(key=lambda m: m['x'])

    # 分配
    feeder_groups = {fid: [] for fid in range(1, 31)}
    idx = 0
    for fid in range(1, 31):
        cap = FEEDER_CAPACITIES[fid - 1]
        for _ in range(cap):
            m = mounts_sorted[idx]
            feeder_groups[fid].append(m)
            idx += 1

    return feeder_groups, feeder_pos


def solve_single_pick_place(feeder_groups, feeder_pos):
    """贪心求解单次拾放路径
    
    算法:
      1. 每个送料器组内的贴装点按到送料器的曼哈顿距离升序排列
      2. 从 PCB 中心 (150, 100) 出发
      3. 每一步在所有有剩余贴装点的送料器中，选择使 
         dist(curr, feeder) + dist(feeder, next_mount) 最小的
      4. 移动到送料器（拾取），再移动到贴装点（放置）
    """
    # 每个送料器组内按最近邻排序
    sorted_groups = {}
    for fid, mounts in feeder_groups.items():
        sorted_groups[fid] = sorted(
            mounts,
            key=lambda m: abs(m['x'] - feeder_pos[fid][0]) + abs(m['y'] - feeder_pos[fid][1])
        )

    # 每个组的指针
    ptr = {fid: 0 for fid in range(1, 31)}
    remaining = sum(len(g) for g in sorted_groups.values())

    current_pos = np.array([150.0, 100.0])  # 从 PCB 中心出发
    sequence = []   # list of (step, feeder_id, mount_id, from_pos, to_feeder, to_mount)
    total_distance = 0.0

    for step in range(1, remaining + 1):
        best_fid = None
        best_cost = float('inf')
        best_mount = None

        for fid in range(1, 31):
            if ptr[fid] < len(sorted_groups[fid]):
                m = sorted_groups[fid][ptr[fid]]
                m_pos = np.array([m['x'], m['y']])
                cost = (manhattan_distance(current_pos, feeder_pos[fid]) +
                        manhattan_distance(feeder_pos[fid], m_pos))
                if cost < best_cost:
                    best_cost = cost
                    best_fid = fid
                    best_mount = m

        # 执行本次拾放
        m_pos = np.array([best_mount['x'], best_mount['y']])
        total_distance += best_cost

        sequence.append({
            'step':       step,
            'feeder_id':  best_fid,
            'mount_id':   best_mount['mid'],
            'from_pos':   current_pos.copy(),
            'feeder_pos': feeder_pos[best_fid].copy(),
            'mount_pos':  m_pos.copy(),
            'cost':       best_cost
        })

        current_pos = m_pos.copy()
        ptr[best_fid] += 1

    return sequence, total_distance


def save_results_and_visualize(sequence, total_distance, feeder_groups,
                               feeder_pos, mount_data):
    """保存结果并生成可视化"""
    n_mounts = mount_data.shape[0]

    # ── CSV 输出 ──
    csv_path = os.path.join(TABLE_DIR, 'p3_single_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Step', 'Feeder_ID', 'Mount_ID',
            'From_X', 'From_Y', 'Feeder_X', 'Feeder_Y',
            'Mount_X', 'Mount_Y', 'Segment_Distance_mm'
        ])
        for s in sequence:
            writer.writerow([
                s['step'], s['feeder_id'], s['mount_id'],
                f"{s['from_pos'][0]:.1f}", f"{s['from_pos'][1]:.1f}",
                f"{s['feeder_pos'][0]:.1f}", f"{s['feeder_pos'][1]:.1f}",
                f"{s['mount_pos'][0]:.1f}", f"{s['mount_pos'][1]:.1f}",
                f"{s['cost']:.1f}"
            ])
    print(f"[保存] 单次拾放结果 → {csv_path}")

    # ── 可视化 ──
    fig, ax = plt.subplots(figsize=(16, 10))

    # 送料器（绿色方块）
    feeder_xs = [feeder_pos[fid][0] for fid in range(1, 31)]
    feeder_ys = [feeder_pos[fid][1] for fid in range(1, 31)]
    ax.scatter(feeder_xs, feeder_ys, c='#2E8B57', s=80, marker='s',
               edgecolors='black', linewidth=0.8, zorder=5, label=f'送料器槽位 (30)')

    # 贴装点（蓝色圆点）
    mount_xs = [mount_data[i, 1] for i in range(n_mounts)]
    mount_ys = [mount_data[i, 2] for i in range(n_mounts)]
    ax.scatter(mount_xs, mount_ys, c='#4472C4', s=20, marker='o',
               alpha=0.7, zorder=3, label=f'贴装点 ({n_mounts})')

    # 路径（红色线段，只画关键段避免拥挤）
    # 每 10 步画一个点，降低视觉密度
    path_x = [150.0]  # 起点
    path_y = [100.0]
    for s in sequence:
        path_x.append(s['feeder_pos'][0])
        path_y.append(s['feeder_pos'][1])
        path_x.append(s['mount_pos'][0])
        path_y.append(s['mount_pos'][1])

    ax.plot(path_x, path_y, 'r-', linewidth=0.4, alpha=0.6, zorder=2, label='拾放路径')
    ax.scatter(150, 100, c='red', s=120, marker='*', zorder=8, label='起始点 (150,100)')

    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title(f'问题3(1) — 单次拾放路径 (总距离: {total_distance:,.0f} mm)',
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-10, 310)
    ax.set_ylim(-10, 310)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, 'p3_single_path.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[保存] 路径可视化 → {fig_path}")

    # ── 打印摘要 ──
    print(f"\n{'='*70}")
    print(f"问题3(1) — 单次拾放结果摘要")
    print(f"{'='*70}")
    print(f"总步数:          {len(sequence)}")
    print(f"总曼哈顿距离:    {total_distance:,.2f} mm")
    print(f"平均每步距离:    {total_distance/len(sequence):.2f} mm")
    print(f"使用送料器数:    30")
    print(f"贴装点数:        {n_mounts}")

    # 验证：feeder 组件计数
    counts = {}
    for s in sequence:
        counts[s['feeder_id']] = counts.get(s['feeder_id'], 0) + 1
    for fid in range(1, 31):
        assert counts.get(fid, 0) == len(feeder_groups[fid]), \
            f"Feeder {fid}: sequence has {counts.get(fid,0)}, expected {len(feeder_groups[fid])}"
    print(f"组件数验证:      [OK] 全部对应")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("问题3(1) — 单次拾放路径规划")
    print("=" * 70)

    mount_data = load_mount_data()
    feeder_data = load_feeder_data()

    # 构建优化分配
    feeder_groups, feeder_pos = build_optimized_assignment(mount_data, feeder_data)

    # 贪心求解
    sequence, total_distance = solve_single_pick_place(feeder_groups, feeder_pos)

    # 保存 & 可视化
    save_results_and_visualize(sequence, total_distance, feeder_groups,
                               feeder_pos, mount_data)
