"""
问题3(2) — 双次拾放路径优化 (容量=2, 预拾取)
算法：送料器组内配对 → 放置顺序优化 → 贪心调度
"""
import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

FEEDER_CAPACITIES = [4]*10 + [3]*20


def build_optimized_assignment(mount_data, feeder_data):
    """基于 X 坐标排序的优化送料器分配（与 single 相同）"""
    n_mounts = mount_data.shape[0]

    feeder_pos = {}
    for i in range(feeder_data.shape[0]):
        fid = int(feeder_data[i, 0])
        feeder_pos[fid] = np.array([feeder_data[i, 1], feeder_data[i, 2]])

    mounts_sorted = []
    for i in range(n_mounts):
        mounts_sorted.append({
            'mid': int(mount_data[i, 0]),
            'x':   mount_data[i, 1],
            'y':   mount_data[i, 2]
        })
    mounts_sorted.sort(key=lambda m: m['x'])

    feeder_groups = {fid: [] for fid in range(1, 31)}
    idx = 0
    for fid in range(1, 31):
        cap = FEEDER_CAPACITIES[fid - 1]
        for _ in range(cap):
            feeder_groups[fid].append(mounts_sorted[idx])
            idx += 1

    return feeder_groups, feeder_pos


def pair_mounts_within_feeder(mounts, feeder_pos):
    """送料器组内配对（按曼哈顿距离最近优先配对）
    
    返回: list of tasks, each task is:
      {'type': 'pair', 'fid': fid, 'mounts': [m1, m2], 
       'order': [m_first, m_second], 'internal_cost': dist(F→first→second)}
      OR
      {'type': 'single', 'fid': fid, 'mounts': [m], 'internal_cost': dist(F→m)}
    """
    fid = mounts[0]['fid'] if 'fid' in mounts[0] else None  # will set later
    tasks = []

    # 计算两两曼哈顿距离矩阵（仅组内）
    n = len(mounts)
    if n == 0:
        return tasks

    pos_list = [np.array([m['x'], m['y']]) for m in mounts]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = manhattan_distance(pos_list[i], pos_list[j])

    # 配对算法: 贪心取最近的一对
    paired = [False] * n

    while True:
        best_i, best_j, best_dist = -1, -1, float('inf')
        for i in range(n):
            if paired[i]:
                continue
            for j in range(i+1, n):
                if paired[j]:
                    continue
                if dist_matrix[i, j] < best_dist:
                    best_dist = dist_matrix[i, j]
                    best_i, best_j = i, j

        if best_i == -1:
            break  # no more pairs possible

        paired[best_i] = True
        paired[best_j] = True

        # 优化放置顺序: 比较 F→Mi→Mj 和 F→Mj→Mi
        m_a = mounts[best_i]
        m_b = mounts[best_j]
        p_a = pos_list[best_i]
        p_b = pos_list[best_j]

        cost_ab = (manhattan_distance(feeder_pos, p_a) +
                   manhattan_distance(p_a, p_b))
        cost_ba = (manhattan_distance(feeder_pos, p_b) +
                   manhattan_distance(p_b, p_a))

        if cost_ab <= cost_ba:
            order = [m_a, m_b]
            internal_cost = cost_ab
        else:
            order = [m_b, m_a]
            internal_cost = cost_ba

        tasks.append({
            'type':          'pair',
            'mounts':        [m_a, m_b],
            'order':         order,    # [first_placed, second_placed]
            'internal_cost': internal_cost  # dist(F→first) + dist(first→second)
        })

    # 剩余未配对的作为单件任务
    for i in range(n):
        if not paired[i]:
            m = mounts[i]
            p = pos_list[i]
            tasks.append({
                'type':          'single',
                'mounts':        [m],
                'order':         [m],
                'internal_cost': manhattan_distance(feeder_pos, p)
            })

    return tasks


def solve_double_pick_place(feeder_groups, feeder_pos):
    """贪心求解双次拾放路径（容量=2, 预拾取）
    
    算法:
      1. 每个送料器组内按最近距离配对
      2. 对每个 pair: 优化放置顺序 (M_a→M_b vs M_b→M_a)
      3. 贪心调度所有任务（pairs 和 singles）
    """
    # 步骤 1: 各组内配对
    all_tasks = []  # list of task dicts with 'fid' added
    for fid, mounts in feeder_groups.items():
        tasks = pair_mounts_within_feeder(mounts, feeder_pos[fid])
        for t in tasks:
            t['fid'] = fid
            all_tasks.append(t)

    # 步骤 2: 贪心调度
    current_pos = np.array([150.0, 100.0])
    sequence = []
    total_distance = 0.0
    pending = list(all_tasks)  # shallow copy

    step = 0
    while pending:
        best_idx = -1
        best_cost = float('inf')

        for i, task in enumerate(pending):
            f_pos = feeder_pos[task['fid']]
            # 到达送料器的成本
            to_feeder_cost = manhattan_distance(current_pos, f_pos)
            # 内部成本（从送料器出发经过所有放置点）已在配对时计算
            total_cost = to_feeder_cost + task['internal_cost']
            if total_cost < best_cost:
                best_cost = total_cost
                best_idx = i

        task = pending.pop(best_idx)
        step += 1
        f_pos = feeder_pos[task['fid']]

        seq_entry = {
            'step':       step,
            'feeder_id':  task['fid'],
            'type':       task['type'],
            'mount_ids':  [m['mid'] for m in task['order']],
            'from_pos':   current_pos.copy(),
            'feeder_pos': f_pos.copy(),
            'mount_positions': [np.array([m['x'], m['y']]) for m in task['order']],
            'cost':       best_cost
        }
        sequence.append(seq_entry)
        total_distance += best_cost

        # 更新当前位置为任务最后一个放置点
        last_m = task['order'][-1]
        current_pos = np.array([last_m['x'], last_m['y']])

    return sequence, total_distance


def save_double_results(sequence, total_distance, feeder_groups,
                        feeder_pos, mount_data, single_distance=None):
    """保存结果、对比和可视化"""
    n_mounts = mount_data.shape[0]

    # ── CSV: 双次拾放详细结果 ──
    csv_path = os.path.join(TABLE_DIR, 'p3_double_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Step', 'Type', 'Feeder_ID',
            'Mount_ID_1', 'Mount_ID_2',
            'From_X', 'From_Y', 'Feeder_X', 'Feeder_Y',
            'Placed_1_X', 'Placed_1_Y', 'Placed_2_X', 'Placed_2_Y',
            'Segment_Distance_mm'
        ])
        for s in sequence:
            mid1 = s['mount_ids'][0]
            mid2 = s['mount_ids'][1] if len(s['mount_ids']) > 1 else ''
            mp1 = s['mount_positions'][0]
            mp2 = s['mount_positions'][1] if len(s['mount_positions']) > 1 else np.array([float('nan'), float('nan')])
            writer.writerow([
                s['step'], s['type'], s['feeder_id'],
                mid1, mid2,
                f"{s['from_pos'][0]:.1f}", f"{s['from_pos'][1]:.1f}",
                f"{s['feeder_pos'][0]:.1f}", f"{s['feeder_pos'][1]:.1f}",
                f"{mp1[0]:.1f}", f"{mp1[1]:.1f}",
                f"{mp2[0]:.1f}" if not np.isnan(mp2[0]) else '',
                f"{mp2[1]:.1f}" if not np.isnan(mp2[1]) else '',
                f"{s['cost']:.1f}"
            ])
    print(f"[保存] 双次拾放结果 → {csv_path}")

    # ── 对比（如果提供了 single 的结果） ──
    if single_distance is not None:
        cmp_path = os.path.join(TABLE_DIR, 'p3_comparison.csv')
        reduction = single_distance - total_distance
        pct = reduction / single_distance * 100
        with open(cmp_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['指标', '单次拾放(问题3-1)', '双次拾放(问题3-2)', '差值', '降低百分比'])
            writer.writerow([
                '总曼哈顿距离 (mm)',
                f'{single_distance:.2f}',
                f'{total_distance:.2f}',
                f'{reduction:.2f}',
                f'{pct:.2f}%'
            ])
            num_tasks_single = n_mounts
            num_tasks_double = len(sequence)
            writer.writerow([
                '任务数',
                f'{num_tasks_single}',
                f'{num_tasks_double}',
                f'{num_tasks_single - num_tasks_double}',
                ''
            ])
        print(f"[保存] 对比结果 → {cmp_path}")

        # ── 对比柱状图 ──
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['单次拾放\n(容量=1)', '双次拾放\n(容量=2)']
        values = [single_distance, total_distance]
        colors = ['#ED7D31', '#4472C4']
        bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.2, width=0.5)

        # 标注数值和减少百分比
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{val:,.0f} mm', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 减少箭头标注
        mid_x = 0.5
        ax.annotate(f'↓ {pct:.1f}%\n(减少 {reduction:,.0f} mm)',
                    xy=(mid_x, max(values) * 0.55),
                    fontsize=12, ha='center', color='#C00000',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE0E0', alpha=0.8))

        ax.set_ylabel('总曼哈顿距离 (mm)', fontsize=12)
        ax.set_title('问题3 — 单次 vs 双次拾放对比', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(values) * 1.2)

        plt.tight_layout()
        fig_path = os.path.join(FIGURE_DIR, 'p3_comparison.png')
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[保存] 对比图 → {fig_path}")

    # ── 双次拾放路径可视化 ──
    fig, ax = plt.subplots(figsize=(16, 10))

    # 送料器
    feeder_xs = [feeder_pos[fid][0] for fid in range(1, 31)]
    feeder_ys = [feeder_pos[fid][1] for fid in range(1, 31)]
    ax.scatter(feeder_xs, feeder_ys, c='#2E8B57', s=80, marker='s',
               edgecolors='black', linewidth=0.8, zorder=5, label=f'送料器槽位 (30)')

    # 贴装点
    mount_xs = [mount_data[i, 1] for i in range(n_mounts)]
    mount_ys = [mount_data[i, 2] for i in range(n_mounts)]
    ax.scatter(mount_xs, mount_ys, c='#4472C4', s=20, marker='o',
               alpha=0.7, zorder=3, label=f'贴装点 ({n_mounts})')

    # 路径（每隔几步画一点降低密度）
    path_x, path_y = [150.0], [100.0]
    for s in sequence:
        path_x.append(s['feeder_pos'][0])
        path_y.append(s['feeder_pos'][1])
        for mp in s['mount_positions']:
            path_x.append(mp[0])
            path_y.append(mp[1])

    ax.plot(path_x, path_y, 'r-', linewidth=0.4, alpha=0.6, zorder=2, label='拾放路径')
    ax.scatter(150, 100, c='red', s=120, marker='*', zorder=8, label='起始点 (150,100)')

    # 标记 pair (两个放置点之间加粗线)
    for s in sequence:
        if s['type'] == 'pair' and len(s['mount_positions']) == 2:
            mp1 = s['mount_positions'][0]
            mp2 = s['mount_positions'][1]
            ax.plot([mp1[0], mp2[0]], [mp1[1], mp2[1]],
                    'm-', linewidth=1.5, alpha=0.5, zorder=4)

    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title(f'问题3(2) — 双次拾放路径 (容量=2, 总距离: {total_distance:,.0f} mm)',
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-10, 310)
    ax.set_ylim(-10, 310)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, 'p3_double_path.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[保存] 路径可视化 → {fig_path}")

    # ── 摘要 ──
    n_pairs = sum(1 for s in sequence if s['type'] == 'pair')
    n_singles = sum(1 for s in sequence if s['type'] == 'single')
    print(f"\n{'='*70}")
    print(f"问题3(2) — 双次拾放结果摘要")
    print(f"{'='*70}")
    print(f"总任务数:        {len(sequence)} (Pair: {n_pairs}, Single: {n_singles})")
    print(f"总曼哈顿距离:    {total_distance:,.2f} mm")
    print(f"平均每任务距离:  {total_distance/len(sequence):.2f} mm")
    if single_distance is not None:
        reduction = single_distance - total_distance
        pct = reduction / single_distance * 100
        print(f"与单次拾放对比:  -{reduction:,.0f} mm (-{pct:.1f}%)")

    # 验证所有贴装点都已处理
    processed_mounts = set()
    for s in sequence:
        for mid in s['mount_ids']:
            processed_mounts.add(mid)
    assert len(processed_mounts) == n_mounts, \
        f"Processed {len(processed_mounts)} mounts, expected {n_mounts}"
    print(f"贴装点覆盖验证:  [OK] ({len(processed_mounts)} 个全部处理)")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("问题3(2) — 双次拾放路径规划（容量=2, 预拾取）")
    print("=" * 70)

    mount_data = load_mount_data()
    feeder_data = load_feeder_data()

    feeder_groups, feeder_pos = build_optimized_assignment(mount_data, feeder_data)

    # 贪心求解
    sequence, total_distance = solve_double_pick_place(feeder_groups, feeder_pos)

    # 同时计算单次拾放距离用于对比
    # (直接调用相同逻辑的单次版本)
    from src.problem3.pick_place_single import (
        build_optimized_assignment as build_single,
        solve_single_pick_place
    )
    fg_s, fp_s = build_single(mount_data, feeder_data)
    seq_single, dist_single = solve_single_pick_place(fg_s, fp_s)
    print(f"\n参考 — 单次拾放总距离: {dist_single:,.2f} mm")

    # 保存 & 可视化（含对比）
    save_double_results(sequence, total_distance, feeder_groups,
                        feeder_pos, mount_data, single_distance=dist_single)
