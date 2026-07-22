"""
问题3(3) — 双向取料路径优化 (容量=2, 异槽取料)
====================================================
策略：送料器配对 → 跨槽最近邻配对 → 双向拾取顺序优化 → 贪心调度
参考文献：彭乾伟 (2022) SMT贴片机双向取料策略

与预拾取 (pick_place_double.py) 的核心区别：
  - 预拾取：两个元件从同一送料器槽位拾取
  - 双向取料：两个元件从两个相邻送料器槽位同时拾取
    每次从槽位对 (2k-1, 2k) 各取一个元件，随后贴装

优势：配对更灵活（跨槽配对），减少往返送料器次数
代价：每次多一段送料器间移动（相邻槽位仅10mm）
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

# 送料器容量（与 single/double 一致）
FEEDER_CAPACITIES = [4] * 10 + [3] * 20


# ══════════════════════════════════════════════════════════════
#  第一阶段：槽位分配（与 single/double 完全相同）
# ══════════════════════════════════════════════════════════════

def build_optimized_assignment(mount_data, feeder_data):
    """基于 X 坐标排序的优化送料器分配（与 single/double 相同）

    定理2：X-排序分配在均衡约束下最小化 |X_mount - X_feeder| 之和。
    """
    feeder_pos = {}
    for i in range(feeder_data.shape[0]):
        fid = int(feeder_data[i, 0])
        feeder_pos[fid] = np.array([feeder_data[i, 1], feeder_data[i, 2]])

    mounts_sorted = []
    for i in range(mount_data.shape[0]):
        mounts_sorted.append({
            'mid': int(mount_data[i, 0]),
            'x': mount_data[i, 1],
            'y': mount_data[i, 2],
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


# ══════════════════════════════════════════════════════════════
#  第二阶段：送料器配对
# ══════════════════════════════════════════════════════════════

def build_feeder_pairs():
    """构建送料器对：相邻槽位配对 (1,2), (3,4), ..., (29,30)

    15 对送料器，每对包含：
      - (4,4) 型：5 对（槽位 1-10，各 4 个元件）
      - (3,3) 型：10 对（槽位 11-30，各 3 个元件）
    所有元件均可配对 → 零单件 → 50 个任务
    """
    return [(fid, fid + 1) for fid in range(1, 30, 2)]


def evaluate_placement_order(m_a, m_b, p_a, p_b, pos_f1, pos_f2):
    """评估四种拾取+放置顺序，返回最优方案

    四种排列：
      (1) f1 → f2 → a → b
      (2) f1 → f2 → b → a
      (3) f2 → f1 → a → b
      (4) f2 → f1 → b → a

    注意：送料器间距离 d(f1, f2) 在任务级别计入，不在 internal_cost 中。
    internal_cost = d(second_f, first_mount) + d(first_mount, second_mount)
    """
    candidates = []

    # (1) f1 → f2 → a → b
    cost = manhattan_distance(pos_f1, p_a) + manhattan_distance(p_a, p_b)
    candidates.append((cost, [m_a, m_b], pos_f1, pos_f2))

    # (2) f1 → f2 → b → a
    cost = manhattan_distance(pos_f1, p_b) + manhattan_distance(p_b, p_a)
    candidates.append((cost, [m_b, m_a], pos_f1, pos_f2))

    # (3) f2 → f1 → a → b
    cost = manhattan_distance(pos_f2, p_a) + manhattan_distance(p_a, p_b)
    candidates.append((cost, [m_a, m_b], pos_f2, pos_f1))

    # (4) f2 → f1 → b → a
    cost = manhattan_distance(pos_f2, p_b) + manhattan_distance(p_b, p_a)
    candidates.append((cost, [m_b, m_a], pos_f2, pos_f1))

    candidates.sort(key=lambda x: x[0])
    return candidates[0]  # (internal_cost, order, first_f, second_f)


def pair_mounts_bidirectional(mounts_f1, mounts_f2, pos_f1, pos_f2):
    """跨槽最近邻配对

    参数:
      mounts_f1, mounts_f2: 两个送料器各自的元件列表
      pos_f1, pos_f2: 两个送料器的位置

    返回: list[dict] — 配对/单件任务
    """
    n1, n2 = len(mounts_f1), len(mounts_f2)

    pos1 = [np.array([m['x'], m['y']]) for m in mounts_f1]
    pos2 = [np.array([m['x'], m['y']]) for m in mounts_f2]

    used1 = [False] * n1
    used2 = [False] * n2
    tasks = []

    # --- 贪心最近邻跨槽配对 ---
    while True:
        best_i, best_j = -1, -1
        best_dist = float('inf')
        for i in range(n1):
            if used1[i]:
                continue
            for j in range(n2):
                if used2[j]:
                    continue
                d = manhattan_distance(pos1[i], pos2[j])
                if d < best_dist:
                    best_dist = d
                    best_i, best_j = i, j

        if best_i == -1:
            break

        used1[best_i] = True
        used2[best_j] = True

        m_a = mounts_f1[best_i]
        m_b = mounts_f2[best_j]
        p_a = pos1[best_i]
        p_b = pos2[best_j]

        internal_cost, order, first_f, second_f = evaluate_placement_order(
            m_a, m_b, p_a, p_b, pos_f1, pos_f2
        )

        tasks.append({
            'type':          'pair',
            'mounts':        [m_a, m_b],
            'order':         order,
            'first_f_pos':   first_f.copy(),
            'second_f_pos':  second_f.copy(),
            'internal_cost': internal_cost,
        })

    # --- 单件（剩余未配对的） ---
    for i in range(n1):
        if not used1[i]:
            m = mounts_f1[i]
            p = pos1[i]
            tasks.append({
                'type':          'single',
                'mounts':        [m],
                'order':         [m],
                'first_f_pos':   pos_f1.copy(),
                'second_f_pos':  None,
                'internal_cost': manhattan_distance(pos_f1, p),
            })
    for j in range(n2):
        if not used2[j]:
            m = mounts_f2[j]
            p = pos2[j]
            tasks.append({
                'type':          'single',
                'mounts':        [m],
                'order':         [m],
                'first_f_pos':   pos_f2.copy(),
                'second_f_pos':  None,
                'internal_cost': manhattan_distance(pos_f2, p),
            })

    return tasks


# ══════════════════════════════════════════════════════════════
#  第三阶段：贪心调度
# ══════════════════════════════════════════════════════════════

def solve_bidirectional_pick(feeder_groups, feeder_pos):
    """贪心求解双向取料路径

    算法流程:
      1. 将 30 个送料器配成 15 对相邻对
      2. 每对内做跨槽最近邻配对
      3. 优化每对的拾取顺序（4种排列取最优）
      4. 贪心调度：每一步选使 total_cost 最小的下一个任务

    任务成本计算:
      - Pair任务: d(curr, first_f) + |first_f_x - second_f_x| + internal_cost
        (internal_cost = d(second_f, first_mount) + d(first_mount, second_mount))
      - Single任务: d(curr, feeder) + d(feeder, mount)
    """
    feeder_pairs = build_feeder_pairs()
    print(f"[信息] 送料器对: {len(feeder_pairs)} 对")

    # 步骤1-3: 配对 + 顺序优化
    all_tasks = []
    for fid1, fid2 in feeder_pairs:
        tasks = pair_mounts_bidirectional(
            feeder_groups[fid1], feeder_groups[fid2],
            feeder_pos[fid1], feeder_pos[fid2]
        )
        for t in tasks:
            t['fid1'] = fid1
            t['fid2'] = fid2
            all_tasks.append(t)

    n_pairs = sum(1 for t in all_tasks if t['type'] == 'pair')
    n_singles = sum(1 for t in all_tasks if t['type'] == 'single')
    print(f"[信息] 总任务: {len(all_tasks)} (Pair: {n_pairs}, Single: {n_singles})")

    # 步骤4: 贪心调度
    current_pos = np.array([150.0, 100.0])
    sequence = []
    total_distance = 0.0
    pending = list(all_tasks)

    step = 0
    while pending:
        best_idx = -1
        best_cost = float('inf')

        for i, task in enumerate(pending):
            ff = task['first_f_pos']

            if task['type'] == 'pair':
                sf = task['second_f_pos']
                d_to_ff = manhattan_distance(current_pos, ff)
                d_ff_sf = manhattan_distance(ff, sf)   # 送料器间 X 距离
                total_cost = d_to_ff + d_ff_sf + task['internal_cost']
            else:
                total_cost = manhattan_distance(current_pos, ff) + task['internal_cost']

            if total_cost < best_cost:
                best_cost = total_cost
                best_idx = i

        task = pending.pop(best_idx)
        step += 1

        # 记录
        seq_entry = {
            'step':            step,
            'fid1':            task['fid1'],
            'fid2':            task['fid2'],
            'type':            task['type'],
            'mount_ids':       [m['mid'] for m in task['order']],
            'from_pos':        current_pos.copy(),
            'first_f_pos':     task['first_f_pos'].copy(),
            'second_f_pos':    task['second_f_pos'].copy() if task['second_f_pos'] is not None else None,
            'mount_positions': [np.array([m['x'], m['y']]) for m in task['order']],
            'cost':            best_cost,
        }
        sequence.append(seq_entry)
        total_distance += best_cost

        last_m = task['order'][-1]
        current_pos = np.array([last_m['x'], last_m['y']])

    return sequence, total_distance


# ══════════════════════════════════════════════════════════════
#  第四阶段：保存 & 可视化
# ══════════════════════════════════════════════════════════════

def get_reference_results():
    """读取已有的单件取放和预拾取结果"""
    from src.problem3.pick_place_single import (
        build_optimized_assignment as build_single,
        solve_single_pick_place
    )
    from src.problem3.pick_place_double import (
        build_optimized_assignment as build_double,
        solve_double_pick_place
    )

    mount_data = load_mount_data()
    feeder_data = load_feeder_data()

    # 单件取放
    fg_s, fp_s = build_single(mount_data, feeder_data)
    _, single_dist = solve_single_pick_place(fg_s, fp_s)

    # 预拾取
    fg_d, fp_d = build_double(mount_data, feeder_data)
    seq_d, double_dist = solve_double_pick_place(fg_d, fp_d)
    double_tasks = len(seq_d)

    return single_dist, double_dist, double_tasks


def save_results(sequence, total_distance, feeder_groups, feeder_pos,
                  mount_data, single_dist, double_dist, double_tasks):
    """保存 CSV 结果、对比表和可视化"""
    n_mounts = mount_data.shape[0]

    # ────────────────────────────────────────────────
    # 1. 双向取料详细结果 CSV
    # ────────────────────────────────────────────────
    csv_path = os.path.join(TABLE_DIR, 'p3_bidirectional.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Step', 'Type',
            'Feeder_ID_1', 'Feeder_ID_2',
            'Mount_ID_1', 'Mount_ID_2',
            'From_X', 'From_Y',
            'Pick_1_X', 'Pick_1_Y',
            'Pick_2_X', 'Pick_2_Y',
            'Place_1_X', 'Place_1_Y',
            'Place_2_X', 'Place_2_Y',
            'FeederGap_mm', 'InternalDist_mm', 'Segment_Distance_mm'
        ])
        for s in sequence:
            fid1 = s['fid1']
            fid2 = s['fid2']
            mid1 = s['mount_ids'][0]
            mid2 = s['mount_ids'][1] if len(s['mount_ids']) > 1 else ''
            mp = s['mount_positions']
            fp1 = s['first_f_pos']
            fp2 = s['second_f_pos'] if s['second_f_pos'] is not None else None

            if s['type'] == 'pair':
                gap = manhattan_distance(fp1, fp2)
                internal = s['cost'] - manhattan_distance(
                    np.array([150.0, 100.0]) if s['step'] == 0 else s['from_pos'],
                    fp1
                ) - gap
            else:
                gap = 0
                internal = s['cost'] - manhattan_distance(s['from_pos'], fp1)

            writer.writerow([
                s['step'], s['type'],
                fid1, fid2,
                mid1, mid2,
                f"{s['from_pos'][0]:.1f}", f"{s['from_pos'][1]:.1f}",
                f"{fp1[0]:.1f}", f"{fp1[1]:.1f}",
                f"{fp2[0]:.1f}" if fp2 is not None else '',
                f"{fp2[1]:.1f}" if fp2 is not None else '',
                f"{mp[0][0]:.1f}", f"{mp[0][1]:.1f}",
                f"{mp[1][0]:.1f}" if len(mp) > 1 else '',
                f"{mp[1][1]:.1f}" if len(mp) > 1 else '',
                f"{gap:.1f}",
                f"{internal:.1f}",
                f"{s['cost']:.1f}"
            ])
    print(f"[保存] 双向取料详细结果 → {csv_path}")

    # ────────────────────────────────────────────────
    # 2. 三种策略对比 CSV
    # ────────────────────────────────────────────────
    n_pairs = sum(1 for s in sequence if s['type'] == 'pair')
    n_singles = sum(1 for s in sequence if s['type'] == 'single')
    bi_tasks = len(sequence)

    reduction_vs_single = single_dist - total_distance
    pct_vs_single = reduction_vs_single / single_dist * 100
    reduction_vs_double = double_dist - total_distance
    pct_vs_double = reduction_vs_double / double_dist * 100

    cmp_path = os.path.join(TABLE_DIR, 'p3_bidirectional_comparison.csv')
    with open(cmp_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            '指标',
            '单件取放(Q=1)',
            '预拾取(Q=2)',
            '双向取料(Q=2)',
            '双向 vs 单件',
            '双向 vs 预拾取'
        ])
        writer.writerow([
            '总曼哈顿距离 (mm)',
            f'{single_dist:.2f}',
            f'{double_dist:.2f}',
            f'{total_distance:.2f}',
            f'{reduction_vs_single:.2f} ({pct_vs_single:.2f}%)',
            f'{reduction_vs_double:.2f} ({pct_vs_double:.2f}%)'
        ])
        writer.writerow([
            '任务数',
            f'{n_mounts}',
            f'{double_tasks}',
            f'{bi_tasks}',
            f'{n_mounts - bi_tasks}',
            f'{double_tasks - bi_tasks}'
        ])
        writer.writerow([
            '配对任务数', '-', f'{double_tasks - 20}',  # 60-40=20 singles, 40 pairs
            f'{n_pairs}', '-', '-'
        ])
        writer.writerow([
            '单件任务数', f'{n_mounts}', '20',  # 20 singles in double
            f'{n_singles}', '-', '-'
        ])
    print(f"[保存] 三种策略对比 → {cmp_path}")

    # ────────────────────────────────────────────────
    # 3. 可视化 — 路径图
    # ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 10))

    # 送料器 (绿色方块)
    feeder_xs = [feeder_pos[fid][0] for fid in range(1, 31)]
    feeder_ys = [feeder_pos[fid][1] for fid in range(1, 31)]
    ax.scatter(feeder_xs, feeder_ys, c='#2E8B57', s=80, marker='s',
               edgecolors='black', linewidth=0.8, zorder=5,
               label=f'送料器槽位 (30)')

    # 高亮送料器对（半透明横向色条）
    pair_colors = plt.cm.Pastel1(np.linspace(0, 1, 15))
    for pi, (f1, f2) in enumerate(build_feeder_pairs()):
        x1 = feeder_pos[f1][0]
        x2 = feeder_pos[f2][0]
        ax.axhspan(-5, 5, xmin=(x1 - 2) / 300, xmax=(x2 + 2) / 300,
                   color=pair_colors[pi], alpha=0.15, zorder=1)

    # 贴装点 (蓝色圆点，按 feeder pair 着色)
    mount_xs = [mount_data[i, 1] for i in range(n_mounts)]
    mount_ys = [mount_data[i, 2] for i in range(n_mounts)]
    ax.scatter(mount_xs, mount_ys, c='#4472C4', s=20, marker='o',
               alpha=0.6, zorder=3, label=f'贴装点 ({n_mounts})')

    # 路径 (红色细线)
    path_x, path_y = [150.0], [100.0]
    for s in sequence:
        path_x.append(s['first_f_pos'][0])
        path_y.append(s['first_f_pos'][1])
        if s['second_f_pos'] is not None:
            path_x.append(s['second_f_pos'][0])
            path_y.append(s['second_f_pos'][1])
        for mp in s['mount_positions']:
            path_x.append(mp[0])
            path_y.append(mp[1])

    ax.plot(path_x, path_y, 'r-', linewidth=0.4, alpha=0.5, zorder=2,
            label='拾放路径')
    ax.scatter(150, 100, c='red', s=120, marker='*', zorder=8,
               label='起始点 (150,100)')

    # 标注配对连线 (两放置点之间的紫色连线)
    for s in sequence:
        if s['type'] == 'pair' and len(s['mount_positions']) == 2:
            mp1 = s['mount_positions'][0]
            mp2 = s['mount_positions'][1]
            ax.plot([mp1[0], mp2[0]], [mp1[1], mp2[1]],
                    'm-', linewidth=0.8, alpha=0.3, zorder=4)

    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title(
        f'问题3(3) — 双向取料路径 (总距离: {total_distance:,.0f} mm | '
        f'{n_pairs} 对 + {n_singles} 单件)',
        fontsize=14
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-10, 310)
    ax.set_ylim(-10, 310)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, 'p3_bidirectional_path.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[保存] 路径可视化 → {fig_path}")

    # ────────────────────────────────────────────────
    # 4. 可视化 — 三种策略对比柱状图
    # ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['单件取放\n(Q=1)', '预拾取\n(Q=2, 同槽)', '双向取料\n(Q=2, 异槽)']
    values = [single_dist, double_dist, total_distance]
    colors = ['#ED7D31', '#4472C4', '#70AD47']

    bars = ax.bar(labels, values, color=colors, edgecolor='white',
                  linewidth=1.2, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 400,
                f'{val:,.0f} mm', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # 改善标注
    mid_x1, mid_x2 = 0.5, 1.5
    pct_single = (single_dist - total_distance) / single_dist * 100
    pct_double = (double_dist - total_distance) / double_dist * 100
    ax.annotate(f'↓ {pct_single:.1f}%',
                xy=(mid_x1, max(values) * 0.50),
                fontsize=11, ha='center', color='#C00000',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#FFE0E0', alpha=0.8))
    ax.annotate(f'↓ {pct_double:.1f}%',
                xy=(mid_x2, max(values) * 0.65),
                fontsize=11, ha='center', color='#C00000',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#FFE0E0', alpha=0.8))

    ax.set_ylabel('总曼哈顿距离 (mm)', fontsize=12)
    ax.set_title('问题3 — 三种取料策略对比', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(values) * 1.25)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, 'p3_bidirectional_comparison.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[保存] 对比图 → {fig_path}")

    # ────────────────────────────────────────────────
    # 打印摘要
    # ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"问题3(3) — 双向取料结果摘要")
    print(f"{'=' * 70}")
    print(f"总任务数:        {bi_tasks} (Pair: {n_pairs}, Single: {n_singles})")
    print(f"总曼哈顿距离:    {total_distance:,.2f} mm")
    print(f"平均每任务距离:  {total_distance / bi_tasks:.2f} mm")
    print(f"")
    print(f"对比 — 单件取放:  {single_dist:>8,.0f} mm → {total_distance:>8,.0f} mm "
          f"(↓ {reduction_vs_single:>6,.0f} mm, ↓ {pct_vs_single:.2f}%)")
    print(f"对比 — 预拾取:    {double_dist:>8,.0f} mm → {total_distance:>8,.0f} mm "
          f"(↓ {reduction_vs_double:>6,.0f} mm, ↓ {pct_vs_double:.2f}%)")
    print(f"")
    print(f"任务数对比:      单件 {n_mounts} → 预拾取 {double_tasks} → 双向 {bi_tasks}")
    print(f"送料器到访次数:  {n_mounts} → {double_tasks} → {bi_tasks}")

    # 验证
    processed_mounts = set()
    for s in sequence:
        for mid in s['mount_ids']:
            processed_mounts.add(mid)
    assert len(processed_mounts) == n_mounts, \
        f"Processed {len(processed_mounts)} mounts, expected {n_mounts}"
    print(f"贴装点覆盖验证:  [OK] ({len(processed_mounts)} 个全部处理)")


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 70)
    print("问题3(3) — 双向取料路径规划（容量=2, 异槽取料）")
    print("参考: 彭乾伟 (2022) SMT贴片机双向取料策略")
    print("=" * 70)

    # 读取数据
    mount_data = load_mount_data()
    feeder_data = load_feeder_data()
    print(f"[数据] 贴装点: {mount_data.shape[0]} 个, "
          f"送料器: {feeder_data.shape[0]} 个槽位")

    # 构建优化分配
    feeder_groups, feeder_pos = build_optimized_assignment(mount_data, feeder_data)

    # 求解双向取料
    sequence, total_distance = solve_bidirectional_pick(feeder_groups, feeder_pos)

    # 获取参考结果
    print(f"\n{'─' * 40}")
    print("获取参考结果（单件取放 & 预拾取）...")
    single_dist, double_dist, double_tasks = get_reference_results()
    print(f"  单件取放距离: {single_dist:,.2f} mm")
    print(f"  预拾取距离:   {double_dist:,.2f} mm ({double_tasks} 任务)")

    # 保存 & 可视化（含对比）
    save_results(sequence, total_distance, feeder_groups, feeder_pos,
                  mount_data, single_dist, double_dist, double_tasks)
