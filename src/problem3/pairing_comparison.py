"""
问题3(2) — 预拾取配对策略对比
比较 3 种送料器组内配对策略的定量性能

策略:
  A — 最近邻配对 (Nearest-neighbor): 组内贪心取曼哈顿距离最近的两个配对
  B — X坐标相邻配对 (X-proximity): 按X排序后相邻配对
  C — 最远优先配对 (Farthest-first): 组内贪心取距离最远的两个配对

所有策略使用相同的送料器分配和相同的贪心访问顺序
"""
import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# ── 路径 ────────────────────────────────────────────────────
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

FEEDER_CAPACITIES = [4] * 10 + [3] * 20   # feeder 1-10: 4个, 11-30: 3个
START_POS = np.array([150.0, 100.0])
BASELINE_SINGLE_DIST = 36324.0  # mm — 问题3(1) 单次拾放距离


# ══════════════════════════════════════════════════════════════
#  送料器分配 (与 pick_place_double.py 完全一致)
# ══════════════════════════════════════════════════════════════

def build_feeder_assignment(mount_data, feeder_data):
    """基于 X 坐标排序的优化送料器分配
    返回: feeder_groups: dict fid -> list of mount dicts
          feeder_pos:    dict fid -> np.array([x, y])
    """
    feeder_pos = {}
    for i in range(feeder_data.shape[0]):
        fid = int(feeder_data[i, 0])
        feeder_pos[fid] = np.array([feeder_data[i, 1], feeder_data[i, 2]])

    mounts_sorted = []
    for i in range(mount_data.shape[0]):
        mounts_sorted.append({
            'mid': int(mount_data[i, 0]),
            'x':   mount_data[i, 1],
            'y':   mount_data[i, 2],
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
#  三种配对策略
# ══════════════════════════════════════════════════════════════

def pair_nearest_neighbor(mounts, feeder_pos):
    """策略A — 最近邻配对
    组内贪心取曼哈顿距离最近的两个贴装点配对，反复迭代
    """
    n = len(mounts)
    if n == 0:
        return []
    pos_list = [np.array([m['x'], m['y']]) for m in mounts]

    # 计算两端内部成本: F→i→j vs F→j→i，取较小值 + 记录最优放置顺序
    def internal_cost(i, j):
        cost_ij = manhattan_distance(feeder_pos, pos_list[i]) + manhattan_distance(pos_list[i], pos_list[j])
        cost_ji = manhattan_distance(feeder_pos, pos_list[j]) + manhattan_distance(pos_list[j], pos_list[i])
        if cost_ij <= cost_ji:
            return cost_ij, i, j  # first→second order
        else:
            return cost_ji, j, i

    paired = [False] * n
    tasks = []

    while True:
        best_i, best_j = -1, -1
        best_dist = float('inf')
        best_first, best_second = -1, -1

        for i in range(n):
            if paired[i]:
                continue
            for j in range(i + 1, n):
                if paired[j]:
                    continue
                d = manhattan_distance(pos_list[i], pos_list[j])
                if d < best_dist:
                    best_dist = d
                    best_i, best_j = i, j

        if best_i == -1:
            break

        paired[best_i] = True
        paired[best_j] = True
        cost, first, second = internal_cost(best_i, best_j)

        tasks.append({
            'type': 'pair',
            'mounts': [mounts[best_i], mounts[best_j]],
            'order': [mounts[first], mounts[second]],
            'internal_cost': cost,
        })

    # 剩余未配对 → single
    for i in range(n):
        if not paired[i]:
            tasks.append({
                'type': 'single',
                'mounts': [mounts[i]],
                'order': [mounts[i]],
                'internal_cost': manhattan_distance(feeder_pos, pos_list[i]),
            })

    return tasks


def pair_x_proximity(mounts, feeder_pos):
    """策略B — X坐标相邻配对
    组内按X升序排列后，相邻配对 (1st+2nd, 3rd+4th, ...)
    """
    n = len(mounts)
    if n == 0:
        return []

    sorted_mounts = sorted(mounts, key=lambda m: m['x'])
    pos_list = [np.array([m['x'], m['y']]) for m in sorted_mounts]

    tasks = []
    i = 0
    while i + 1 < n:
        # 优化放置顺序: F→i→i+1 vs F→i+1→i
        cost_ij = manhattan_distance(feeder_pos, pos_list[i]) + manhattan_distance(pos_list[i], pos_list[i + 1])
        cost_ji = manhattan_distance(feeder_pos, pos_list[i + 1]) + manhattan_distance(pos_list[i + 1], pos_list[i])
        if cost_ij <= cost_ji:
            order = [sorted_mounts[i], sorted_mounts[i + 1]]
            internal_cost = cost_ij
        else:
            order = [sorted_mounts[i + 1], sorted_mounts[i]]
            internal_cost = cost_ji

        tasks.append({
            'type': 'pair',
            'mounts': [sorted_mounts[i], sorted_mounts[i + 1]],
            'order': order,
            'internal_cost': internal_cost,
        })
        i += 2

    # 剩余未配对 (奇数情况)
    if i < n:
        tasks.append({
            'type': 'single',
            'mounts': [sorted_mounts[i]],
            'order': [sorted_mounts[i]],
            'internal_cost': manhattan_distance(feeder_pos, pos_list[i]),
        })

    return tasks


def pair_farthest_first(mounts, feeder_pos):
    """策略C — 最远优先配对
    组内贪心取曼哈顿距离最远的两个贴装点配对，反复迭代
    理由: 利用 pair 内两个放置点之间的"免费"移动
    """
    n = len(mounts)
    if n == 0:
        return []
    pos_list = [np.array([m['x'], m['y']]) for m in mounts]

    def internal_cost(i, j):
        cost_ij = manhattan_distance(feeder_pos, pos_list[i]) + manhattan_distance(pos_list[i], pos_list[j])
        cost_ji = manhattan_distance(feeder_pos, pos_list[j]) + manhattan_distance(pos_list[j], pos_list[i])
        if cost_ij <= cost_ji:
            return cost_ij, i, j
        else:
            return cost_ji, j, i

    paired = [False] * n
    tasks = []

    while True:
        best_i, best_j = -1, -1
        best_dist = -1.0  # 寻找最远

        for i in range(n):
            if paired[i]:
                continue
            for j in range(i + 1, n):
                if paired[j]:
                    continue
                d = manhattan_distance(pos_list[i], pos_list[j])
                if d > best_dist:
                    best_dist = d
                    best_i, best_j = i, j

        if best_i == -1:
            break

        paired[best_i] = True
        paired[best_j] = True
        cost, first, second = internal_cost(best_i, best_j)

        tasks.append({
            'type': 'pair',
            'mounts': [mounts[best_i], mounts[best_j]],
            'order': [mounts[first], mounts[second]],
            'internal_cost': cost,
        })

    for i in range(n):
        if not paired[i]:
            tasks.append({
                'type': 'single',
                'mounts': [mounts[i]],
                'order': [mounts[i]],
                'internal_cost': manhattan_distance(feeder_pos, pos_list[i]),
            })

    return tasks


# ══════════════════════════════════════════════════════════════
#  贪心调度器 (所有策略共用)
# ══════════════════════════════════════════════════════════════

def greedy_scheduler(all_tasks, feeder_pos):
    """贪心调度所有任务 (pairs + singles)
    从 START_POS 出发，每步选使 total_cost 最小的任务
    返回: total_distance, num_pairs, num_singles
    """
    current_pos = START_POS.copy()
    pending = list(all_tasks)
    total_distance = 0.0
    n_pairs = 0
    n_singles = 0

    while pending:
        best_idx = -1
        best_cost = float('inf')

        for i, task in enumerate(pending):
            f_pos = feeder_pos[task['fid']]
            to_feeder = manhattan_distance(current_pos, f_pos)
            total_cost = to_feeder + task['internal_cost']
            if total_cost < best_cost:
                best_cost = total_cost
                best_idx = i

        task = pending.pop(best_idx)
        total_distance += best_cost

        if task['type'] == 'pair':
            n_pairs += 1
        else:
            n_singles += 1

        # 移动到最后一个放置点
        last_m = task['order'][-1]
        current_pos = np.array([last_m['x'], last_m['y']])

    return total_distance, n_pairs, n_singles


# ══════════════════════════════════════════════════════════════
#  主流程: 对每种策略执行配对 + 调度
# ══════════════════════════════════════════════════════════════

def evaluate_strategy(name, pair_func, feeder_groups, feeder_pos):
    """对给定配对策略执行完整评估"""
    all_tasks = []
    for fid, mounts in feeder_groups.items():
        tasks = pair_func(mounts, feeder_pos[fid])
        for t in tasks:
            t['fid'] = fid
            all_tasks.append(t)

    total_dist, n_pairs, n_singles = greedy_scheduler(all_tasks, feeder_pos)
    return {
        'strategy': name,
        'total_distance': total_dist,
        'num_pairs': n_pairs,
        'num_singles': n_singles,
    }


def main():
    print("=" * 75)
    print("问题3(2) — 预拾取配对策略对比")
    print("=" * 75)

    # ── 加载数据 & 构建送料器分配 ──
    mount_data = load_mount_data()
    feeder_data = load_feeder_data()
    feeder_groups, feeder_pos = build_feeder_assignment(mount_data, feeder_data)

    # 验证
    total_mounts = sum(len(g) for g in feeder_groups.values())
    assert total_mounts == 100, f"Expected 100 mounts, got {total_mounts}"
    print(f"送料器分配: [OK] 100 个贴装点 → 30 个送料器")
    print(f"  前10个送料器: 各4个, 后20个: 各3个\n")

    # ── 评估三种策略 ──
    strategies = [
        ("A — 最近邻配对", pair_nearest_neighbor),
        ("B — X坐标相邻配对", pair_x_proximity),
        ("C — 最远优先配对", pair_farthest_first),
    ]

    results = []
    for name, pair_func in strategies:
        print(f"计算策略: {name} ...")
        r = evaluate_strategy(name, pair_func, feeder_groups, feeder_pos)
        results.append(r)
        print(f"  总距离: {r['total_distance']:,.0f} mm  "
              f"(Pairs: {r['num_pairs']}, Singles: {r['num_singles']})\n")

    # ── 取策略A为基准计算改善率 ──
    baseline_dist = results[0]['total_distance']

    # ── 控制台对比表 ──
    print("=" * 75)
    print(f"{'策略':<28}{'总距离 (mm)':>14}{'Pairs':>8}{'Singles':>10}{'vs基线 (%)':>12}")
    print("-" * 75)
    for r in results:
        improvement = (baseline_dist - r['total_distance']) / baseline_dist * 100
        sign = '-' if improvement < 0 else '+'
        print(f"{r['strategy']:<28}{r['total_distance']:>14,.0f}"
              f"{r['num_pairs']:>8}{r['num_singles']:>10}"
              f"{sign}{abs(improvement):>10.2f}%")
    print("-" * 75)
    print(f"  基线: 策略A (最近邻) = {baseline_dist:,.0f} mm")
    print(f"  参考: 单次拾放     = {BASELINE_SINGLE_DIST:,.0f} mm")
    print("=" * 75)

    # ── 保存 CSV ──
    csv_path = os.path.join(TABLE_DIR, 'p3_pairing_comparison.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Strategy', 'Total_Distance_mm', 'Num_Pairs',
                         'Num_Singles', 'Improvement_vs_Baseline_pct'])
        for r in results:
            imp = (baseline_dist - r['total_distance']) / baseline_dist * 100
            writer.writerow([
                r['strategy'],
                f"{r['total_distance']:.2f}",
                r['num_pairs'],
                r['num_singles'],
                f"{imp:.2f}"
            ])
    print(f"\n[保存] 对比结果 → {csv_path}")

    # ── 分组柱状图 ──
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [r['strategy'] for r in results]
    values = [r['total_distance'] for r in results]

    colors = ['#4472C4', '#ED7D31', '#70AD47']
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, values, color=colors, edgecolor='white',
                  linewidth=1.2, width=0.55)

    # 标注距离值
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 400,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # 标注改善率
        imp = (baseline_dist - val) / baseline_dist * 100
        if imp != 0:
            sign = '-' if imp < 0 else '+'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1200,
                    f'({sign}{abs(imp):.1f}%)', ha='center', va='bottom',
                    fontsize=9, color='#C00000' if abs(imp) > 0.1 else '#555555')

    # 水平虚线: 单次拾放基线
    ax.axhline(y=BASELINE_SINGLE_DIST, color='#A0A0A0', linewidth=1.5,
               linestyle='--', label=f'单次拾放基线 ({BASELINE_SINGLE_DIST:,.0f} mm)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('总曼哈顿距离 (mm)', fontsize=12)
    ax.set_title('问题3(2) — 预拾取配对策略对比\n(相同送料器分配 & 贪心调度, 仅组内配对方式不同)',
                 fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    y_max = max(max(values), BASELINE_SINGLE_DIST)
    ax.set_ylim(0, y_max * 1.15)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, 'p3_pairing_comparison.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[保存] 对比图 → {fig_path}")

    print("\n完成。所有策略均已计算并比较。\n")


if __name__ == '__main__':
    main()
