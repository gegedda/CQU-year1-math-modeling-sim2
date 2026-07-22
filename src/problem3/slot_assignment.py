"""
问题3 — 送料器槽位分配优化
功能：
  1. 分析原始（随机）Feeder_ID 分配的性能
  2. 设计基于 X 坐标排序的优化分配方案
  3. 对比两种方案并输出结果
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


def load_and_validate():
    """加载数据并验证"""
    mount_data = load_mount_data()   # shape=(100,4): Mount_ID, X, Y, Feeder_ID
    feeder_data = load_feeder_data() # shape=(30,3):  Feeder_ID, X, Y

    # 构建 feeder_id → (fx, fy) 的字典
    feeder_pos = {}
    feeder_ids_sorted = []
    for i in range(feeder_data.shape[0]):
        fid = int(feeder_data[i, 0])
        fx = feeder_data[i, 1]
        fy = feeder_data[i, 2]
        feeder_pos[fid] = np.array([fx, fy])
        feeder_ids_sorted.append(fid)
    feeder_ids_sorted.sort()

    # 验证 feeder X 坐标从 5 到 295，步长 10
    xs = [feeder_pos[fid][0] for fid in feeder_ids_sorted]
    assert len(xs) == 30, f"Feeder count expected 30, got {len(xs)}"
    assert abs(xs[0] - 5) < 0.01, f"First feeder X should be 5, got {xs[0]}"
    assert abs(xs[-1] - 295) < 0.01, f"Last feeder X should be 295, got {xs[-1]}"

    return mount_data, feeder_pos, feeder_ids_sorted


def compute_avg_dx(mount_data, feeder_pos, assignment_map):
    """计算给定分配方案下的平均 |mount_x - feeder_x|
    
    assignment_map: dict mount_idx → feeder_id
    """
    total_dx = 0.0
    for i in range(mount_data.shape[0]):
        mid = int(mount_data[i, 0])
        mx = mount_data[i, 1]
        fid = assignment_map[mid]
        fx = feeder_pos[fid][0]
        total_dx += abs(mx - fx)
    return total_dx / mount_data.shape[0]


def analyze_random_assignment(mount_data, feeder_pos):
    """分析原始随机分配的性能"""
    n_mounts = mount_data.shape[0]
    print("=" * 70)
    print("【步骤 1】原始（随机）Feeder_ID 分配分析")
    print("=" * 70)

    # 按原始 Feeder_ID 分组
    feeder_groups = {}
    for i in range(n_mounts):
        fid = int(mount_data[i, 3])
        if fid not in feeder_groups:
            feeder_groups[fid] = []
        feeder_groups[fid].append({
            'mid': int(mount_data[i, 0]),
            'mx':  mount_data[i, 1],
            'my':  mount_data[i, 2]
        })

    # 统计每个 feeder 的组件数
    assigned_fids = sorted(feeder_groups.keys())
    print(f"使用到的送料器数量: {len(assigned_fids)} (共 30 个)")
    print(f"\n{'Feeder_ID':<12}{'X(mm)':<10}{'组件数':<10}{'|mount_x - feeder_x| 均值'}")
    print("-" * 65)

    random_dx_list = []
    for fid in assigned_fids:
        mounts = feeder_groups[fid]
        count = len(mounts)
        fx = feeder_pos[fid][0]
        dx_vals = [abs(m['mx'] - fx) for m in mounts]
        avg_dx = np.mean(dx_vals)
        random_dx_list.extend(dx_vals)
        print(f"{fid:<12}{fx:<10.0f}{count:<10}{avg_dx:<12.1f}")

    overall_avg_dx = np.mean(random_dx_list)
    counts = [len(feeder_groups.get(fid, [])) for fid in range(1, 31)]
    print(f"\n原始分配统计:")
    print(f"  每 feeder 组件数 — 最小: {min(counts)}, 最大: {max(counts)}, "
          f"均值: {np.mean(counts):.2f}, 标准差: {np.std(counts):.2f}")
    print(f"  平均 |mount_x - feeder_x| = {overall_avg_dx:.2f} mm  ← 基准值")

    return overall_avg_dx, counts


def design_optimized_assignment(mount_data, feeder_ids_sorted):
    """基于 X 坐标排序的优化分配
    
    算法：
      1. 将 100 个贴装点按 X 坐标升序排列
      2. 送料器槽位已按 X 升序排列（ID 1→30, X: 5→295）
      3. 前 10 个 feeder 各分配 4 个 (共 40)，后 20 个各分配 3 个 (共 60)
    """
    n_mounts = mount_data.shape[0]
    n_feeders = len(feeder_ids_sorted)

    print("\n" + "=" * 70)
    print("【步骤 2】X 坐标排序优化分配")
    print("=" * 70)

    # 按 X 坐标排序贴装点（保留原始信息）
    mount_list = []
    for i in range(n_mounts):
        mount_list.append({
            'mid':  int(mount_data[i, 0]),
            'mx':   mount_data[i, 1],
            'my':   mount_data[i, 2],
            'old_fid': int(mount_data[i, 3])
        })
    mount_list.sort(key=lambda m: m['mx'])

    # 分配规则: feeder 1-10 各 4 个, feeder 11-30 各 3 个
    new_assignment = {}   # mount_id → feeder_id
    feeder_new_groups = {fid: [] for fid in feeder_ids_sorted}

    idx = 0
    for i, fid in enumerate(feeder_ids_sorted):
        count = 4 if i < 10 else 3
        for _ in range(count):
            if idx < n_mounts:
                m = mount_list[idx]
                new_assignment[m['mid']] = fid
                feeder_new_groups[fid].append(m)
                idx += 1

    assert idx == n_mounts, f"Not all mounts assigned: {idx} != {n_mounts}"

    print(f"分配方案: 前 10 个送料器各 4 个, 后 20 个各 3 个")
    print(f"总计: {10*4 + 20*3} 个贴装点 [OK]")
    print(f"\n{'Feeder_ID':<12}{'X(mm)':<10}{'组件数':<10}示例 Mount_IDs")
    print("-" * 65)
    for fid in feeder_ids_sorted:
        mounts = feeder_new_groups[fid]
        sample_ids = [str(m['mid']) for m in mounts[:3]]
        print(f"{fid:<12}{fx_map[fid]:<10.0f}{len(mounts):<10}{', '.join(sample_ids)}")
        if len(mounts) > 3:
            print(f"{'':12}{'':10}{'':10}(还有 {len(mounts)-3} 个)")

    return new_assignment, feeder_new_groups, mount_list


def compare_assignments(mount_data, feeder_pos, old_assignment, new_assignment):
    """对比两种分配方案"""
    print("\n" + "=" * 70)
    print("【步骤 3】分配方案对比")
    print("=" * 70)

    # 原始分配
    old_map = {int(mount_data[i, 0]): int(mount_data[i, 3]) for i in range(mount_data.shape[0])}
    old_avg = compute_avg_dx(mount_data, feeder_pos, old_map)
    new_avg = compute_avg_dx(mount_data, feeder_pos, new_assignment)
    improvement = (old_avg - new_avg) / old_avg * 100

    print(f"\n{'指标':<40}{'原始随机分配':>15}{'优化分配':>15}{'改善':>15}")
    print("-" * 85)
    print(f"{'平均|mount_x - feeder_x| (mm)':<40}{old_avg:>15.2f}{new_avg:>15.2f}{improvement:>14.1f}%")

    # 组件数分布对比
    old_counts = [0] * 30
    new_counts = [0] * 30
    for i in range(30):
        fid = i + 1
        old_counts[i] = sum(1 for v in old_map.values() if v == fid)
        new_counts[i] = sum(1 for v in new_assignment.values() if v == fid)

    print(f"\n组件数分布对比:")
    print(f"{'':20}{'原始分配':>12}{'优化分配':>12}")
    print(f"{'每 feeder 最小值':<20}{min(old_counts):>12}{min(new_counts):>12}")
    print(f"{'每 feeder 最大值':<20}{max(old_counts):>12}{max(new_counts):>12}")
    print(f"{'每 feeder 均值':<20}{np.mean(old_counts):>12.2f}{np.mean(new_counts):>12.2f}")
    print(f"{'每 feeder 标准差':<20}{np.std(old_counts):>12.2f}{np.std(new_counts):>12.2f}")

    if new_avg < old_avg:
        print(f"\n[OK] 优化方案将平均 |dx| 从 {old_avg:.2f} 降至 {new_avg:.2f} mm"
              f"（降低 {improvement:.1f}%）")
    else:
        print(f"\n[WARN] 优化方案未改善，请检查算法")

    return old_avg, new_avg, improvement, old_counts, new_counts


def save_results(mount_data, new_assignment, feeder_new_groups,
                 feeder_ids_sorted, old_counts, new_counts):
    """保存结果到 CSV 和图表"""
    n_mounts = mount_data.shape[0]

    # ── CSV: 优化分配表 ──
    csv_path = os.path.join(TABLE_DIR, 'p3_optimized_assignment.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Mount_ID', 'X', 'Y', 'Old_Feeder_ID', 'New_Feeder_ID'])
        for i in range(n_mounts):
            mid = int(mount_data[i, 0])
            mx  = mount_data[i, 1]
            my  = mount_data[i, 2]
            old_fid = int(mount_data[i, 3])
            new_fid = new_assignment[mid]
            writer.writerow([mid, mx, my, old_fid, new_fid])
    print(f"\n[保存] 优化分配表 → {csv_path}")

    # ── 柱状图: 送料器组件分布 ──
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(1, 31)
    width = 0.35

    bars1 = ax.bar(x - width/2, old_counts, width, label='原始随机分配',
                   color='#ED7D31', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, new_counts, width, label='优化 X-sort 分配',
                   color='#4472C4', edgecolor='white', linewidth=0.5)

    ax.axhline(y=100/30, color='black', linestyle='--', linewidth=1,
               label=f'理想均值 = 3.33')

    ax.set_xlabel('送料器编号', fontsize=12)
    ax.set_ylabel('贴装点数量', fontsize=12)
    ax.set_title('问题3 — 送料器槽位分配对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(range(1, 31), rotation=45, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(old_counts), max(new_counts)) + 1.5)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, 'p3_feeder_distribution.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[保存] 分布对比图 → {fig_path}")


# ══════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mount_data, feeder_pos, feeder_ids_sorted = load_and_validate()

    # 构建全局 fx_map（供 design_optimized_assignment 中打印用）
    fx_map = {fid: feeder_pos[fid][0] for fid in feeder_ids_sorted}

    # 步骤1: 分析原始随机分配
    old_avg_dx, old_counts = analyze_random_assignment(mount_data, feeder_pos)

    # 步骤2: 设计优化分配
    new_assignment, feeder_new_groups, sorted_mounts = \
        design_optimized_assignment(mount_data, feeder_ids_sorted)

    # 步骤3: 对比
    old_avg, new_avg, improvement, _, new_counts = compare_assignments(
        mount_data, feeder_pos,
        {int(mount_data[i,0]): int(mount_data[i,3]) for i in range(mount_data.shape[0])},
        new_assignment
    )

    # 保存结果
    save_results(mount_data, new_assignment, feeder_new_groups,
                 feeder_ids_sorted, old_counts, new_counts)

    # 返回关键数据供其他模块使用
    print("\n" + "=" * 70)
    print("槽位分配优化完成。输出供 pick_place_single / pick_place_double 使用。")
    print("=" * 70)
