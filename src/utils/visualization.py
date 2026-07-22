"""
可视化模块
支持路径图、收敛曲线、对比图等各种论文用图
使用方法：from utils.visualization import plot_path, plot_convergence, plot_comparison
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，用于服务器/脚本环境
import matplotlib.pyplot as plt
import os

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output", "figures")

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_path(coords: np.ndarray, order: list, title: str, filename: str,
              origin: tuple = (0, 0), figsize=(10, 8)):
    """
    绘制路径图
    
    参数:
        coords: shape=(n, 2), 所有点的坐标
        order: 访问顺序（索引列表）
        title: 图表标题
        filename: 输出文件名（不含路径）
        origin: 原点坐标
        figsize: 图大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制所有点
    xs = coords[:, 0]
    ys = coords[:, 1]
    ax.scatter(xs, ys, c='blue', s=10, alpha=0.6, label='钻孔点')
    ax.scatter(origin[0], origin[1], c='red', s=100, marker='*', label='原点 O', zorder=5)
    
    # 绘制路径
    path_coords = coords[order]
    ax.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=0.5, alpha=0.7)
    
    # 标注起点终点
    ax.annotate('O', origin, xytext=(5, 5), textcoords='offset points', fontsize=12, color='red')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"[可视化] 已保存: {filename}")


def plot_convergence(history: list, title: str, filename: str, figsize=(8, 5)):
    """
    绘制收敛曲线
    
    参数:
        history: 每次迭代的目标函数值列表
        title: 图表标题
        filename: 输出文件名
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(history, 'b-', linewidth=1, alpha=0.8)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('路径总长度 (mm)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"[可视化] 已保存: {filename}")


def plot_comparison(results: dict, title: str, filename: str, figsize=(10, 6)):
    """
    绘制算法对比柱状图
    
    参数:
        results: {算法名: [4个规模的结果值], ...}
        title: 标题
        filename: 文件名
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scales = ['n=50', 'n=198', 'n=442', 'n=1173']
    x = np.arange(len(scales))
    width = 0.8 / len(results)
    
    for i, (algo, values) in enumerate(results.items()):
        bars = ax.bar(x + i * width, values, width, label=algo)
    
    ax.set_xlabel('PCB 规模')
    ax.set_ylabel('路径总长度 (mm)')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(scales)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"[可视化] 已保存: {filename}")


def plot_grouped_path(coords: np.ndarray, groups: list, group_colors: list,
                      title: str, filename: str, figsize=(10, 8)):
    """
    绘制按组着色的路径图（用于问题2 多孔径）
    
    参数:
        coords: 所有点坐标
        groups: [group1_indices, group2_indices, ...] 每组在 coords 中的索引
        group_colors: 每组对应的颜色
        title: 标题
        filename: 文件名
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(0, 0, c='red', s=100, marker='*', label='原点 O', zorder=5)
    
    for i, (group, color) in enumerate(zip(groups, group_colors)):
        g_coords = coords[group]
        ax.scatter(g_coords[:, 0], g_coords[:, 1], c=color, s=15, alpha=0.7,
                   label=f'{chr(65+i)}型孔')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"[可视化] 已保存: {filename}")
