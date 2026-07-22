"""
共用数据加载模块
功能：读取所有 B 题的 CSV 数据文件，统一返回 numpy 数组格式
使用方法：from utils.data_loader import load_drill_data, load_mount_data, load_feeder_data
"""

import os
import numpy as np
import pandas as pd

# 数据根目录（相对于项目根目录）
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")


def load_drill_data(n_points: int) -> np.ndarray:
    """
    加载钻孔数据（问题1 和 问题2 共用）
    
    参数:
        n_points: 点数规模，可选 50, 198, 442, 1173
    
    返回:
        np.ndarray, shape=(n, 4), 列: [ID(int), X(float), Y(float), Type(str)]
        其中 Type 是孔径类型: 'A'/'B'/'C'
    """
    filename = f"Q1_Q2_drill_data{n_points}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # 提取关键列
    ids = df["ID"].values
    xs = df["X"].values.astype(float)
    ys = df["Y"].values.astype(float)
    types = df["Type"].values if "Type" in df.columns else np.array(["A"] * len(df))
    
    return np.column_stack([ids, xs, ys, types])


def load_mount_data() -> np.ndarray:
    """
    加载贴装数据（问题3）
    
    返回:
        np.ndarray, shape=(100, 4), 列: [Mount_ID(int), X(float), Y(float), Feeder_ID(int)]
    """
    filepath = os.path.join(DATA_DIR, "Q3_mount_data.csv")
    df = pd.read_csv(filepath)
    
    ids = df["Mount_ID"].values
    xs = df["X"].values.astype(float)
    ys = df["Y"].values.astype(float)
    feeder_ids = df["Feeder_ID"].values.astype(int)
    
    return np.column_stack([ids, xs, ys, feeder_ids])


def load_feeder_data() -> np.ndarray:
    """
    加载送料器槽位数据（问题3）
    
    返回:
        np.ndarray, shape=(30, 3), 列: [Feeder_ID(int), X(float), Y(float)]
    """
    filepath = os.path.join(DATA_DIR, "Q3_feeder_data.csv")
    df = pd.read_csv(filepath)
    
    ids = df["Feeder_ID"].values
    xs = df["X"].values.astype(float)
    ys = df["Y"].values.astype(float)
    
    return np.column_stack([ids, xs, ys])


def get_coords(data: np.ndarray) -> np.ndarray:
    """从数据数组中提取 X, Y 坐标列"""
    return data[:, 1:3].astype(float)


def get_types(data: np.ndarray) -> np.ndarray:
    """从数据数组中提取 Type 列"""
    return data[:, 3] if data.shape[1] >= 4 else np.array([])


# 测试代码
if __name__ == "__main__":
    for n in [50, 198, 442, 1173]:
        data = load_drill_data(n)
        print(f"n={n}: {data.shape[0]} 个点, "
              f"A={sum(data[:,3]=='A')}, "
              f"B={sum(data[:,3]=='B')}, "
              f"C={sum(data[:,3]=='C')}")
    
    mount = load_mount_data()
    feeder = load_feeder_data()
    print(f"\n贴装点: {mount.shape[0]} 个")
    print(f"送料器: {feeder.shape[0]} 个槽位")
