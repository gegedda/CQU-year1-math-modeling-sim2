"""
距离计算模块
支持欧氏距离（问题1 & 2）和曼哈顿距离（问题3）
使用方法：from utils.distance import euclidean_distance_matrix, manhattan_distance_matrix, total_path_length
"""

import numpy as np


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    计算欧氏距离矩阵
    
    参数:
        coords: shape=(n, 2), 每行 (x, y)
    
    返回:
        shape=(n, n) 的对称距离矩阵, D[i,j] = sqrt((xi-xj)^2 + (yi-yj)^2)
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def manhattan_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    计算曼哈顿距离矩阵
    
    参数:
        coords: shape=(n, 2), 每行 (x, y)
    
    返回:
        shape=(n, n) 的对称距离矩阵, D[i,j] = |xi-xj| + |yi-yj|
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sum(np.abs(diff), axis=2)


def total_path_length(order: list, dist_matrix: np.ndarray) -> float:
    """
    计算给定访问顺序的路径总长度
    
    参数:
        order: 访问顺序（索引列表）, 如 [0, 3, 1, 2, 0]
        dist_matrix: 距离矩阵
    
    返回:
        总距离
    """
    total = 0.0
    for i in range(len(order) - 1):
        total += dist_matrix[order[i], order[i+1]]
    return total


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """两点间的欧氏距离"""
    return np.sqrt(np.sum((p1 - p2) ** 2))


def manhattan_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """两点间的曼哈顿距离"""
    return np.sum(np.abs(p1 - p2))


# 测试代码
if __name__ == "__main__":
    coords = np.array([[0, 0], [3, 4], [6, 8]])
    print("欧氏距离矩阵:\n", euclidean_distance_matrix(coords))
    print("曼哈顿距离矩阵:\n", manhattan_distance_matrix(coords))
    
    order = [0, 1, 2, 0]
    print(f"路径 {order} 总长 (欧氏) = {total_path_length(order, euclidean_distance_matrix(coords)):.2f}")
