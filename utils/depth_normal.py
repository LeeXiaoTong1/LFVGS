import numpy as np
import cv2
import math
from utils.graphics_utils import fov2focal

def depth_to_normal(depth_map, fov_x, fov_y, width, height):
    """
    将深度图转换为法线图
    :param depth_map: 输入的深度图 (H, W)，深度值应为浮点数。
    :param fov_x: 水平方向的视场角 (FoV)，单位为度数。
    :param fov_y: 垂直方向的视场角 (FoV)，单位为度数。
    :param width: 图像宽度。
    :param height: 图像高度。
    :return: 法线图 (H, W, 3)，RGB格式。
    """
    # 计算焦距
    focal_x = fov2focal(fov_x, width)
    focal_y = fov2focal(fov_y, height)

    # 计算梯度
    dz_dx = np.gradient(depth_map, axis=1)  # x方向梯度
    dz_dy = np.gradient(depth_map, axis=0)  # y方向梯度

    # 归一化 x 和 y 的梯度以适应焦距
    dz_dx /= focal_x
    dz_dy /= focal_y

    # 构造法线向量 (-dz/dx, -dz/dy, 1)
    normal = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth_map)))

    # 归一化法线向量
    normal_length = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= normal_length

    # # 转换到 [0, 255] 的 RGB 颜色空间
    # normal_rgb = (normal + 1.0) / 2.0 * 255.0
    # normal_rgb = normal_rgb.astype(np.uint8)

    return normal
