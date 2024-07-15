"""
@File    :   compute.py
@Time    :   2024/05/28 21:22:02
@Author  :   Xiang Lei 
@Version :   1.0
@Desc    :   使用两个时间点的四元数，计算两个时间点之间的旋转四元数
"""

import numpy as np


def quaternion_conjugate(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w])


q1 = np.array([0.6579, 0.6492, -0.2920, -0.2460])

q2 = np.array([0.6475, 0.6422, -0.2963, -0.2838])

q1_conjugate = quaternion_conjugate(q1)

q_delta = quaternion_multiply(q2, q1_conjugate)


print("q_delta (relative quaternion):")
print(q_delta)
