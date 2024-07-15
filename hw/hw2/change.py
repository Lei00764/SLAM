"""
@File    :   change.py
@Time    :   2024/05/28 21:35:27
@Author  :   Xiang Lei 
@Version :   1.0
@Desc    :   计算两个四元数之间的差距，使用 MAS，MSE 和 余弦相似度
"""

import numpy as np

q_real = np.array([0.02259146, 0.03212753, -0.00783342, 0.99924089])
q_pred1 = np.array([0.02256674, 0.01091099, -0.0113942, 0.99962086])
q_pred2 = np.array([-0.03661601, -0.01350624, 0.01240963, 0.99916107])
q_pred3 = np.array([0.0298867, -0.00375415, 0.004064, 0.99953798])
q_pred4 = np.array([2.98866996e-02, -3.75415252e-03, -1.12251091e-04, 9.99537981e-01])


def compute_MAE(q1, q2):
    return np.mean(np.abs(q1 - q2))


def compute_MSE(q1, q2):
    return np.mean((q1 - q2) ** 2)


def compute_cosine_similarity(q1, q2):
    return np.dot(q1, q2) / (np.linalg.norm(q1) * np.linalg.norm(q2))


def compute_metrics(q_real, q_pred):
    MAE = compute_MAE(q_real, q_pred)
    MSE = compute_MSE(q_real, q_pred)
    cosine_similarity = compute_cosine_similarity(q_real, q_pred)
    return MAE, MSE, cosine_similarity


if __name__ == "__main__":
    MAE, MSE, cosine_similarity = compute_metrics(q_real, q_pred4)
    print("MAE:", MAE)
    print("MSE:", MSE)
    print("Cosine Similarity:", cosine_similarity)
