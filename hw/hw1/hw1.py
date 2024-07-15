"""
@File    :   hw1.py
@Time    :   2024/04/15 20:21:21
@Author  :   Xiang Lei 
@Version :   1.0
@Desc    :   None
"""

import cv2
import numpy as np


def rotate_image(image_path, angle):
    image = cv2.imread(image_path)
    w, h, c = image.shape
    M = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (h, w))
    return rotated_image


image_path = "D:/third-year/SLAM/hw1/image.png"

image = cv2.imread(image_path)

h, w, c = image.shape

print("Image width: ", w)
print("Image height: ", h)
print("Number of channels: ", c)

# 学号：2053932 -> 将图片顺（逆）旋转 32 度
rotated_image1 = rotate_image(image_path, 32)
rotated_image2 = rotate_image(image_path, -32)

cv2.imwrite("rotated_image1.png", rotated_image1)
cv2.imwrite("rotated_image2.png", rotated_image2)
