"""
文件名称: cifar.py

描述:
    实现从cifar数据集中读取灰度图像并转换成矩阵形式存储.
    但是由于复现效果不太好最后没有使用数据


功能:
    - load_random_image_as_grayscale: 读取对应标签的图片转换成灰度矩阵

用法:
    python cifar.py

作者: chenyuyue
日期: 2024/4/28
"""
import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt

class CIFARLoader:
    def __init__(self, directory="./data/CIFAR-10"):
        self.directory = directory

    def load_random_image_as_grayscale(self, label):
        """
        从指定目录随机加载一个指定标签的图片，并将其转换为灰度矩阵。
        """
        image_files = [f for f in os.listdir(self.directory) if f.endswith('.png') and f.split('_')[1].split('.')[0] == str(label)]
        if not image_files:
            print(f"No images found for label {label}.")
            return None
        random_image_file = random.choice(image_files)
        image_path = os.path.join(self.directory, random_image_file)
        image = Image.open(image_path)
        image_matrix = np.array(image.convert('L'))  # 使用'L'参数将图像转换为灰度
        return image_matrix

# 示例使用
if __name__ == "__main__":
    loader = CIFARLoader("./data/CIFAR-10")  # 确保路径正确
    labels = list(range(10))  # CIFAR-10 数据集的标签为 0-9
    label = random.choice(labels)  # 随机选择一个标签
    image_matrix = loader.load_random_image_as_grayscale(label)  # 加载灰度图像

    if image_matrix is not None:
        # 展示加载的图片矩阵的形状，以验证加载成功
        print("Loaded grayscale image matrix shape:", image_matrix)

        # 展示图片
        plt.imshow(image_matrix, cmap='gray')  # 使用灰度颜色映射
        plt.title(f"Label: {label}")
        plt.show()
    else:
        print("Failed to load an image.")
