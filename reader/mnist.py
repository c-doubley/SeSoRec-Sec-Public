"""
文件名称: mnist.py

描述:
    实现从mnist数据集中读取灰度图像并转换成矩阵形式存储


功能:
    - load_random_image: 从指定目录随机加载一个指定标签的图片，并将其转换为矩阵。
    - load_random_images(self, label, num_images=5):从指定目录随机加载指定数量的指定标签的图片，并将其转换为矩阵。

用法:
    python mnist.py

作者: chenyuyue
日期: 2024/4/28
"""

import numpy as np
from PIL import Image
import os
import random

class MNISTLoader:
    def __init__(self, directory="./data/minist"):
        self.directory = directory

    def load_random_image(self, label):
        """
        从指定目录随机加载一个指定标签的图片，并将其转换为矩阵。
        """
        # 构建指定标签的图片文件路径列表
        image_files = [f for f in os.listdir(self.directory) if f.endswith('.png') and f.startswith(str(label))]
        
        # 随机选择一个图片文件
        random_image_file = random.choice(image_files)
        image_path = os.path.join(self.directory, random_image_file)
        
        # 使用Pillow库读取图片
        image = Image.open(image_path)
        
        # 将图片转换为灰度模式，然后转换为NumPy矩阵
        image_matrix = np.array(image.convert('L'))
        
        return image_matrix

    def load_random_images(self, label, num_images=5):
        """
        从指定目录随机加载指定数量的指定标签的图片，并将其转换为矩阵。
        """
        # 构建指定标签的图片文件路径列表
        image_files = [f for f in os.listdir(self.directory) if f.endswith('.png') and f.startswith(str(label))]
        
        # 如果图片数量不足要求的数量，则返回空列表
        if len(image_files) < num_images:
            return []

        # 随机选择指定数量的图片文件
        random_image_files = random.sample(image_files, num_images)

        # 存储图片矩阵的列表
        image_matrices = []

        for random_image_file in random_image_files:
            image_path = os.path.join(self.directory, random_image_file)
            
            # 使用Pillow库读取图片
            image = Image.open(image_path)
            
            # 将图片转换为灰度模式，然后转换为NumPy矩阵
            image_matrix = np.array(image.convert('L'))
            
            # 将图片矩阵添加到列表中
            image_matrices.append(image_matrix)
        
        return image_matrices

# 示例使用
if __name__ == "__main__":
    loader = MNISTLoader("./data/minist")  # 如果存放MNIST数据的目录不同，请调整路径
    label = 0  # 指定想要加载的图片的标签
    image_matrix = loader.load_random_image(label)
    
    # 展示加载的图片矩阵的形状，以验证加载成功
    print("Loaded image matrix shape:", image_matrix.shape)
    
    # 如果需要，可以使用matplotlib展示图片
    # import matplotlib.pyplot as plt
    # plt.imshow(image_matrix, cmap='gray')
    # plt.show()

