"""
文件名称: experiment2.py

描述:
    这个文件要完成MNIST部分的实验。分为四步
    1. 从MNIST数据集中选取灰度图像，转换为矩阵
    2. 生成泄露信息，即矩阵的奇数列+偶数列  偶数行-奇数行
    3. 根据2种泄露信息重建2个矩阵
    4. 把原始矩阵+2个重建矩阵画成图像


功能:
    - process_matrices: 根据灰度图像矩阵生成2种泄露信息，并根据这2种泄露信息重建矩阵

用法:
    python experiment2.py

作者: chenyuyue
日期: 2024/4/28
"""
import sys
sys.path.append("/home/cyy/PPSR/ECAI2024")
import numpy as np
from PIL import Image
import os
import random
from reader.mnist import MNISTLoader
from util.painting import MatrixPainter

class PictureProcessor:
    def __init__(self, matrix):
        self.matrix = matrix

    def process_matrices(self):
        # matrix = self._add_column_if_needed(self.matrix)
        # 生成左边参与方的泄露信息 奇数列 + 偶数列
        leak_left = self._generate_leak_left(self.matrix)
        # 根据 奇数列 + 偶数列 重建矩阵
        restore_left = self._generate_matrix_C(leak_left)
        # 生成右边参与方的泄露信息 偶数行 - 奇数行
        leak_right = self._generate_matrix_D(self.matrix)
        # 根据 偶数行 - 奇数行 重建矩阵
        restore_right = self._generate_matrix_E(leak_right)
        return leak_left, restore_left, leak_right, restore_right

    def _add_column_if_needed(self, matrix):
        # 如果列数不为偶数，则在最右侧添加一个全为0的列
        if matrix.shape[1] % 2 != 0:
            matrix = np.hstack((matrix, np.zeros((matrix.shape[0], 1), dtype=matrix.dtype)))
        return matrix

    def _generate_leak_left(self, matrix):
        # 将矩阵的偶数列和奇数列相加得到新的矩阵leak_left
        leak_left = np.zeros((matrix.shape[0], matrix.shape[1] // 2), dtype=matrix.dtype)
        for j in range(leak_left.shape[1]):
            leak_left[:, j] = matrix[:, 2*j] + matrix[:, 2*j+1]
        return leak_left

    def _generate_matrix_C(self, leak_left):
        # 根据矩阵leak_left生成新的矩阵C
        restore_left = np.zeros((leak_left.shape[0], leak_left.shape[1] * 2), dtype=leak_left.dtype)
        for i in range(leak_left.shape[0]):
            for j in range(leak_left.shape[1]):
                if leak_left[i, j] != 0:
                    restore_left[i, 2*j+1] = leak_left[i, j]
        return restore_left

    def _add_row_if_needed(self, matrix):
        # 如果行数不为偶数，则在最下面添加一个全为0的行
        if matrix.shape[0] % 2 != 0:
            matrix = np.vstack((matrix, np.zeros((1, matrix.shape[1]), dtype=matrix.dtype)))
        return matrix

    def _generate_matrix_D(self, matrix):
        # 将矩阵的偶数行和奇数行相减得到新的矩阵D
        leak_right = np.zeros((matrix.shape[0] // 2, matrix.shape[1]), dtype=matrix.dtype)
        for i in range(leak_right.shape[0]):
            leak_right[i, :] = matrix[2*i, :] - matrix[2*i + 1, :]
        return leak_right

    def _generate_matrix_E(self, leak_right):
        # 根据矩阵D生成新的矩阵E
        restore_right = np.zeros((leak_right.shape[0] * 2, leak_right.shape[1]), dtype=leak_right.dtype)
        for i in range(leak_right.shape[0]):
            for j in range(leak_right.shape[1]):
                if leak_right[i, j] == 0:
                    restore_right[2*i, j] = restore_right[2*i + 1, j] = 0
                elif leak_right[i, j] < 0:
                    restore_right[2*i, j] = -leak_right[i, j] * 2 
                    restore_right[2*i + 1, j] = 0
                else:
                    restore_right[2*i, j] = 0
                    restore_right[2*i + 1, j] = leak_right[i, j] * 2
        return restore_right

def save_image(matrix, file_path):
    """
    将矩阵保存为PNG格式的灰度图像。
    """
    # 将矩阵转换为图像
    image = Image.fromarray(matrix.astype(np.uint8), 'L')
    # 保存图像
    image.save(file_path)

# 示例使用
if __name__ == "__main__":

    # 确保picture目录存在
    os.makedirs('picture', exist_ok=True)

    loader = MNISTLoader("data/minist")

    tuple_matrices1 = []
    tuple_matrices2 = []
    # 读取标签0-4的灰度图像转换成矩阵
    for i in range(5):
        list_matrices = loader.load_random_images(i)        
        for matrix in list_matrices:
            processor = PictureProcessor(matrix)
            leak_left, restore_left, leak_right, restore_right = processor.process_matrices()
            tuple_matrices1.append((matrix, restore_left, restore_right))
    # 读取标签5-9的灰度图像转换成矩阵
    for i in range(5, 10):
        list_matrices = loader.load_random_images(i)        
        for matrix in list_matrices:
            processor = PictureProcessor(matrix)
            leak_left, restore_left, leak_right, restore_right = processor.process_matrices()
            tuple_matrices2.append((matrix, restore_left, restore_right))

    
    painter = MatrixPainter()
    # 绘制矩阵比较图
    painter.mnist_stack_images(tuple_matrices1, tuple_matrices2)


    # 这里提供一个只画一张图的使用示例
    # 随机加载一张标签的图片并将其转换为矩阵
    loader = MNISTLoader("data/minist")
    matrix = loader.load_random_image(0)
    
    # 确保picture目录存在
    os.makedirs('picture', exist_ok=True)
    # 保存选取的图片到picture目录
    save_image(matrix, 'picture/mnist_original_image.png')
    
    # 处理矩阵并生成矩阵leak_left和C
    processor = PictureProcessor(matrix)
    leak_left, restore_left, leak_right, restore_right = processor.process_matrices()
    
    # 保存矩阵C到picture目录
    save_image(restore_left, 'picture/mnist_leak_left.png')
    save_image(restore_right, 'picture/mnist_leak_right.png')

    painter = MatrixPainter()
    # 绘制矩阵比较图
    painter.minist_painting(matrix, restore_left, restore_right)

