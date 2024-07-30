"""
文件名称: experiment6.py

描述:
    这个文件要完成MNIST部分的实验。分为四步
    1. 从MNIST数据集中选取灰度图像，转换为矩阵
    2. 生成泄露信息，即矩阵的奇数列+偶数列  偶数行-奇数行
       奇数列+偶数列 之前是  A_even + B_odd = A0_even + A0_odd - E0
       替换成                                (A0_even + A0_odd) - (E1_even + E1_odd)

       偶数行-奇数行 之前是  B_even - B_odd = F1 - (B1_even - B1_odd)
       替换成                 （F0_even - F0_odd）- (B1_even - B1_odd) ***

    3. 根据2种泄露信息重建2个矩阵
    4. 把原始矩阵+2个重建矩阵画成图像


功能:
    - process_matrices: 根据灰度图像矩阵生成2种泄露信息，并根据这2种泄露信息重建矩阵

用法:
    python experiment5.py

作者: chenyuyue
日期: 2024/7/25
"""
import sys
sys.path.append("/home/cyy/PPSR/ECAI2024")
import numpy as np
from PIL import Image
import os
import random
from reader.mnist import MNISTLoader
from util.painting import MatrixPainter
from scipy.sparse import vstack, hstack, random as sparse_random

class PictureProcessor:
    def __init__(self, matrix):
        self.matrix = matrix
        


    def Matrix_Triple(self, row, col):
        def random_integers(low, high, size):
            return np.random.randint(low, high, size, dtype=np.uint8)

        # 创建随机的 NumPy 矩阵
        E = random_integers(0, 255, (row, row))
        R_E = random_integers(0, 255, (row, row))
        F = random_integers(0, 255, (row, col))
        R_F = random_integers(0, 255, (row, col))
        
        # Calculate E1, F1, EF1
        E0 = E - R_E
        E1 = R_E
        F0 = F - R_F
        F1 = R_F

        
        return E0, E1, F0, F1

    def process_matrices(self):
        # matrix = self._add_column_if_needed(self.matrix)
        E0, E1, F0, F1 = self.Matrix_Triple(28, 28)

        # 生成左边参与方的泄露信息 对应原来的奇数列 + 偶数列
        leak_left = self._generate_leak_left(self.matrix, E0, E1)
        # 根据 奇数列 + 偶数列 重建矩阵
        restore_left = self._generate_matrix_C(leak_left)

        # 生成右边参与方的泄露信息 对应原来的偶数行 - 奇数行
        leak_right = self._generate_matrix_right(self.matrix, F0, F1)
        # 根据 偶数行 - 奇数行 重建矩阵
        restore_right = self._generate_matrix_E(leak_right)
        return leak_left, restore_left, leak_right, restore_right

    def _add_column_if_needed(self, matrix):
        # 如果列数不为偶数，则在最右侧添加一个全为0的列
        if matrix.shape[1] % 2 != 0:
            matrix = np.hstack((matrix, np.zeros((matrix.shape[0], 1), dtype=matrix.dtype)))
        return matrix

    def _generate_leak_left(self, matrix, matrixE0, matrixE1):
        #  leak_left = (A0_even + A0_odd) - (E1_even + E1_odd)
        
        # A0 = A - <E>_0
        # 将稀疏矩阵转换为密集矩阵
        A0 = matrix - matrixE0

           # 计算偶数行和奇数行的和
        def sum_even_odd(matrix):
            if len(matrix.shape) == 2:
                even_rows = matrix[:, ::2]
                odd_rows = matrix[:, 1::2]
                return even_rows + odd_rows
            else:
                raise ValueError("Matrix must be a 2-dimensional array")

        # 计算 A 和 E1_
        A = sum_even_odd(A0)
        E1_ = sum_even_odd(matrixE1)

        # 计算 leak_left
        leak_left = A - E1_

        # <E>_1  后面要 求 E1_even + E1_odd       
        # diff_E1 = []
        # for i in range(0, cols-1, 2):
        #     diff = matrixE1[:, i+1] + matrixE1[:, i]
        #     diff_E1.append(diff)

        # 这里为了求 (A0_even + A0_odd)
        # diff_MatrixA = []
        # for i in range(0, cols-1, 2):
        #     diff = MatrixA[:, i+1] + MatrixA[:, i]
        #     diff_MatrixA.append(diff)

        # # 将列表转换为 NumPy 数组
        # diff_MatrixA = np.array(diff_MatrixA)
        # diff_E1 = np.array(diff_E1)

        # print(diff_MatrixA.ndim)  # 这应该输出 2

        # E1 = vstack(diff_E1)
        # A0 = vstack(diff_MatrixA)
        # leak_left = A0_  - E1_

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

    def _generate_matrix_right(self, matrix, MatrixF0, MatrixF1):
        # 将矩阵的偶数行和奇数行相减得到新的矩阵D

        B1 = matrix - MatrixF1

           # 计算偶数列和奇数列的差
        def sub_even_odd(matrix):
            if len(matrix.shape) == 2:
                even_cols = matrix[0::2, :]
                odd_cols = matrix[1::2, :]
                return even_cols - odd_cols
            else:
                raise ValueError("Matrix must be a 2-dimensional array")

        # 计算 A 和 E1_
        B = sub_even_odd(B1)
        F0_ = sub_even_odd(MatrixF0)

        # 计算 leak_left
        leak_right= B - F0_


        # rows, cols = matrix.shape
        # # B1 = B - <F>_1
        # MatrixB = matrix - MatrixF1
        # # <F>_0  后面要 求 <F>_0even - <F>_0odd
        # diff_F0 = []
        # for i in range(0, rows-1, 2):
        #     diff = MatrixF0[i+1, :] - MatrixF0[i, :]
        #     diff_F0.append(diff)

        # # 这里为了求 (B1_even - B1_odd)
        # diff_MatrixB = []
        # for i in range(0, rows-1, 2):
        #     diff = MatrixB[i+1, :] - MatrixB[i, :]
        #     diff_MatrixB.append(diff)

        
        # F0 = vstack(diff_F0)
        # B1 = vstack(diff_MatrixB)
        # leak_right = F0  - B1
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


def add_255_border(matrix):
    """
    给一个NumPy矩阵在最后一行和最右边添加两个全为255的行和列

    参数:
    matrix (numpy.ndarray): 输入的二维灰度图像矩阵

    返回:
    numpy.ndarray: 添加了255边界后的矩阵
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("输入必须是一个NumPy矩阵")
    
    # 创建一个全为255的行和列
    row_255 = np.full((1, matrix.shape[1]), 255, dtype=np.uint8)
    col_255 = np.full((matrix.shape[0] + 1, 1), 255, dtype=np.uint8)

    # 先添加行，再添加列
    matrix_with_row = np.vstack([matrix, row_255])
    matrix_with_row_and_col = np.hstack([matrix_with_row, col_255])
    
    return matrix_with_row_and_col

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
            tuple_matrices1.append((add_255_border(matrix), add_255_border(restore_left), add_255_border(restore_right)))
    # 读取标签5-9的灰度图像转换成矩阵
    for i in range(5, 10):
        list_matrices = loader.load_random_images(i)        
        for matrix in list_matrices:
            processor = PictureProcessor(matrix)
            leak_left, restore_left, leak_right, restore_right = processor.process_matrices()
            tuple_matrices2.append((add_255_border(matrix), add_255_border(restore_left), add_255_border(restore_right)))
            # tuple_matrices2.append((matrix, restore_left, restore_right))

    
    painter = MatrixPainter()
    # 绘制矩阵比较图
    painter.mnist_stack_images(tuple_matrices1, tuple_matrices2)


    # # 这里提供一个只画一张图的使用示例
    # # 随机加载一张标签的图片并将其转换为矩阵
    # loader = MNISTLoader("../data/minist")
    # matrix = loader.load_random_image(0)
    
    # # 确保picture目录存在
    # os.makedirs('picture', exist_ok=True)
    # # 保存选取的图片到picture目录
    # save_image(matrix, '../picture/mnist_original_image.png')
    
    # # 处理矩阵并生成矩阵leak_left和C
    # processor = PictureProcessor(matrix)
    # leak_left, restore_left, leak_right, restore_right = processor.process_matrices()
    
    # # 保存矩阵C到picture目录
    # save_image(restore_left, '../picture/mnist_leak_left.png')
    # save_image(restore_right, '../picture/mnist_leak_right.png')

    # painter = MatrixPainter()
    # # 绘制矩阵比较图
    # painter.minist_painting(matrix, restore_left, restore_right)

