import numpy as np
from PIL import Image
import os
import random
from reader.minist import MNISTLoader
from util.painting import MatrixPainter

class PictureProcessor:
    def __init__(self, matrix):
        self.matrix = matrix

    def process_matrices(self):
        # matrix = self._add_column_if_needed(self.matrix)
        leak_left = self._generate_leak_left(matrix)
        restore_left = self._generate_matrix_C(leak_left)
        leak_right = self._generate_matrix_D(matrix)
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

    loader = MNISTLoader("./data/minist")

    tuple_matrices1 = []
    tuple_matrices2 = []
    for i in range(5):
        list_matrices = loader.load_random_images(i)        
        for matrix in list_matrices:
            processor = PictureProcessor(matrix)
            leak_left, restore_left, leak_right, restore_right = processor.process_matrices()
            tuple_matrices1.append((matrix, restore_left, restore_right))
    for i in range(5, 10):
        list_matrices = loader.load_random_images(i)        
        for matrix in list_matrices:
            processor = PictureProcessor(matrix)
            leak_left, restore_left, leak_right, restore_right = processor.process_matrices()
            tuple_matrices2.append((matrix, restore_left, restore_right))

    
    painter = MatrixPainter()
    # 绘制矩阵比较图
    painter.stack_images(tuple_matrices1, tuple_matrices2)

    # tuple_matrices = []
    # for i in range(10):
    #     list_matrices = loader.load_random_images(i)        
    #     for matrix in list_matrices:
    #         processor = PictureProcessor(matrix)
    #         leak_left, restore_left, leak_right, restore_right = processor.process_matrices()
    #         tuple_matrices.append((matrix, restore_left, restore_right))

    
    # painter = MatrixPainter()
    # # 绘制矩阵比较图
    # painter.stack_images(tuple_matrices)



    # # 随机加载一张标签为0的图片并将其转换为矩阵
    # loader = MNISTLoader("./data/minist")
    # matrix = loader.load_random_image(8)
    
    # # 确保picture目录存在
    # os.makedirs('picture', exist_ok=True)
    
    # # 保存选取的图片到picture目录
    # save_image(matrix, 'picture/original_image.png')
    
    # # 处理矩阵并生成矩阵leak_left和C
    # processor = PictureProcessor(matrix)
    # leak_left, restore_left, leak_right, restore_right = processor.process_matrices()
    
    # # 保存矩阵C到picture目录
    # save_image(restore_left, 'picture/processed_imageC.png')
    # save_image(restore_right, 'picture/processed_imageE.png')

    # painter = MatrixPainter()
    # # 绘制矩阵比较图
    # painter.minist_painting(matrix, restore_left, restore_right)

    # # 输出结果矩阵leak_left和C的形状以验证
    # print("Matrix leak_left shape:", leak_left.shape)
    # print("Matrix restore_left shape:", restore_left.shape)
    # print("Matrix leak_right shape:", leak_right.shape)
    # print("Matrix restore_right shape:", restore_right.shape)
