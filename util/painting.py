# painting.py
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

class MatrixPainter:
    def __init__(self, matrix_A, matrix_B):
        """
        初始化MatrixPainter类。

        参数:
        - matrix_A: 第一个稀疏矩阵 (csr_matrix)
        - matrix_B: 第二个稀疏矩阵 (csr_matrix)
        """
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B

    def paint_comparison(self, row, col, num, file_path='picture/comparison.png'):
        """
        绘制两个矩阵的比较图，并将结果保存为PNG文件。

        参数:
        - file_path: 保存图片的路径和文件名 (默认为'picture/comparison.png')
        """
        # 转换为密集格式并仅考虑前10x10的元素
        dense_A = self.matrix_A.toarray()[row:row+num, col:col+num]
        dense_B = self.matrix_B.toarray()[row:row+num, col:col+num]

        # 计算两个矩阵中相同元素的位置
        mask = dense_A == dense_B

        # 创建图形和轴对象
        fig, ax = plt.subplots()

        # 画出矩阵的每一个元素
        for i in range(dense_A.shape[0]):
            for j in range(dense_A.shape[1]):
                # 使用红色或白色背景
                color = 'red' if dense_A[i, j] == dense_B[i, j] else 'white'
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))


        # 设置坐标轴的范围
        ax.set_xlim(0, num)
        ax.set_ylim(0, num)

        # 移除刻度线
        ax.tick_params(
            axis='both',       # 应用于x轴和y轴的改变
            which='both',      # 应用于主刻度线和次刻度线的改变
            bottom=False,      # 移除x轴底部的刻度线
            top=False,         # 移除x轴顶部的刻度线
            left=False,        # 移除y轴左侧的刻度线
            right=False,       # 移除y轴右侧的刻度线
            labelbottom=False, # 移除x轴底部的刻度标签
            labelleft=False    # 移除y轴左侧的刻度标签
        )

        # 绘制外围边框线
        for spine in ax.spines.values():
            spine.set_visible(True)

        # 保留内部网格线
        for i in range(1, num):
            ax.axhline(i, color='black', linestyle='-', linewidth=2)
            ax.axvline(i, color='black', linestyle='-', linewidth=2)


        # 翻转y轴，以便第一行在顶部
        ax.invert_yaxis()

        # 不显示图形，而是保存到文件
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()


# 使用示例
if __name__ == '__main__':

    # 创建示例稀疏矩阵A和B
    A = csr_matrix(np.random.randint(0, 2, size=(50, 50)))
    B = csr_matrix(np.random.randint(0, 2, size=(50, 50)))
    # 创建MatrixPainter实例
    painter = MatrixPainter(A, B)

    # 绘制矩阵比较图
    painter.paint_comparison(30,30)
