"""
文件名称: experiment1.py

描述:
    这个实验和3做类似的事情，但是因为把整个稀疏矩阵都恢复了，后面两个大数据集没法运行，所以优化都实验3上。
    保留参考但不使用。


功能:


用法:
    python experiment3.py

作者: chenyuyue
日期: 2024/4/28
"""
# exp.py
import sys
sys.path.append('../reader')  # 添加TrustGetter类所在目录到模块搜索路径
import numpy as np
from reader.trust import TrustGetter  # 导入TrustGetter类
from reader.epinions import EpinionsGetter
from scipy.sparse import csr_matrix, vstack, lil_matrix
import matplotlib.pyplot as plt
from util.painting import MatrixPainter
import random

class Exp:
    def __init__(self, trust_path):
        self.trust_path = trust_path
    
    def print_matrix_front(self, matrix, rows=40, cols=40):
        # 确保矩阵是csr格式以支持下标访问
        matrix = matrix.tocsr()
        
        # 获取矩阵的行数和列数
        rows_, cols_ = matrix.shape
        
        # 计算要输出的行数和列数
        output_rows = min(rows, rows_)
        output_cols = min(cols, cols_)
        
        # 转换为密集矩阵并输出前30x30的元素
        dense_matrix = matrix[:output_rows, :output_cols].toarray()
        print(dense_matrix)

    def calculate_similarity_percentage(self, matrix_A, matrix_B):
        """
        计算两个稀疏矩阵中相同元素的百分比。

        参数:
        - matrix_A: 第一个稀疏矩阵 (csr_matrix)
        - matrix_B: 第二个稀疏矩阵 (csr_matrix)

        返回:
        - 相同元素的百分比 (float)
        """
        # 确保两个矩阵的形状相同
        assert matrix_A.shape == matrix_B.shape, "两个矩阵的维度必须相同"
        
        # 转换成密集形式再比较
        A_dense = matrix_A.toarray()
        B_dense = matrix_B.toarray()
        
        # 计算相同元素的数量
        same_elements_count = np.sum(A_dense == B_dense)
        
        # 计算总元素数
        total_elements = matrix_A.shape[0] * matrix_A.shape[1]
        
        # 计算相同元素的百分比
        percentage_same = (same_elements_count / total_elements) * 100
        
        return percentage_same



    # 根据偶数行-奇数行的矩阵还原原始矩阵
    def restore_original_matrix(self, matrix_A):
        rows_A, cols_A = matrix_A.shape
        matrix_B = lil_matrix((2 * rows_A, cols_A), dtype=matrix_A.dtype)  # 所有元素默认为0

        for i in range(rows_A):
            for j in range(cols_A):
                Aij = matrix_A[i, j]
                if Aij > 0:
                    # 只需要设置B中非零的元素
                    matrix_B[2*i + 1, j] = Aij
                elif Aij < 0:
                    # 只需要设置B中非零的元素
                    matrix_B[2*i, j] = -Aij
                # 对于Aij == 0的情况，不需要任何操作，因为B的元素已默认为0

        return matrix_B.tocsr()

    # 看泄露了S_B的多少数据
    def leak_SB(self, flag):
        if flag == "TrustGetter":
            tg = TrustGetter(self.trust_path)
        elif flag == "EpinionsGetter":
            tg = EpinionsGetter(self.trust_path)
        else:
            # 当flag既不等于"TrustGetter"也不等于"EpinionsGetter"时执行这里的代码
            print("flag既不是'TrustGetter'也不是'EpinionsGetter'。")
        # 创建TrustGetter实例
        
        
        # 调用get_relations方法并获取矩阵
        matrix = tg.get_relations()

        # matrix1 = np.array([[1, 0, 1, 0],
        #            [1, 1, 0, 1],
        #            [1, 0, 0, 1],
        #            [1, 1, 1, 1]])

        matrix = csr_matrix(matrix)
        
        matrix = matrix.tocsr()
    
        rows, cols = matrix.shape
        
        # 检查行数是否为偶数，如果不是，则添加一个零行
        if rows % 2 != 0:
            zero_row = csr_matrix((1, cols), dtype=matrix.dtype)
            matrix = vstack([matrix, zero_row])
        
        # 使用列表来存储计算结果的行
        result_rows = []
        
        # 偶数行减去奇数行
        for i in range(0, rows, 2):
            diff = matrix[i + 1] - matrix[i]
            result_rows.append(diff)
        
        # 将结果行堆叠为一个新的csr_matrix
        leak_matrix = vstack(result_rows)

        # self.print_matrix_front(leak_matrix)

        restore_matrix = self.restore_original_matrix(leak_matrix)
        # self.print_matrix_front(restore_matrix)

        simi = self.calculate_similarity_percentage(matrix, restore_matrix)
        return simi
        # print("两个矩阵一样的元素有： ")
        # print(simi)

        # random_row = random.randint(0, rows - 50)
        # random_col = random.randint(0, cols - 50)
        # # 创建MatrixPainter实例
        # painter = MatrixPainter(matrix, restore_matrix)

        # # 绘制矩阵比较图
        # painter.paint_comparison(random_row, random_col, 50)






# 使用示例
if __name__ == '__main__':
    # exp_tg = Exp('./data/ft_trust.txt')   
    # percentage_tg = exp_tg.leak_SB("TrustGetter")
    # message = f"ft_trust数据集中泄露了{percentage_tg}%的数据"
    # print(message)

    exp_eg = Exp('./data/epinions.txt')  
    percentage_eg = exp_eg.leak_SB("EpinionsGetter")
    message = f"epinions数据集中泄露了{percentage_eg}%的数据"
    print(message)

