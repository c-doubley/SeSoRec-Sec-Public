# exp.py
import sys
sys.path.append('../reader')  # 添加TrustGetter类所在目录到模块搜索路径
import numpy as np
from reader.trust import TrustGetter  # 导入TrustGetter类
from reader.epinions import EpinionsGetter
from reader.douban import DoubanGetter
from scipy.sparse import coo_matrix, csr_matrix, vstack, lil_matrix, find
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



    def calculate_similarity_sparse(self, matrix_A, matrix_B):
        assert matrix_A.shape == matrix_B.shape, "两个矩阵的维度必须相同"
        
        # 计算总元素数
        total_elements = matrix_A.shape[0] * matrix_A.shape[1]
        
        # 获取A和B的非零元素位置和值
        A_rows, A_cols, A_data = find(matrix_A)
        B_rows, B_cols, B_data = find(matrix_B)

        # 计算A中为0的元素个数及其占比
        A_zero_count = total_elements - len(A_data)
        percentage_A_zero = (A_zero_count / total_elements) * 100

        # 计算A和B非零且相等的元素个数
        match_nonzero_count = sum(1 for i, j, v in zip(A_rows, A_cols, A_data) if matrix_B[i, j] == v)
        percentage_nonzero_match = (match_nonzero_count / len(A_data)) * 100 if len(A_data) > 0 else 0

        # 计算A和B所有元素中相等的比例
        all_match_count = sum(1 for i, j, v in zip(A_rows, A_cols, A_data) if matrix_B[i, j] == v) + sum(1 for i, j, v in zip(B_rows, B_cols, B_data) if matrix_A[i, j] == v and matrix_A[i, j] == 0)
        percentage_all_match = (all_match_count / total_elements) * 100

        return percentage_A_zero, percentage_nonzero_match, percentage_all_match



    def restore_original_matrix_optimized(self, matrix_A):
        # 将输入矩阵转换为COO格式以便于处理非零元素
        matrix_A_coo = matrix_A.tocoo()
        rows_A, cols_A = matrix_A.shape
        
        # 准备新的行索引、列索引和数据数组
        rows_B = []
        cols_B = []
        data_B = []
        
        for i, j, v in zip(matrix_A_coo.row, matrix_A_coo.col, matrix_A_coo.data):
            if v == 0:
                continue  # 对于v为0的情况，已经默认为0，无需添加
            elif v == 1:
                rows_B.extend([2 * i + 1])
                cols_B.extend([j])
                data_B.extend([1])
            elif v == -1:
                rows_B.extend([2 * i])
                cols_B.extend([j])
                data_B.extend([1])
            elif v == 2:
                rows_B.extend([2 * i, 2 * i + 1])
                cols_B.extend([j, j])
                data_B.extend([-1, 1])
            elif v == -2:  # 假设v为-2表示A中元素为1时的第二种情况
                rows_B.extend([2 * i, 2 * i + 1])
                cols_B.extend([j, j])
                data_B.extend([1, -1])
        
        # 使用新的行索引、列索引和数据数组创建新的COO矩阵
        matrix_B_coo = coo_matrix((data_B, (rows_B, cols_B)), shape=(2 * rows_A, cols_A))

        # self.print_matrix_front(matrix_B_coo)
        
        # 最终转换为CSR格式以获得最佳性能
        return matrix_B_coo.tocsr()


    # 看泄露了S_B的多少数据
    def leak_SB(self, flag):
        if flag == "TrustGetter":
            tg = TrustGetter(self.trust_path)
        elif flag == "EpinionsGetter":
            tg = EpinionsGetter(self.trust_path)
        else:
            tg = DoubanGetter(self.trust_path)

        
        
        # 调用get_relations方法并获取矩阵
        matrix = tg.relation_matrix

        # 输出稀疏矩阵的维度
        print("稀疏矩阵的维度为:", matrix.shape)

        # matrix1 = np.array([[0, 0, 0, 0],
        #            [-1, 1, 0, 1],
        #            [1, 0, -1, 1],
        #            [1, 1, 1, 1]])

        matrix = csr_matrix(matrix)

        # 检查矩阵是否为偶数行，如果不是，则添加一个全零行
        if matrix.shape[0] % 2 != 0:
            zero_row = csr_matrix(np.zeros((1, matrix.shape[1])))  # 创建一个与矩阵列数相同的全零行
            matrix = vstack([matrix, zero_row])  # 将全零行添加到矩阵末尾
        
        rows, cols = matrix.shape
        
        diff_rows = []
        
        for i in range(0, rows-1, 2):
            diff = matrix[i+1, :] - matrix[i, :]
            diff_rows.append(diff)
        
        leak_matrix = vstack(diff_rows)
        restore_matrix = self.restore_original_matrix_optimized(leak_matrix)
        similarity,c,d = self.calculate_similarity_sparse(matrix, restore_matrix)
        print("A中为0的元素个数及其占比： ")
        print(similarity)
        print("A和B非零且相等的元素占比： ")
        print(c)
        print("A和B所有元素中相等的比例： ")
        print(d)
        return similarity
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
    # # message = f"ft_trust数据集中泄露了{percentage_tg}%的数据"
    # # print(message)

    # exp_eg = Exp('./data/epinions.txt')  
    # percentage_eg = exp_eg.leak_SB("EpinionsGetter")
    # # message = f"epinions数据集中泄露了{percentage_eg}%的数据"
    # # print(message)

    exp_dg = Exp('./data/out.douban')   
    percentage_dg = exp_dg.leak_SB("DoubanGetter")

