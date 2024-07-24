"""
文件名称: experiment5.py

描述:
    这个文件要完成SeSoRec中关于FilmTrust、Epinions、Douban这三个数据集涉及到的实验。
    1. 生成相关三元组（使用可信第三方，自己生成模拟？）***
    2. 从3个数据集中根据内容将有向图/无向图的边权信息转换为稀疏矩阵
    3. 生成需要交互的矩阵信息即 B1 = B + F1, 以及发送 E1 ***
    4. 模拟社交平台的泄露信息，之前是 B_even - B_odd = F1 - (B1_even - B1_odd)
                             替换成                  （E1_even - E1_odd）- (B1_even - B1_odd) ***
    5. 根据泄露信息重建矩阵 (试一下) 
    6. 把原始矩阵 和 重建矩阵 中非零的数据作比较，看还原出多少非零信息


功能:
    - calculate_similarity_sparse: 输入原始矩阵 和 重建矩阵，比较有多少非零元素是一样的
    - restore_original_matrix_optimized： 根据泄露信息重建矩阵，只重建了非零的部分（稀疏矩阵太大了否则计算量非常大）
    - leak_SB: 实现完整流程，先读取数据，然后构造泄露信息，根据泄露信息重建矩阵最后比较相似的数据有多少

用法:
    python experiment3.py

作者: chenyuyue
日期: 2024/4/28
"""
# exp.py
import sys
sys.path.append('/home/cyy/PPSR/ECAI2024')
import numpy as np

from reader.trust import TrustGetter  # 导入TrustGetter类
from reader.epinions import EpinionsGetter
from reader.douban import DoubanGetter
from scipy.sparse import coo_matrix, csr_matrix, vstack, lil_matrix, find, random as sparse_random
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



    def Matrix_Triple(self, row, col):
        # Generate random sparse matrices
        E = sparse_random(row, row, density=0.1, format='coo', data_rvs=np.random.randn)
        # R_E = sparse_random(row, row, density=0.1, format='coo', data_rvs=np.random.randn)
        F = sparse_random(row, col, density=0.1, format='coo', data_rvs=np.random.randn)
        R_F = sparse_random(row, col, density=0.1, format='coo', data_rvs=np.random.randn)
        R_EF = sparse_random(row, col, density=0.1, format='coo', data_rvs=np.random.randn)
        
        # Convert matrices from COO to CSR format for efficient arithmetic operations
        E = E.tocsr()
        # R_E = R_E.tocsr()
        F = F.tocsr()
        R_F = R_F.tocsr()
        R_EF = R_EF.tocsr()
        
        # Calculate E1, F1, EF1
        F0 = F - R_F
        F1 = R_F
        EF1 = E.dot(F) - R_EF  # Matrix multiplication of E and F, subtract R_EF

        return F0, F1, EF1
        

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

        return percentage_A_zero, percentage_nonzero_match



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

        F0, F1, EF1 = self.Matrix_Triple(rows, cols)

        '''
        原来获得的信息是 B1 = F - B, F1 = F_even - F_odd
        原来获取泄露信息的方法是 B_even - B_odd  = F1 - (B1_even - B1_odd)
        新方案获得的信息是 B1 = B - <F>_1, <E>
        对应的获取泄露信息的方法为 B_even - B_odd  = (<F>_0even - <F>_0odd) - (B1_even - B1_odd)
        '''
        # B1 = B - <F>_1
        MatrixB = matrix - F1
        # <F>_0  后面要 求 <F>_0even - <F>_0odd
        diff_F0 = []
        for i in range(0, rows-1, 2):
            diff = F0[i+1, :] - F0[i, :]
            diff_F0.append(diff)

        # 这里为了求 (B1_even - B1_odd)
        diff_MatrixB = []
        for i in range(0, rows-1, 2):
            diff = MatrixB[i+1, :] - MatrixB[i, :]
            diff_MatrixB.append(diff)

        
        Matrix_F0 = vstack(diff_F0)
        Matrix_B1 = vstack(diff_MatrixB)
        leak_matrix = Matrix_F0  - Matrix_B1

        restore_matrix = self.restore_original_matrix_optimized(leak_matrix)
        similarity,c = self.calculate_similarity_sparse(matrix, restore_matrix)
        print("A中为0的元素个数及其占比： ")
        print(similarity)
        print("A和B非零且相等的元素占比： ")
        print(c)
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
    # exp_tg = Exp('data/ft_trust.txt')   
    # percentage_tg = exp_tg.leak_SB("TrustGetter")
    # message = f"ft_trust数据集中泄露了{percentage_tg}%的数据"
    # print(message)

    # 这个实验应该是做不了，稀疏矩阵乘法太大了需要超大的内存
    exp_eg = Exp('data/epinions.txt')  
    percentage_eg = exp_eg.leak_SB("EpinionsGetter")
    # message = f"epinions数据集中泄露了{percentage_eg}%的数据"
    # print(message)

    # exp_dg = Exp('../data/out.douban')   
    # percentage_dg = exp_dg.leak_SB("DoubanGetter")

