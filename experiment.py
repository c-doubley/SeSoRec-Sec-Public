# exp.py
import sys
sys.path.append('../reader')  # 添加TrustGetter类所在目录到模块搜索路径
import numpy as np
from reader.trust import TrustGetter  # 导入TrustGetter类
from scipy.sparse import csr_matrix, vstack, lil_matrix

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

    def leak_SB(self):
        # 创建TrustGetter实例
        # tg = TrustGetter(self.trust_path)
        
        # # 调用get_relations方法并获取矩阵
        # matrix = tg.get_relations()

        matrix1 = np.array([[1, 0, 1, 0],
                   [1, 1, 0, 1],
                   [1, 0, 0, 1],
                   [1, 1, 1, 1]])

        matrix = csr_matrix(matrix1)
        
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

        self.print_matrix_front(leak_matrix)

        restore_matrix = self.restore_original_matrix(leak_matrix)
        self.print_matrix_front(restore_matrix)


# 使用示例
if __name__ == '__main__':
    exp = Exp('./data/ft_trust.txt')  # 假设ft_trust.txt位于data目录
    exp.leak_SB()
