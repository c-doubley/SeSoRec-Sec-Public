# encoding:utf-8
import sys

sys.path.append("..")
import numpy as np
import os
from scipy.sparse import dok_matrix, csr_matrix

# from configx.configx import ConfigX


class TrustGetter(object):
    """
    docstring for TrustGetter
    read trust data and save the global parameters

    """

    def __init__(self, trust_path):
        self.trust_path = trust_path
        self.sep = ' '
        self.get_relations()



    def get_relations(self):
        if not os.path.isfile(self.trust_path):
            print("the format of trust data is wrong")
            sys.exit()

        # 第一遍扫描，找出最大的用户ID
        max_id = 0
        with open(self.trust_path, 'r') as f:
            for line in f:
                u_from, u_to, _ = line.strip('\r\n').split(self.sep)
                max_id = max(max_id, int(u_from), int(u_to))
        
        # 创建一个稀疏矩阵
        # 注意：如果用户ID是从1开始的，可能需要使用max_id + 1
        relation_matrix = dok_matrix((max_id + 1, max_id + 1), dtype=np.int32)
        
        # 第二遍扫描，填充矩阵
        with open(self.trust_path, 'r') as f:
            for line in f:
                u_from, u_to, t = line.strip('\r\n').split(self.sep)
                relation_matrix[int(u_from), int(u_to)] = int(t)
        
        # print(relation_matrix[0:10, 0:10])
        # # 选取前10x10的部分
        # sub_matrix = relation_matrix[1521, 1075]

        # # 将选取的部分转换为CSR格式，这通常是转换为密集格式前的推荐步骤
        # sub_matrix_csr = csr_matrix(sub_matrix)

        # # 将稀疏矩阵的这一部分转换为密集格式
        # sub_matrix_dense = sub_matrix_csr.toarray()

        # # 打印密集格式的矩阵
        # print(sub_matrix_dense)
        
        return relation_matrix



if __name__ == '__main__':
    tg = TrustGetter('./data/ft_trust.txt')

