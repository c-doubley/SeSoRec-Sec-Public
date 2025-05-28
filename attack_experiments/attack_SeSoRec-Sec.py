"""
File name: attack_SeSoRec-Sec.py

Description:
    This file implements experiments related to FilmTrust, Epinions, and Douban datasets in SeSoRec.
    1. Generate related triples (using trusted third party, self-generated simulation?) ***
    2. Convert edge weight information from directed/undirected graphs to sparse matrices from 3 datasets
    3. Generate matrix information for interaction: B1 = B + F1, and send E1 ***
    4. Simulate social platform information leakage, previously: B_even - B_odd = F1 - (B1_even - B1_odd)
                              replaced with: (F0_even - F0_odd) - (B1_even - B1_odd) ***
    5. Reconstruct matrix based on leaked information (try it)
    6. Compare non-zero data between original matrix and reconstructed matrix to see how much non-zero information is restored

Functions:
    - calculate_similarity_sparse: Input original matrix and reconstructed matrix, compare how many non-zero elements are the same
    - restore_original_matrix_optimized: Reconstruct matrix based on leaked information, only reconstruct non-zero parts (sparse matrix is too large otherwise computation would be very large)
    - leak_SB: Implement complete process, first read data, then construct leaked information, reconstruct matrix based on leaked information and finally compare how many similar data points exist

Usage:
    python attack_experiments/attack_SeSoRec-Sec.py

Author:
Date: 2024/7/24
"""
# exp.py
import sys
from pathlib import Path
# Get project root directory
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
import numpy as np

from reader.trust import TrustGetter  # Import TrustGetter class
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
        # Ensure matrix is in csr format to support index access
        matrix = matrix.tocsr()
        
        # Get matrix dimensions
        rows_, cols_ = matrix.shape
        
        # Calculate number of rows and columns to output
        output_rows = min(rows, rows_)
        output_cols = min(cols, cols_)
        
        # Convert to dense matrix and output first 30x30 elements
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
        assert matrix_A.shape == matrix_B.shape, "Matrix dimensions must be the same"
        
        # Calculate total number of elements
        total_elements = matrix_A.shape[0] * matrix_A.shape[1]
        
        # Get non-zero element positions and values for A and B
        A_rows, A_cols, A_data = find(matrix_A)
        B_rows, B_cols, B_data = find(matrix_B)

        # Calculate number of zero elements in A and their percentage
        A_zero_count = total_elements - len(A_data)
        percentage_A_zero = (A_zero_count / total_elements) * 100

        # Calculate number of non-zero and equal elements between A and B
        match_nonzero_count = sum(1 for i, j, v in zip(A_rows, A_cols, A_data) if matrix_B[i, j] == v)
        percentage_nonzero_match = (match_nonzero_count / len(A_data)) * 100 if len(A_data) > 0 else 0

        return percentage_A_zero, percentage_nonzero_match

    def restore_original_matrix_optimized(self, matrix_A):
        # Convert input matrix to COO format for processing non-zero elements
        matrix_A_coo = matrix_A.tocoo()
        rows_A, cols_A = matrix_A.shape
        
        # Prepare new row indices, column indices and data arrays
        rows_B = []
        cols_B = []
        data_B = []
        
        for i, j, v in zip(matrix_A_coo.row, matrix_A_coo.col, matrix_A_coo.data):
            if v == 0:
                continue  # For v=0 case, already default to 0, no need to add
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
            elif v == -2:  # Assume v=-2 represents second case when element in A is 1
                rows_B.extend([2 * i, 2 * i + 1])
                cols_B.extend([j, j])
                data_B.extend([1, -1])
        
        # Create new COO matrix using new row indices, column indices and data arrays
        matrix_B_coo = coo_matrix((data_B, (rows_B, cols_B)), shape=(2 * rows_A, cols_A))
        
        # Convert to CSR format for best performance
        return matrix_B_coo.tocsr()

    # Check how much data is leaked from S_B
    def leak_SB(self, flag):
        if flag == "TrustGetter":
            tg = TrustGetter(self.trust_path)
        elif flag == "EpinionsGetter":
            tg = EpinionsGetter(self.trust_path)
        else:
            tg = DoubanGetter(self.trust_path)
        
        # Call get_relations method and get matrix
        matrix = tg.relation_matrix

        # Output sparse matrix dimensions
        print("Sparse matrix dimensions:", matrix.shape)

        matrix = csr_matrix(matrix)

        # Check if matrix has even number of rows, if not add a zero row
        if matrix.shape[0] % 2 != 0:
            zero_row = csr_matrix(np.zeros((1, matrix.shape[1])))  # Create a zero row with same number of columns as matrix
            matrix = vstack([matrix, zero_row])  # Add zero row to end of matrix
        
        rows, cols = matrix.shape

        F0, F1, EF1 = self.Matrix_Triple(rows, cols)

        '''
        Previously obtained information: B1 = F - B, F1 = F_even - F_odd
        Previous method to get leaked information: B_even - B_odd = F1 - (B1_even - B1_odd)
        New scheme obtained information: B1 = B - <F>_1, <E>
        Corresponding method to get leaked information: B_even - B_odd = (<F>_0even - <F>_0odd) - (B1_even - B1_odd)
        '''
        # B1 = B - <F>_1
        MatrixB = matrix - F1
        # <F>_0  later need to calculate <F>_0even - <F>_0odd
        diff_F0 = []
        for i in range(0, rows-1, 2):
            diff = F0[i+1, :] - F0[i, :]
            diff_F0.append(diff)

        # Here to calculate (B1_even - B1_odd)
        diff_MatrixB = []
        for i in range(0, rows-1, 2):
            diff = MatrixB[i+1, :] - MatrixB[i, :]
            diff_MatrixB.append(diff)

        Matrix_F0 = vstack(diff_F0)
        Matrix_B1 = vstack(diff_MatrixB)
        leak_matrix = Matrix_F0 - Matrix_B1

        restore_matrix = self.restore_original_matrix_optimized(leak_matrix)
        similarity,c = self.calculate_similarity_sparse(matrix, restore_matrix)
        print("Number and percentage of zero elements in A: ")
        print(similarity)
        print("Percentage of non-zero and equal elements between A and B: ")
        print(c)
        return similarity

# Usage example
if __name__ == '__main__':
    exp_tg = Exp('data/ft_trust.txt')   
    percentage_tg = exp_tg.leak_SB("TrustGetter")
    # message = f"Percentage of data leaked from ft_trust dataset: {percentage_tg}%"
    # print(message)

    # This experiment cannot be done, sparse matrix multiplication requires too much memory
    # exp_eg = Exp('data/epinions.txt')  
    # percentage_eg = exp_eg.leak_SB("EpinionsGetter")
    # message = f"Percentage of data leaked from epinions dataset: {percentage_eg}%"
    # print(message)

    # exp_dg = Exp('../data/out.douban')   
    # percentage_dg = exp_dg.leak_SB("DoubanGetter")

