"""
File Name: attack_SeSoRec.py

Description:
    This file implements experiments related to the FilmTrust, Epinions, and Douban datasets in SeSoRec.
    1. Convert edge weight information of directed/undirected graphs from the three datasets into sparse matrices.
    2. Generate leaked information from the social platform, i.e., even rows minus odd rows of the matrix.
    3. Reconstruct the matrix based on the leaked information.
    4. Compare non-zero data between the original matrix and the reconstructed matrix to see how much non-zero information is recovered.

Functionality:
    - calculate_similarity_sparse: Takes the original matrix and reconstructed matrix as input and compares how many non-zero elements are the same.
    - restore_original_matrix_optimized: Reconstructs the matrix based on leaked information, only reconstructing non-zero parts (sparse matrices are too large, otherwise the computation is very intensive).
    - leak_SB: Implements the complete process: reads data, constructs leaked information, reconstructs the matrix based on leaked information, and compares how much similar data is present.

Usage:
    python attack_experiments/attack_SeSoRec.py

Author: 
Date: 2024/4/28
"""

import sys
from pathlib import Path
# Get project root directory
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
import numpy as np
from reader.trust import TrustGetter  # Import TrustGetter class
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
        # Ensure the matrix is in CSR format to support indexing
        matrix = matrix.tocsr()
        
        # Get the number of rows and columns of the matrix
        rows_, cols_ = matrix.shape
        
        # Calculate the number of rows and columns to output
        output_rows = min(rows, rows_)
        output_cols = min(cols, cols_)
        
        # Convert to dense matrix and output the first 30x30 elements
        dense_matrix = matrix[:output_rows, :output_cols].toarray()
        print(dense_matrix)

    def calculate_similarity_sparse(self, matrix_A, matrix_B):
        assert matrix_A.shape == matrix_B.shape, "The dimensions of the two matrices must be the same"
        
        # Calculate the total number of elements
        total_elements = matrix_A.shape[0] * matrix_A.shape[1]
        
        # Get the positions and values of non-zero elements in A and B
        A_rows, A_cols, A_data = find(matrix_A)
        B_rows, B_cols, B_data = find(matrix_B)

        # Calculate the number and proportion of zero elements in A
        A_zero_count = total_elements - len(A_data)
        percentage_A_zero = (A_zero_count / total_elements) * 100

        # Calculate the number of non-zero elements that are equal in A and B
        match_nonzero_count = sum(1 for i, j, v in zip(A_rows, A_cols, A_data) if matrix_B[i, j] == v)
        percentage_nonzero_match = (match_nonzero_count / len(A_data)) * 100 if len(A_data) > 0 else 0

        return percentage_A_zero, percentage_nonzero_match

    def restore_original_matrix_optimized(self, matrix_A):
        # Convert the input matrix to COO format for easier handling of non-zero elements
        matrix_A_coo = matrix_A.tocoo()
        rows_A, cols_A = matrix_A.shape
        
        # Prepare new row indices, column indices, and data arrays
        rows_B = []
        cols_B = []
        data_B = []
        
        for i, j, v in zip(matrix_A_coo.row, matrix_A_coo.col, matrix_A_coo.data):
            if v == 0:
                continue  # For v equal to 0, it is already defaulted to 0, no need to add
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
            elif v == -2:  # Assume v equals -2 represents the second case when the element in A is 1
                rows_B.extend([2 * i, 2 * i + 1])
                cols_B.extend([j, j])
                data_B.extend([1, -1])
        
        # Create a new COO matrix using the new row indices, column indices, and data arrays
        matrix_B_coo = coo_matrix((data_B, (rows_B, cols_B)), shape=(2 * rows_A, cols_A))

        # self.print_matrix_front(matrix_B_coo)
        
        # Finally, convert to CSR format for optimal performance
        return matrix_B_coo.tocsr()

    # Check how much data of S_B is leaked
    def leak_SB(self, flag):
        if flag == "TrustGetter":
            tg = TrustGetter(self.trust_path)
        elif flag == "EpinionsGetter":
            tg = EpinionsGetter(self.trust_path)
        else:
            tg = DoubanGetter(self.trust_path)

        # Call the get_relations method and obtain the matrix
        matrix = tg.relation_matrix

        # Output the dimensions of the sparse matrix
        print("The dimensions of the sparse matrix are:", matrix.shape)

        # matrix1 = np.array([[0, 0, 0, 0],
        #            [-1, 1, 0, 1],
        #            [1, 0, -1, 1],
        #            [1, 1, 1, 1]])

        matrix = csr_matrix(matrix)

        # Check if the matrix has an even number of rows; if not, add a zero row
        if matrix.shape[0] % 2 != 0:
            zero_row = csr_matrix(np.zeros((1, matrix.shape[1])))  # Create a zero row with the same number of columns as the matrix
            matrix = vstack([matrix, zero_row])  # Append the zero row to the end of the matrix
        
        rows, cols = matrix.shape
        
        diff_rows = []
        
        for i in range(0, rows-1, 2):
            diff = matrix[i+1, :] - matrix[i, :]
            diff_rows.append(diff)
        
        leak_matrix = vstack(diff_rows)
        restore_matrix = self.restore_original_matrix_optimized(leak_matrix)
        similarity, c = self.calculate_similarity_sparse(matrix, restore_matrix)
        print("Number and proportion of zero elements in A:")
        print(similarity)
        print("Proportion of non-zero elements that are equal in A and B:")
        print(c)
        return similarity
        # print("Elements that are the same in both matrices:")
        # print(simi)

        # random_row = random.randint(0, rows - 50)
        # random_col = random.randint(0, cols - 50)
        # # Create MatrixPainter instance
        # painter = MatrixPainter(matrix, restore_matrix)

        # # Draw matrix comparison plot
        # painter.paint_comparison(random_row, random_col, 50)

# Usage example
if __name__ == '__main__':
    exp_tg = Exp('./data/ft_trust.txt')   
    percentage_tg = exp_tg.leak_SB("TrustGetter")
    # message = f"In the ft_trust dataset, {percentage_tg}% of the data was leaked"
    # print(message)

    exp_eg = Exp('./data/epinions.txt')  
    percentage_eg = exp_eg.leak_SB("EpinionsGetter")
    # message = f"In the epinions dataset, {percentage_eg}% of the data was leaked"
    # print(message)

    exp_dg = Exp('./data/out.douban')   
    percentage_dg = exp_dg.leak_SB("DoubanGetter")