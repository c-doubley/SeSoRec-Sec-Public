
import sys
import os
import numpy as np
from scipy.sparse import dok_matrix

class DoubanGetter(object):
    """
    This class reads data from a given path and saves it in a sparse matrix.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.sep = ' '
        self.relation_matrix = self.get_relations()

    def get_relations(self):
        if not os.path.isfile(self.data_path):
            print("The path specified does not exist or is not a file.")
            sys.exit()

        # First pass: identify the maximum user ID to determine the size of the matrix
        max_id = 0
        with open(self.data_path, 'r') as f:
            # Skip the first line
            next(f)
            for line in f:
                parts = line.strip().split(self.sep)
                if len(parts) < 2:  # Ensure the line has at least 2 parts
                    continue
                u_from, u_to = map(int, parts[:2])
                max_id = max(max_id, u_from, u_to)

        # Initialize a sparse matrix
        relation_matrix = dok_matrix((max_id + 1, max_id + 1), dtype=np.int32)

        # Second pass: populate the matrix
        with open(self.data_path, 'r') as f:
            next(f)  # Skip the first line again           
            for line in f:
                parts = line.strip().split(self.sep)
                if len(parts) < 2:  # Ensure the line has at least 2 parts
                    continue
                u_from, u_to = map(int, parts[:2])
                relation_matrix[u_from, u_to] = 1
                relation_matrix[u_to, u_from] = 1  # Assuming this is an undirected graph

        return relation_matrix

    def print_matrix_sample(self):
        """
        Prints a 10x10 sample of the relation matrix.
        """
        # Ensure the matrix is at least 10x10
        max_index = min(self.relation_matrix.shape[0], 10)
        # Convert the relevant part of the matrix to dense format for easy printing
        sample = self.relation_matrix[:10, :10].toarray()
        
        print("Sample of the relation matrix (first 10x10 elements):")
        for row in sample:
            print(" ".join(f"{val:3}" for val in row))


if __name__ == '__main__':
    # Replace './data/out.douban' with the actual path to your Douban data file
    dg = DoubanGetter('./data/out.douban')
    dg.print_matrix_sample()
