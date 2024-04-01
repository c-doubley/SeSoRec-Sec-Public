import numpy as np
import os
from scipy.sparse import dok_matrix

class EpinionsGetter(object):
    """
    This class reads trust data from a given path and saves it in a sparse matrix.
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
            for line in f:
                parts = line.strip().split(self.sep)
                if len(parts) < 3:  # Ensure the line has at least 3 parts
                    continue
                u_from, u_to, _ = map(int, parts[:3])
                max_id = max(max_id, u_from, u_to)

        # Initialize a sparse matrix
        relation_matrix = dok_matrix((max_id + 1, max_id + 1), dtype=np.int32)

        # Second pass: populate the matrix
        with open(self.data_path, 'r') as f:
            for line in f:
                parts = line.strip().split(self.sep)
                if len(parts) < 3:  # Ensure the line has at least 3 parts
                    continue
                u_from, u_to, weight = map(int, parts[:3])
                relation_matrix[u_from, u_to] = weight

        return relation_matrix

    def print_matrix_sample(self):
        """
        Prints a 10x10 sample of the relation matrix.
        """
        # Ensure the matrix is at least 10x10
        max_index = min(self.relation_matrix.shape[0], 10)
        # Convert the relevant part of the matrix to dense format for easy printing
        sample = self.relation_matrix[5:15, 66601:66611].toarray()
        
        print("Sample of the relation matrix (first 10x10 elements):")
        for row in sample:
            print(" ".join(f"{val:3}" for val in row))



def clean_data(file_path):
        # 数据清洗并且重新编码
        # 假设原始文件路径是 './data/epinions_user_rating.txt'，你将处理后的数据保存到 './data/epinions.txt'

        # 读取原始数据，只保留前三列
        data = pd.read_csv(file_path, sep='\t', header=None, usecols=[0, 1, 2], names=['from_id', 'to_id', 'weight'])

        # 生成新的连续用户ID
        unique_ids = np.unique(data[['from_id', 'to_id']].values.flatten())
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

        # 应用映射
        data['from_id'] = data['from_id'].map(id_mapping)
        data['to_id'] = data['to_id'].map(id_mapping)

        # 保存处理后的数据
        data.to_csv('./data/epinions.txt', sep=' ', index=False, header=False)



if __name__ == '__main__':
    # Replace './data/epinions.txt' with the actual path to your cleaned data file
    eg = EpinionsGetter('./data/epinions.txt')
    eg.print_matrix_sample()


