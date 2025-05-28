# cross_validation.py
# encoding:utf-8
import os
import numpy as np
from configx import ConfigX

def split_5_folds(config):
    if not os.path.exists(config.rating_cv_path):
        os.makedirs(config.rating_cv_path)
    
    ratings = []
    with open(config.rating_path, 'r') as f:
        for line in f:
            ratings.append(line.strip())
    
    np.random.seed(0)
    np.random.shuffle(ratings)
    fold_size = len(ratings) // config.k_fold_num
    
    for k in range(config.k_fold_num):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < config.k_fold_num - 1 else len(ratings)
        test_data = ratings[test_start:test_end]
        train_data = ratings[:test_start] + ratings[test_end:]
        
        with open(f"{config.rating_cv_path}{config.dataset_name}-{k}.txt", 'w') as f:
            f.write('\n'.join(test_data))
        with open(f"{config.rating_cv_path}{config.dataset_name}-{k}-train.txt", 'w') as f:
            f.write('\n'.join(train_data))

if __name__ == "__main__":
    config = ConfigX()
    config.k_fold_num = 5
    config.rating_path = "./data/ft_ratings.txt"  # 调整为相对路径
    config.rating_cv_path = "./data/cv/"
    split_5_folds(config)