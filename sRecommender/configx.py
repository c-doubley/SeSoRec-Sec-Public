class ConfigX:
    def __init__(self):
        self.dataset_name = "ft"
        self.rating_cv_path = "./data/cv/"
        self.trust_path = "./data/ft_trust.txt"
        self.sep = " "
        self.min_val = 0.5
        self.max_val = 4.0
        self.factor = 10
        self.lr = 0.01
        self.maxIter = 100  # 增加到 500
        self.lambdaP = 0.001
        self.gamma = 2  # 初始值，可调
        self.threshold = 1e-4