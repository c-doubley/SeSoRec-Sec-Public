import numpy as np
from PIL import Image
import os
import random

class MNISTLoader:
    def __init__(self, directory="./data/minist"):
        self.directory = directory

    def load_random_image(self, label):
        """
        从指定目录随机加载一个指定标签的图片，并将其转换为矩阵。
        """
        # 构建指定标签的图片文件路径列表
        image_files = [f for f in os.listdir(self.directory) if f.endswith('.png') and f.startswith(str(label))]
        
        # 随机选择一个图片文件
        random_image_file = random.choice(image_files)
        image_path = os.path.join(self.directory, random_image_file)
        
        # 使用Pillow库读取图片
        image = Image.open(image_path)
        
        # 将图片转换为灰度模式，然后转换为NumPy矩阵
        image_matrix = np.array(image.convert('L'))
        
        return image_matrix

# 示例使用
if __name__ == "__main__":
    loader = MNISTLoader("./data/minist")  # 如果存放MNIST数据的目录不同，请调整路径
    label = 0  # 指定想要加载的图片的标签
    image_matrix = loader.load_random_image(label)
    
    # 展示加载的图片矩阵的形状，以验证加载成功
    print("Loaded image matrix shape:", image_matrix.shape)
    
    # 如果需要，可以使用matplotlib展示图片
    # import matplotlib.pyplot as plt
    # plt.imshow(image_matrix, cmap='gray')
    # plt.show()

