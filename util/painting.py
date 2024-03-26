# painting.py
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class MatrixPainter:
    def __init__(self, ):
        """
        初始化MatrixPainter类。

        参数:
        - matrix_A: 第一个稀疏矩阵 (csr_matrix)
        - matrix_B: 第二个稀疏矩阵 (csr_matrix)
        """
      

    def paint_comparison(self, matrix_A, matrix_B, row, col, num, file_path='picture/comparison.png'):
        """
        绘制两个矩阵的比较图，并将结果保存为PNG文件。

        参数:
        - file_path: 保存图片的路径和文件名 (默认为'picture/comparison.png')
        """
        # 转换为密集格式并仅考虑前10x10的元素
        dense_A = matrix_A.toarray()[row:row+num, col:col+num]
        dense_B = matrix_B.toarray()[row:row+num, col:col+num]

        # 计算两个矩阵中相同元素的位置
        mask = dense_A == dense_B

        # 创建图形和轴对象
        fig, ax = plt.subplots()

        # 画出矩阵的每一个元素
        for i in range(dense_A.shape[0]):
            for j in range(dense_A.shape[1]):
                # 使用红色或白色背景
                color = 'red' if dense_A[i, j] == dense_B[i, j] else 'white'
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))


        # 设置坐标轴的范围
        ax.set_xlim(0, num)
        ax.set_ylim(0, num)

        # 移除刻度线
        ax.tick_params(
            axis='both',       # 应用于x轴和y轴的改变
            which='both',      # 应用于主刻度线和次刻度线的改变
            bottom=False,      # 移除x轴底部的刻度线
            top=False,         # 移除x轴顶部的刻度线
            left=False,        # 移除y轴左侧的刻度线
            right=False,       # 移除y轴右侧的刻度线
            labelbottom=False, # 移除x轴底部的刻度标签
            labelleft=False    # 移除y轴左侧的刻度标签
        )

        # 绘制外围边框线
        for spine in ax.spines.values():
            spine.set_visible(True)

        # 保留内部网格线
        for i in range(1, num):
            ax.axhline(i, color='black', linestyle='-', linewidth=2)
            ax.axvline(i, color='black', linestyle='-', linewidth=2)


        # 翻转y轴，以便第一行在顶部
        ax.invert_yaxis()

        # 不显示图形，而是保存到文件
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
    

    def minist_painting(self, matrix_A, matrix_B, matrix_C):
        # 假设A, B, C为图片对应的numpy矩阵
        # 先将numpy数组转为PIL图片
        A_img = Image.fromarray(matrix_A)
        B_img = Image.fromarray(matrix_B)
        C_img = Image.fromarray(matrix_C)

        # 获取A图片尺寸 进行resize
        a_width, a_height = A_img.size
        A_img_resized = A_img.resize((a_width * 2, a_height *2))
        # B_img_resized = B_img.resize((a_width//2, a_height//2))
        # C_img_resized = C_img.resize((a_width//2, a_height//2))

        # 再将resize后的B C PIL图片转为numpy数组
        B_resized = np.array(B_img)
        C_resized = np.array(C_img)

        # 垂直堆叠B和C图像
        stacked_BC = np.vstack((B_resized, C_resized))
        
        # 水平堆叠A, B_resized, C_resized
        stacked_imgs = np.hstack((A_img_resized, stacked_BC))

        # 显示并保存图像
        plt.imshow(stacked_imgs, cmap='gray')
        plt.axis('off')
        plt.savefig('picture/stacked.png')
        plt.show()



    def stack_images(self, pics):
        # 使用函数：
        # pics = [(A1, B1, C1), (A2, B2, C2), ... , (A5, B5, C5)] 假设这样的列表存在
        # final_img = stack_images(pics)
                   
        
        stacked_imgs_list = []
        final_images = []

        for i, (A, B, C) in enumerate(pics):
            # 将numpy数组转成PIL图片
            A_img, B_img, C_img = map(Image.fromarray, [A, B, C])
        
            # 水平堆叠A图像和堆叠后的BC图像
            stacked_imgs = np.hstack((A_img, B_img, C_img))

            stacked_imgs_list.append(stacked_imgs)

            # 每5次垂直堆叠后进行一次水平堆叠
            if (i + 1) % 5 == 0:
                final_images.append(np.vstack(stacked_imgs_list))
                stacked_imgs_list = []
            
        # 为了保证如果有不足五组的情况也能处理，所以我们在循环结束后也要进行垂直堆叠
        if stacked_imgs_list:
            final_images.append(np.vstack(stacked_imgs_list))
        
        # 最终的垂直堆叠图像进行水平堆叠
        image = np.hstack(final_images)

        # 显示并保存图像
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig('picture/final_stacked.png')
        plt.show()

        # return final_stacked_img

    def stack_images(self, pics1, pics2):
        # 使用函数：
        # pics = [(A1, B1, C1), (A2, B2, C2), ... , (A5, B5, C5)] 假设这样的列表存在
        # final_img = stack_images(pics)

        
        stacked_imgs_list = []
        final_images1 = []
        final_images2 = []

        for i, (A, B, C) in enumerate(pics1):
            # 将numpy数组转成PIL图片
            A_img, B_img, C_img = map(Image.fromarray, [A, B, C])
        
            # 水平堆叠A图像和堆叠后的BC图像
            stacked_imgs = np.hstack((A_img, B_img, C_img))

            stacked_imgs_list.append(stacked_imgs)

            # 每5次垂直堆叠后进行一次水平堆叠
            if (i + 1) % 5 == 0:
                final_images1.append(np.vstack(stacked_imgs_list))
                stacked_imgs_list = []
            
        # 为了保证如果有不足五组的情况也能处理，所以我们在循环结束后也要进行垂直堆叠
        if stacked_imgs_list:
            final_images1.append(np.vstack(stacked_imgs_list))
        
        # 最终的垂直堆叠图像进行水平堆叠
        image1 = np.hstack(final_images1)


        stacked_imgs_list = []
        for i, (A, B, C) in enumerate(pics2):
            # 将numpy数组转成PIL图片
            A_img, B_img, C_img = map(Image.fromarray, [A, B, C])
        
            # 水平堆叠A图像和堆叠后的BC图像
            stacked_imgs = np.hstack((A_img, B_img, C_img))

            stacked_imgs_list.append(stacked_imgs)

            # 每5次垂直堆叠后进行一次水平堆叠
            if (i + 1) % 5 == 0:
                final_images2.append(np.vstack(stacked_imgs_list))
                stacked_imgs_list = []
            
        # 为了保证如果有不足五组的情况也能处理，所以我们在循环结束后也要进行垂直堆叠
        if stacked_imgs_list:
            final_images2.append(np.vstack(stacked_imgs_list))
        
        # 最终的垂直堆叠图像进行水平堆叠
        image2 = np.hstack(final_images2)

        image = np.vstack((image1, image2))


        # 显示并保存图像
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig('picture/final_stacked.png')
        plt.show()

        # return final_stacked_img

# 使用示例
if __name__ == '__main__':

    # 创建示例稀疏矩阵A和B
    A = csr_matrix(np.random.randint(0, 2, size=(50, 50)))
    B = csr_matrix(np.random.randint(0, 2, size=(50, 50)))
    # 创建MatrixPainter实例
    painter = MatrixPainter()

    # 绘制矩阵比较图
    painter.paint_comparison(A, B, 30,30)
