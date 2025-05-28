"""
File name: attack_MNIST_SSMM.py

Description:
    This file needs to complete the MNIST part of the experiment. It is divided into four steps:
    1. Select grayscale images from the MNIST dataset and convert them to matrices
    2. Generate leaked information, i.e., odd columns + even columns, even rows - odd rows
    3. Reconstruct 2 matrices based on 2 types of leaked information
    4. Draw the original matrix + 2 reconstructed matrices as images

Function:
    - process_matrices: Generate 2 types of leaked information based on grayscale image matrix, 
      and reconstruct matrices based on these 2 types of leaked information

Usage:
    python attack_experiments/attack_MNIST_SSMM.py

Output:
    picture/attack_MNIST_PPMM.png
    (util/paiting.mnist_stack_images or util/paiting.minist_painting)

Author: 
Date: 2024/4/28
"""
import sys
from pathlib import Path
# Get project root directory
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
import numpy as np
from PIL import Image
import os
import random
from reader.mnist import MNISTLoader
from util.painting import MatrixPainter

class PictureProcessor:
    def __init__(self, matrix):
        self.matrix = matrix

    def process_matrices(self):
        # matrix = self._add_column_if_needed(self.matrix)
        # Generate leaked information for left party, odd columns + even columns
        leak_left = self._generate_leak_left(self.matrix)
        # Reconstruct matrix based on odd columns + even columns
        restore_left = self._generate_matrix_C(leak_left)
        # Generate leaked information for right party, even rows - odd rows
        leak_right = self._generate_matrix_D(self.matrix)
        # Reconstruct matrix based on even rows - odd rows
        restore_right = self._generate_matrix_E(leak_right)
        return leak_left, restore_left, leak_right, restore_right

    def _add_column_if_needed(self, matrix):
        # If the number of columns is not even, add a column of zeros to the right
        if matrix.shape[1] % 2 != 0:
            matrix = np.hstack((matrix, np.zeros((matrix.shape[0], 1), dtype=matrix.dtype)))
        return matrix

    def _generate_leak_left(self, matrix):
        # Add even columns and odd columns to get new matrix leak_left
        leak_left = np.zeros((matrix.shape[0], matrix.shape[1] // 2), dtype=matrix.dtype)
        for j in range(leak_left.shape[1]):
            leak_left[:, j] = matrix[:, 2*j] + matrix[:, 2*j+1]
        return leak_left

    def _generate_matrix_C(self, leak_left):
        # Generate new matrix C based on matrix leak_left
        restore_left = np.zeros((leak_left.shape[0], leak_left.shape[1] * 2), dtype=leak_left.dtype)
        for i in range(leak_left.shape[0]):
            for j in range(leak_left.shape[1]):
                if leak_left[i, j] != 0:
                    restore_left[i, 2*j+1] = leak_left[i, j]
        return restore_left

    def _add_row_if_needed(self, matrix):
        # If the number of rows is not even, add a row of zeros at the bottom
        if matrix.shape[0] % 2 != 0:
            matrix = np.vstack((matrix, np.zeros((1, matrix.shape[1]), dtype=matrix.dtype)))
        return matrix

    def _generate_matrix_D(self, matrix):
        # Subtract odd rows from even rows to get new matrix D
        leak_right = np.zeros((matrix.shape[0] // 2, matrix.shape[1]), dtype=matrix.dtype)
        for i in range(leak_right.shape[0]):
            leak_right[i, :] = matrix[2*i, :] - matrix[2*i + 1, :]
        return leak_right

    def _generate_matrix_E(self, leak_right):
        # Generate new matrix E based on matrix D
        restore_right = np.zeros((leak_right.shape[0] * 2, leak_right.shape[1]), dtype=leak_right.dtype)
        for i in range(leak_right.shape[0]):
            for j in range(leak_right.shape[1]):
                if leak_right[i, j] == 0:
                    restore_right[2*i, j] = restore_right[2*i + 1, j] = 0
                elif leak_right[i, j] < 0:
                    restore_right[2*i, j] = -leak_right[i, j] * 2 
                    restore_right[2*i + 1, j] = 0
                else:
                    restore_right[2*i, j] = 0
                    restore_right[2*i + 1, j] = leak_right[i, j] * 2
        return restore_right

def save_image(matrix, file_path):
    """
    Save the matrix as a grayscale PNG image.
    """
    # Convert matrix to image
    image = Image.fromarray(matrix.astype(np.uint8), 'L')
    # Save image
    image.save(file_path)


def add_255_border(matrix):
    """
    Add two rows and columns of 255 to the last row and rightmost column of a NumPy matrix

    Parameters:
    matrix (numpy.ndarray): Input 2D grayscale image matrix

    Returns:
    numpy.ndarray: Matrix with 255 border added
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy matrix")
    
    # Create a row and column of 255
    row_255 = np.full((1, matrix.shape[1]), 255, dtype=np.uint8)
    col_255 = np.full((matrix.shape[0] + 1, 1), 255, dtype=np.uint8)

    # Add row first, then column
    matrix_with_row = np.vstack([matrix, row_255])
    matrix_with_row_and_col = np.hstack([matrix_with_row, col_255])
    
    return matrix_with_row_and_col


# Example usage
if __name__ == "__main__":

    # Ensure picture directory exists
    os.makedirs('picture', exist_ok=True)

    loader = MNISTLoader("data/minist")

    tuple_matrices1 = []
    tuple_matrices2 = []
    # Read grayscale images with labels 0-4 and convert to matrices
    for i in range(5):
        list_matrices = loader.load_random_images(i)        
        for matrix in list_matrices:
            processor = PictureProcessor(matrix)
            leak_left, restore_left, leak_right, restore_right = processor.process_matrices()
            # tuple_matrices1.append((matrix, restore_left, restore_right))
            tuple_matrices1.append((add_255_border(matrix), add_255_border(restore_left), add_255_border(restore_right)))
    # Read grayscale images with labels 5-9 and convert to matrices
    for i in range(5, 10):
        list_matrices = loader.load_random_images(i)        
        for matrix in list_matrices:
            processor = PictureProcessor(matrix)
            leak_left, restore_left, leak_right, restore_right = processor.process_matrices()
            # tuple_matrices2.append((matrix, restore_left, restore_right))
            tuple_matrices2.append((add_255_border(matrix), add_255_border(restore_left), add_255_border(restore_right)))

    
    painter = MatrixPainter()
    # Draw matrix comparison plot
    painter.mnist_stack_images(tuple_matrices1, tuple_matrices2)


    # Here's an example of drawing just one image
    # Randomly load an image with label and convert it to matrix
    loader = MNISTLoader("data/minist")
    matrix = loader.load_random_image(0)
    
    # Ensure picture directory exists
    os.makedirs('picture', exist_ok=True)
    # Save selected image to picture directory
    save_image(matrix, 'picture/mnist_original_image.png')
    
    # Process matrix and generate matrices leak_left and C
    processor = PictureProcessor(matrix)
    leak_left, restore_left, leak_right, restore_right = processor.process_matrices()
    
    # Save matrix C to picture directory
    save_image(restore_left, 'picture/mnist_leak_left.png')
    save_image(restore_right, 'picture/mnist_leak_right.png')

    painter = MatrixPainter()
    # Draw matrix comparison plot
    painter.minist_painting(matrix, restore_left, restore_right)

