import numpy as np
import socket
import struct
from numpy.random import Generator, MT19937, SeedSequence
from transfer import send_matrix, receive_matrix

def OMMProtocol(sock, inputMatrix, PartyNum):
    matrix = np.array(inputMatrix)
    rows, cols = matrix.shape

    # 随机数生成器
    rg = Generator(MT19937(SeedSequence()))

    # 参与方P0
    if PartyNum == 0:
        # 如果A的列数是奇数，则给A在最后添加一个全零的列
        if cols % 2 != 0:
            matrix = np.hstack((matrix, np.zeros((rows, 1))))
            cols += 1

        # Step 1: random E
        E = rg.integers(low=0, high=np.iinfo(np.uint64).max, size=(rows, cols), dtype=np.uint64)
        # E = np.array([[1, 1], [1, 1]])
        print("E:")
        print(E)

        # Step 2: Extract odd and even columns from E
        E_odd = E[:, 0::2]
        E_even = E[:, 1::2]

        # Step 3: Calculate A0, E0 and send to P1
        E0 = E_odd + E_even
        A0 = matrix - E
        send_matrix(sock, A0)
        send_matrix(sock, E0)

        # Receive B1, F1 from P1
        B1 = receive_matrix(sock)
        F1 = receive_matrix(sock)

        # Step 4: Calculate C0
        C0 = matrix @ B1 + E_odd @ F1
        print("matrix:")
        print(matrix)
        print("C0:")
        print(C0)
   
        return C0

    else:
        # 如果B的行数是奇数，则给B在最后添加一个全零的行
        if rows % 2 != 0:
            matrix = np.vstack((matrix, np.zeros((1, cols))))
            rows += 1

        # Step 1: random F
        F = rg.integers(low=0, high=np.iinfo(np.uint64).max, size=(rows, cols), dtype=np.uint64)
        # F = np.array([[2, 2], [2, 2]])

        # Step 2: Extract odd and even rows of F
        F_odd = F[0::2, :]
        F_even = F[1::2, :]

        # Receive A0, E0 from P0
        A0 = receive_matrix(sock)
        E0 = receive_matrix(sock)

        # Step 3: Calculate B1, F1 and send to P0
        B1 = matrix - F
        F1 = F_odd - F_even

        send_matrix(sock, B1)
        send_matrix(sock, F1)

        # Step 4: Calculate C1
        C1 = A0 @ F + E0 @ F_even

        return C1
