import socket
import numpy as np
import json
from util.transfer import send_matrix, receive_matrix
from omm import OMMProtocol
from numpy.random import Generator, MT19937, SeedSequence

def generate_random_matrix():
    return np.random.rand(2, 2)

def main():
    host = 'localhost'
    port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print("Server listening on port", port)

    conn, addr = server_socket.accept()
    print("Connection from: " + str(addr))

    # 随机数生成器
    # rg = Generator(MT19937(SeedSequence()))
    # A = rg.integers(low=0, high=np.iinfo(np.uint64).max, size=(2, 2), dtype=np.uint64)
    A = np.array([[1, 2], [3, 4]], dtype=np.uint64)

    C0 = OMMProtocol(conn, A, 0)

    B = receive_matrix(conn)
    C1 = receive_matrix(conn)

    print("A * B:")
    print(A @ B)
    print("C0 + C1:")
    print(C0 + C1)

    conn.close()

if __name__ == '__main__':
    main()
