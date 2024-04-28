import socket
import numpy as np
import json
from util.transfer import send_matrix, receive_matrix
from omm import OMMProtocol
from numpy.random import Generator, MT19937, SeedSequence

def main():
    host = 'localhost'
    port = 12345

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # Receiving matrix from server
    # 随机数生成器
    # rg = Generator(MT19937(SeedSequence()))
    # B = rg.integers(low=0, high=np.iinfo(np.uint64).max, size=(2, 2), dtype=np.uint64)
    B = np.array([[2, 2], [5, 4]], dtype=np.uint64)

    C1 = OMMProtocol(client_socket, B, 1)

    send_matrix(client_socket, B)
    send_matrix(client_socket, C1)


    # Sending modified matrix back to server
    # client_socket.send(json.dumps(modified_matrix.tolist()).encode('utf-8'))
    client_socket.close()


if __name__ == '__main__':
    main()
