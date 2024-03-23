
import socket
import numpy as np
import pickle

def send_matrix(conn, matrix):
    # 使用pickle序列化numpy矩阵
    serialized_matrix = pickle.dumps(matrix)

    # 获取序列化后的数据大小
    size = len(serialized_matrix)

    # 发送矩阵大小信息，首先将大小转换为4字节的形式
    conn.sendall(size.to_bytes(4, byteorder='big'))

    # 发送矩阵数据
    conn.sendall(serialized_matrix)

    # 如果需要调试信息，可以打印发送的数据大小
    print("Sending matrix, size:", size, "bytes")




def receive_matrix(conn):
    # 接收矩阵大小信息
    size_data = conn.recv(4)
    size = int.from_bytes(size_data, byteorder='big')

    # 接收矩阵数据
    data = conn.recv(size)

    # 反序列化矩阵
    matrix = pickle.loads(data)

    # 如果需要调试
    # print(f"Received matrix, size: {size} bytes")

    return matrix
