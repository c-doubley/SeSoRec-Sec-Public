# TTP.py - 可信第三方服务器
import socket
import pickle
import numpy as np
import threading

def send_with_length(sock, data):
    serialized_data = pickle.dumps(data)
    length = len(serialized_data)
    sock.sendall(length.to_bytes(8, byteorder='big'))
    sock.sendall(serialized_data)

def receive_with_length(sock):
    length_bytes = sock.recv(8)
    if not length_bytes:
        return None
    length = int.from_bytes(length_bytes, byteorder='big')
    data = b""
    while len(data) < length:
        part = sock.recv(min(4096, length - len(data)))
        if not part:
            raise ConnectionError("Connection closed")
        data += part
    return pickle.loads(data)

def generate_matrix_triple(d1, d2, d3):
    """生成矩阵乘法三元组 (E, F, EF)"""
    # 确保维度是整数
    d1, d2, d3 = int(d1), int(d2), int(d3)
    
    print(f"Generating matrices with shapes: E({d1}x{d2}), F({d2}x{d3})")
    
    # 生成正确形状的矩阵
    E = np.random.rand(d1, d2)
    F = np.random.rand(d2, d3)
    EF = E @ F
    
    # 将E和F分成两份
    E_0 = np.random.rand(d1, d2)
    E_1 = E - E_0
    
    F_0 = np.random.rand(d2, d3)
    F_1 = F - F_0
    
    # 将EF分成两份
    EF_0 = E_0 @ F_0 + E_0 @ F_1 + E_1 @ F_0
    EF_1 = EF - EF_0
    
    # 打印形状信息以便调试
    print(f"Generated triple shapes: E_0({E_0.shape}), F_0({F_0.shape}), EF_0({EF_0.shape})")
    print(f"Generated triple shapes: E_1({E_1.shape}), F_1({F_1.shape}), EF_1({EF_1.shape})")
    
    return (E_0, F_0, EF_0), (E_1, F_1, EF_1)

def handle_client(client_socket, addr):
    """处理客户端连接"""
    print(f"Connected by {addr}")
    try:
        while True:
            request = receive_with_length(client_socket)
            if not request:
                print(f"Connection closed by {addr}")
                break
                
            if isinstance(request, tuple) and request[0] == "GET_TRIPLE":
                _, d1, d2, d3 = request
                print(f"Received request for matrix triple with dimensions: {d1}x{d2}, {d2}x{d3}")
                
                # 确保所有维度都是有效的
                if d1 is None or d1 <= 0:
                    d1 = 10
                    print(f"Invalid d1, using default: {d1}")
                if d2 is None or d2 <= 0:
                    d2 = 10
                    print(f"Invalid d2, using default: {d2}")
                if d3 is None or d3 <= 0:
                    d3 = 10
                    print(f"Invalid d3, using default: {d3}")
                
                print(f"Generating matrix triple with dimensions: {d1}x{d2}, {d2}x{d3}")
                
                # 生成三元组
                triple_0, triple_1 = generate_matrix_triple(d1, d2, d3)
                
                # 根据客户端地址决定发送哪个三元组
                # 假设Recommender是第一个连接的客户端
                if addr[1] % 2 == 0:  # 简单的区分方法，可以根据实际情况调整
                    send_with_length(client_socket, triple_0)
                    print(f"Sent triple_0 to {addr}")
                else:
                    send_with_length(client_socket, triple_1)
                    print(f"Sent triple_1 to {addr}")
            else:
                print(f"Unknown request from {addr}: {request}")
                
    except Exception as e:
        print(f"Error handling client {addr}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client_socket.close()
        print(f"Connection with {addr} closed")

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind(('localhost', 12346))
        server_socket.listen(5)
        print("TTP server started, waiting for connections...")
        
        while True:
            client_socket, addr = server_socket.accept()
            client_thread = threading.Thread(target=handle_client, args=(client_socket, addr))
            client_thread.daemon = True
            client_thread.start()
            
    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        server_socket.close()

if __name__ == "__main__":
    main()