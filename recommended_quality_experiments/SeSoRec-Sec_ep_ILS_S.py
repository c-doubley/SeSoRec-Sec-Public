"""
文件名称: Social6.py

描述:
    SeSoRec-Sec实现
    使用PPMM替换SSMM
    使用Epinion数据集
功能:


用法:
    python Social6.py

作者: chenyuyue
日期: 2025/4/17
"""
# Social.py
import sys
import socket
import pickle
import numpy as np
import time
from collections import defaultdict
from configx import ConfigX

sys.path.append("..")

def pearson_sp(x1, x2):
    if x1 is None or x2 is None or not x1 or not x2:
        return 0.0
    common = set(x1.keys()) & set(x2.keys())
    if not common:
        return 0
    ratingList1 = [x1[i] for i in common]
    ratingList2 = [x2[i] for i in common]
    if not ratingList1:
        return 0
    avg1 = sum(ratingList1) / len(ratingList1)
    avg2 = sum(ratingList2) / len(ratingList2)
    mult, sum1, sum2 = 0.0, 0.0, 0.0
    for i in range(len(ratingList1)):
        mult += (ratingList1[i] - avg1) * (ratingList2[i] - avg2)
        sum1 += (ratingList1[i] - avg1) ** 2
        sum2 += (ratingList2[i] - avg2) ** 2
    return mult / (np.sqrt(sum1) * np.sqrt(sum2)) if sum1 > 0 and sum2 > 0 else 0

def send_with_length(sock, data):
    serialized_data = pickle.dumps(data, protocol=4)
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
            raise ConnectionError("Connection closed before receiving all data")
        data += part
    return pickle.loads(data)

def ppmm_b(B, sock):
    d2, d3 = B.shape
    print(f"PPMM_B input shape: {B.shape}")
    # 接收 A 的维度
    a_dims = receive_with_length(sock)
    if a_dims is None:
        raise ValueError("Failed to receive A dimensions from Recommender.py")
    d1, a_d2 = a_dims
    # 发送 B 的维度
    send_with_length(sock, (d2, d3))
    E_1 = receive_with_length(sock)
    if E_1 is None:
        raise ValueError("Failed to receive E_1 from Recommender.py")
    E_1 = np.array(E_1, dtype=np.float64)
    print(f"Received E_1 shape: {E_1.shape}")
    F = np.random.rand(d2, d3)
    F_1 = np.random.rand(d2, d3)
    F_0 = F - F_1
    send_with_length(sock, (F_0, F))
    print(f"Sent F_0 shape: {F_0.shape}, F shape: {F.shape}")
    EF_1 = receive_with_length(sock)
    if EF_1 is None:
        raise ValueError("Failed to receive EF_1 from Recommender.py")
    EF_1 = np.array(EF_1, dtype=np.float64)
    A_hat = receive_with_length(sock)
    if A_hat is None:
        raise ValueError("Failed to receive A_hat from Recommender.py")
    A_hat = np.array(A_hat, dtype=np.float64)
    B_hat = B - F
    send_with_length(sock, B_hat)
    print(f"Sent B_hat shape: {B_hat.shape}")
    C_1 = A_hat @ B_hat + E_1 @ B_hat + A_hat @ F_1 + EF_1
    send_with_length(sock, C_1)
    print(f"PPMM_B completed for shape: {B.shape}")

def client():
    social = Social()
    for k in range(5):
        print(f"Starting fold {k}")
        max_retries = 5
        retry_count = 0
        connected = False
        while retry_count < max_retries and not connected:
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect(('localhost', 12345))
                connected = True
                print(f"Successfully connected to server")
            except ConnectionRefusedError:
                retry_count += 1
                print(f"Connection refused, retrying... ({retry_count}/{max_retries})")
                time.sleep(2)
        if not connected:
            print(f"Cannot connect to server, skipping fold {k}")
            continue
        with client_socket:
            try:
                print("Sending TRAIN request to server")
                send_with_length(client_socket, ("TRAIN", None))
                while True:
                    request = receive_with_length(client_socket)
                    if request is None:
                        print("Connection closed by server or no data received")
                        break
                    if isinstance(request, tuple) and len(request) >= 1:
                        command = request[0]
                        if command == "GET_SOCIAL_USERS":
                            all_users = list(social.tg.user.keys())
                            social_users = all_users  
                            print(f"Sending social users, total: {len(social_users)}")
                            send_with_length(client_socket, social_users)
                        elif command in ["PPMM_D", "PPMM_S", "PPMM_E"]:
                            batch_users = request[1]
                            print(f"Processing {command} request, batch size: {len(batch_users)}")
                            if command == "PPMM_D":
                                D_B, _, _ = social.get_social_matrices(batch_users)
                                ppmm_b(D_B.T, client_socket)
                            elif command == "PPMM_S":
                                _, S_B, _ = social.get_social_matrices(batch_users)
                                ppmm_b(S_B.T, client_socket)
                            elif command == "PPMM_E":
                                _, _, E_B = social.get_social_matrices(batch_users)
                                ppmm_b(E_B.T, client_socket)
                        elif command == "TRAIN_DONE":
                            print(f"Training completed for fold {k}")
                            break
                        else:
                            print(f"Unknown command: {command}")
                    else:
                        print(f"Received unexpected data: {request}")
            except Exception as e:
                print(f"Error in client: {e}")
                import traceback
                traceback.print_exc()
            print(f"Fold {k} client shutting down")
        time.sleep(5)

class TrustGetter:
    def __init__(self):
        self.config = ConfigX()
        self.config.trust_path = "./data/cv/Epinions/trust.txt"
        self.user = {}
        self.followees = defaultdict(dict)
        self.followers = defaultdict(dict)
        self._load_data()

    def _load_data(self):
        with open(self.config.trust_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    print(f"Skipping invalid line: {line.strip()}")
                    continue
                u_from, u_to, t = map(float, parts)
                u_from, u_to = int(u_from), int(u_to)
                if u_from not in self.user:
                    self.user[u_from] = len(self.user)
                if u_to not in self.user:
                    self.user[u_to] = len(self.user)
                self.followees[u_from][u_to] = t
                self.followers[u_to][u_from] = t

    def get_followees(self, u):
        return self.followees[u]

    def get_followers(self, u):
        return self.followers[u]

class Social:
    def __init__(self):
        self.config = ConfigX()
        self.tg = TrustGetter()
        self.user_sim = {}
        self.S = np.zeros((len(self.tg.user), len(self.tg.user)))
        self.user_idx_map = {u: idx for u, idx in self.tg.user.items()}
        self.idx_user_map = {idx: u for u, idx in self.tg.user.items()}

    def init_user_sim(self, sock):
        print("Constructing user-user similarity matrix...")
        all_ratings = {}
        for u in self.tg.user:
            print(f"Requesting ratings for user {u}")
            send_with_length(sock, ("GET_RATINGS", u))
            u_ratings = receive_with_length(sock)
            all_ratings[u] = u_ratings
        for u in self.tg.user:
            u_idx = self.tg.user[u]
            for f in self.tg.get_followees(u):
                f_idx = self.tg.user[f]
                sim = (pearson_sp(all_ratings.get(u, {}), all_ratings.get(f, {})) + 1.0) / 2.0
                self.S[u_idx, f_idx] = sim
                self.user_sim[f"{u}-{f}"] = sim
        print("User similarity matrix constructed")

    def get_social_matrices(self, batch_users):
        n_users = len(self.tg.user)
        valid_batch_users = [u for u in batch_users if u in self.tg.user]
        if not valid_batch_users:
            return np.zeros((1, 1)), np.zeros((1, n_users)), np.zeros((1, 1))
        n_batch_users = len(valid_batch_users)
        batch_idx_map = {u: i for i, u in enumerate(valid_batch_users)}
        D_B = np.zeros((n_batch_users, n_batch_users))
        S_B = np.zeros((n_batch_users, n_users))
        E_B = np.zeros((n_batch_users, n_batch_users))
        for b_idx, b in enumerate(valid_batch_users):
            b_global_idx = self.tg.user[b]
            d_b = np.sum(self.S[b_global_idx, :])
            D_B[b_idx, b_idx] = d_b
            for f_global_idx in range(n_users):
                S_B[b_idx, f_global_idx] = self.S[b_global_idx, f_global_idx]
        for b_idx, b in enumerate(valid_batch_users):
            b_global_idx = self.tg.user[b]
            e_b = np.sum(self.S[:, b_global_idx])
            E_B[b_idx, b_idx] = e_b
        return D_B, S_B, E_B

if __name__ == "__main__":
    client()