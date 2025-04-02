
# 能跑的代码 临时保存一下
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
            raise ConnectionError("Connection closed before receiving all data")
        data += part
    return pickle.loads(data)

class TrustGetter:
    def __init__(self):
        self.config = ConfigX()
        self.config.trust_path = "./data/ft_trust.txt"
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
        self._init_social_matrix()

    def _init_social_matrix(self):
        print("Initializing social matrix S...")
        for u in self.tg.user:
            for f in self.tg.get_followees(u):
                self.S[self.tg.user[u], self.tg.user[f]] = self.tg.followees[u][f]

    def init_user_sim(self, sock):
        print("Constructing user-user similarity matrix...")
        for u in self.tg.user:
            for f in self.tg.get_followees(u):
                key = f"{u}-{f}"
                if key not in self.user_sim:
                    print(f"Requesting ratings for user {u}")
                    send_with_length(sock, ("GET_RATINGS", u))
                    u_ratings = receive_with_length(sock)
                    print(f"Received ratings for user {u}: {u_ratings}")
                    print(f"Requesting ratings for user {f}")
                    send_with_length(sock, ("GET_RATINGS", f))
                    f_ratings = receive_with_length(sock)
                    print(f"Received ratings for user {f}: {f_ratings}")
                    sim = (pearson_sp(u_ratings, f_ratings) + 1.0) / 2.0
                    self.user_sim[key] = sim

    def get_social_matrices(self, batch_users):
        n_users = len(self.tg.user)
        valid_batch_users = [u for u in batch_users if u in self.tg.user]
        if not valid_batch_users:
            return np.zeros((1, 1)), np.zeros((1, n_users)), np.zeros((1, 1))
        n_batch_users = len(valid_batch_users)
        u_idx = {u: i for i, u in enumerate(valid_batch_users)}

        D_B = np.zeros((n_batch_users, n_batch_users))
        for b in valid_batch_users:
            d_b = np.sum([self.S[u_idx[b], self.tg.user[f]] for f in self.tg.user])
            D_B[u_idx[b], u_idx[b]] = d_b

        S_B = np.zeros((n_batch_users, n_users))
        for b in valid_batch_users:
            for f in range(n_users):
                S_B[u_idx[b], f] = self.S[u_idx[b], f]

        E_B = np.zeros((n_batch_users, n_batch_users))
        for b in valid_batch_users:
            e_b = np.sum([self.S[self.tg.user[u], self.tg.user[b]] for u in self.tg.user if u in self.tg.user])
            E_B[u_idx[b], u_idx[b]] = e_b

        return D_B, S_B, E_B

def client():
    social = Social()
    for k in range(5):
        print(f"Starting fold {k}")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 12345 + k))
        with client_socket:
            social.init_user_sim(client_socket)
            send_with_length(client_socket, ("TRAIN", None))
            iteration_count = 0
            while True:
                request = receive_with_length(client_socket)
                if not request:
                    print("Connection closed by server")
                    break
                if request[0] == "GET_SOCIAL_USERS":
                    print("Sending social users")
                    send_with_length(client_socket, list(social.tg.user.keys()))
                elif request[0] == "GET_SOCIAL_MATRICES":
                    iteration_count += 1
                    batch_users = request[1]
                    print(f"Processing GET_SOCIAL_MATRICES request {iteration_count}, batch size: {len(batch_users)}")
                    D_B, S_B, E_B = social.get_social_matrices(batch_users)
                    send_with_length(client_socket, (D_B, S_B, E_B))
                elif request[0] == "TRAIN_DONE":
                    print("Training completed for fold {k}")
                    break
            print(f"Fold {k} client shutting down")
        time.sleep(1)  # 短暂等待，确保服务器准备好下一个折

if __name__ == "__main__":
    client()