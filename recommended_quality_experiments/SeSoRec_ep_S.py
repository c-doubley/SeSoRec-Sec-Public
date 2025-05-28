"""
File name: SeSoRec_ep_S.py

Description:
    Reproduction version of SeSoRec, including SSMM protocol
    Using Epinions dataset
    Corresponds to SeSoRec_ep_S.py

Usage:
    cd recommended_quality_experiments
    python SeSoRec_ep_R.py
    python SeSoRec_ep_S.py

Author: 
Date: 2025/4/17
"""


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

def ssmm_b(Q, sock):
    Q = np.array(Q, dtype=np.float64)
    Q = Q.T
    y, z = Q.shape
    pad_row = y % 2
    if pad_row:
        Q = np.pad(Q, ((0, 1), (0, 0)), 'constant')
        y += 1
    
    Q_prime = np.random.rand(y, z)
    Q_e = Q_prime[::2, :]
    Q_o = Q_prime[1::2, :]
    Q1 = Q_prime - Q
    Q2 = Q_e - Q_o
    
    response = receive_with_length(sock)
    if not isinstance(response, tuple) or len(response) != 2:
        raise TypeError(f"Expected tuple of length 2, got {type(response)}")
    P1, P2 = response
    P1 = np.array(P1, dtype=np.float64)
    P2 = np.array(P2, dtype=np.float64)
    
    send_with_length(sock, (Q1, Q2))
    N = P1 @ (2 * Q - Q_prime) - P2 @ (Q2 + Q_e)
    if pad_row:
        N = N[:, :-1]
    send_with_length(sock, N)

class TrustGetter:
    def __init__(self):
        self.config = ConfigX()
        self.config.trust_path = "./data/cv/Epinions/trust.txt"  #
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
                    if not request:
                        print("Connection closed by server")
                        break
                        
                    if isinstance(request, tuple) and len(request) >= 1:
                        command = request[0]
                    else:
                        command = request
                        
                    if command == "GET_SOCIAL_USERS":
                        print("Sending social users")
                        send_with_length(client_socket, list(social.tg.user.keys()))
                    elif command == "GET_RATINGS":
                        user = request[1]
                        print(f"Received request for ratings of user {user}")
                        send_with_length(client_socket, {})
                    elif command == "SSMM_D":
                        batch_users = request[1]
                        print(f"Processing SSMM_D request, batch size: {len(batch_users)}")
                        D_B, _, _ = social.get_social_matrices(batch_users)
                        ssmm_b(D_B, client_socket)
                    elif command == "SSMM_S":
                        batch_users = request[1]
                        print(f"Processing SSMM_S request, batch size: {len(batch_users)}")
                        _, S_B, _ = social.get_social_matrices(batch_users)
                        ssmm_b(S_B, client_socket)
                    elif command == "SSMM_E":
                        batch_users = request[1]
                        print(f"Processing SSMM_E request, batch size: {len(batch_users)}")
                        _, _, E_B = social.get_social_matrices(batch_users)
                        ssmm_b(E_B, client_socket)
                    elif command == "TRAIN_DONE":
                        print(f"Training completed for fold {k}")
                        break
                    else:
                        print(f"Unknown command: {command}")
                        
            except Exception as e:
                print(f"Error in client: {e}")
                import traceback
                traceback.print_exc()
                
            print(f"Fold {k} client shutting down")
        time.sleep(5)

if __name__ == "__main__":
    client()