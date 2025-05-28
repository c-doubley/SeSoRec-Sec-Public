"""
File name: soreg_tf_ILS_S.py

Description:
    A non-privacy version of SeSoRec, which does not include the SSMM protocol and directly transmits the required matrix data
    TrustFilm dataset
    Corresponding to soreg_tf_ILS_R.py
    Includes ILS evaluation for diversity.

Function:

    Added ILS (Intra-List Similarity) evaluation as a diversity indicator

Usage:

    cd recommended_quality_experiments
    python soreg_tf_ILS_R.py
    python soreg_tf_ILS_S.py

Author:  

Date: 2025/4/17
"""
# Non-privacy-preserving
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
        self.config.trust_path = "./data/TrustFilm/ft_trust.txt"
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
        # Initialize the social similarity matrix instead of directly using social relationships
        self.S = np.zeros((len(self.tg.user), len(self.tg.user)))
        # Mapping from user ID to index
        self.user_idx_map = {u: idx for u, idx in self.tg.user.items()}
        self.idx_user_map = {idx: u for u, idx in self.tg.user.items()}

    def init_user_sim(self, sock):
        print("Constructing user-user similarity matrix...")
        # Obtain rating data for all users
        all_ratings = {}
        for u in self.tg.user:
            print(f"Requesting ratings for user {u}")
            send_with_length(sock, ("GET_RATINGS", u))
            u_ratings = receive_with_length(sock)
            all_ratings[u] = u_ratings
            
        # Compute user similarity and populate the S matrix
        for u in self.tg.user:
            u_idx = self.tg.user[u]
            # For each user, compute similarity with users they have social relationships with
            for f in self.tg.get_followees(u):
                f_idx = self.tg.user[f]
                # Calculate similarity using Pearson correlation coefficient
                sim = (pearson_sp(all_ratings.get(u, {}), all_ratings.get(f, {})) + 1.0) / 2.0
                # Store similarity in the S matrix
                self.S[u_idx, f_idx] = sim
                # Also store in the user_sim dictionary for easy lookup
                self.user_sim[f"{u}-{f}"] = sim
                
        print("User similarity matrix constructed")

    def get_social_matrices(self, batch_users):
        """
        Generate social matrices for batch users
        D_B: Diagonal matrix representing users' out-degree
        S_B: Social similarity matrix
        E_B: Diagonal matrix representing users' in-degree
        """
        n_users = len(self.tg.user)
        # Filter out users not in the social network
        valid_batch_users = [u for u in batch_users if u in self.tg.user]
        
        if not valid_batch_users:
            return np.zeros((1, 1)), np.zeros((1, n_users)), np.zeros((1, 1))
            
        n_batch_users = len(valid_batch_users)
        # Mapping from user ID to index within the batch
        batch_idx_map = {u: i for i, u in enumerate(valid_batch_users)}
        
        # Initialize matrices
        D_B = np.zeros((n_batch_users, n_batch_users))
        S_B = np.zeros((n_batch_users, n_users))
        E_B = np.zeros((n_batch_users, n_batch_users))
        
        # Compute D_B and S_B
        for b_idx, b in enumerate(valid_batch_users):
            b_global_idx = self.tg.user[b]
            
            # Compute user b's out-degree (diagonal elements of D_B)
            # Use global index to retrieve data from the S matrix
            d_b = np.sum(self.S[b_global_idx, :])
            D_B[b_idx, b_idx] = d_b
            
            # Populate the S_B matrix
            for f_global_idx in range(n_users):
                S_B[b_idx, f_global_idx] = self.S[b_global_idx, f_global_idx]
        
        # Compute E_B
        for b_idx, b in enumerate(valid_batch_users):
            b_global_idx = self.tg.user[b]
            
            # Compute user b's in-degree (diagonal elements of E_B)
            e_b = np.sum(self.S[:, b_global_idx])
            E_B[b_idx, b_idx] = e_b
            
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
                elif request == "TRAIN_DONE":
                    print(f"Training completed for fold {k}")
                    break
            print(f"Fold {k} client shutting down")
        time.sleep(1)  # Brief wait to ensure the server is ready for the next fold

if __name__ == "__main__":
    client()