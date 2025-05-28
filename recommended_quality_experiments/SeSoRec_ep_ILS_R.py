"""
File Name: SeSoRec_ep_ILS_R.py

Description:
    A reimplementation of SeSoRec with SSMM protocol.
    Uses the Epinions dataset.
    Corresponds to SeSoRec_ep_ILS_S.py.
    Includes ILS evaluation for diversity.

Usage:
    cd recommended_quality_experiments
    python SeSoRec_ep_ILS_R.py
    python SeSoRec_ep_ILS_S.py

Author: 
Date: 2025/4/17
"""


import sys
import socket
import pickle
import numpy as np
from collections import defaultdict
from configx import ConfigX
import time
import scipy.sparse as sp

sys.path.append("..")

def normalize(rating, min_val=1.0, max_val=5.0):
    return (rating - min_val) / (max_val - min_val)

def denormalize(rating, min_val=1.0, max_val=5.0):
    return rating * (max_val - min_val) + min_val

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

def ssmm_a(P, sock):
    original_shape = P.shape
    print(f"SSMM_A input shape: {original_shape}")
    
    x, y = P.shape
    pad_col = y % 2
    if pad_col:
        P = np.pad(P, ((0, 0), (0, 1)), 'constant')
        y += 1
    
    P_prime = np.random.rand(x, y)
    P_e = P_prime[:, ::2]
    P_o = P_prime[:, 1::2]
    P1 = P + P_prime
    P2 = P_e + P_o
    
    send_with_length(sock, (P1, P2))
    
    response = receive_with_length(sock)
    if not isinstance(response, tuple) or len(response) != 2:
        raise TypeError(f"Expected tuple of length 2, got {type(response)}")
    Q1, Q2 = response
    Q1 = np.array(Q1, dtype=np.float64)
    Q2 = np.array(Q2, dtype=np.float64)
    
    M = (P + 2 * P_prime) @ Q1 + (P2 + P_o) @ Q2
    N = receive_with_length(sock)
    N = np.array(N, dtype=np.float64)
    
    if M.shape != N.shape:
        min_rows = min(M.shape[0], N.shape[0])
        min_cols = min(M.shape[1], N.shape[1])
        M = M[:min_rows, :min_cols]
        N = N[:min_rows, :min_cols]
    
    result = M + N
    if result.shape != original_shape:
        if result.shape[1] < original_shape[1]:
            result = np.pad(result, ((0, 0), (0, original_shape[1] - result.shape[1])), 'constant')
        elif result.shape[1] > original_shape[1]:
            result = result[:, :original_shape[1]]
    
    print(f"SSMM_A output shape: {result.shape}")
    return result

class RatingGetter:
    def __init__(self, k):
        self.config = ConfigX()
        self.config.rating_cv_path = "./data/cv/Epinions/"
        self.config.dataset_name = "Epinions"
        self.config.min_val = 1.0
        self.config.max_val = 5.0
        self.config.sep = "\t"
        self.k = k
        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.trainSet_u = defaultdict(dict)
        self.testSet_u = defaultdict(dict)
        self._load_data()

    def _load_data(self):
        train_path = f"{self.config.rating_cv_path}{self.config.dataset_name}-{self.k}-train.txt"
        with open(train_path, 'r') as f:
            for line in f:
                u, i, r = map(float, line.strip().split(self.config.sep))
                u, i = int(u), int(i)
                r = normalize(r, self.config.min_val, self.config.max_val)
                if u not in self.user:
                    self.user[u] = len(self.user)
                    self.id2user[self.user[u]] = u
                if i not in self.item:
                    self.item[i] = len(self.item)
                    self.id2item[self.item[i]] = i
                self.trainSet_u[u][i] = r

        test_path = f"{self.config.rating_cv_path}{self.config.dataset_name}-{self.k}.txt"
        with open(test_path, 'r') as f:
            for line in f:
                u, i, r = map(float, line.strip().split(self.config.sep))
                u, i = int(u), int(i)
                r = normalize(r, self.config.min_val, self.config.max_val)
                if u not in self.user:
                    self.user[u] = len(self.user)
                    self.id2user[self.user[u]] = u
                if i not in self.item:
                    self.item[i] = len(self.item)
                    self.id2item[self.item[i]] = i
                self.testSet_u[u][i] = r

    def get_row(self, u):
        return self.trainSet_u.get(u, {})

    def get_batch(self, batch_size, social_users):
        ratings = [(u, i, r) for u in self.trainSet_u for i, r in self.trainSet_u[u].items() if u in social_users]
        if not ratings:
            raise ValueError("No valid ratings found")
        np.random.shuffle(ratings)
        return ratings[:min(batch_size, len(ratings))]

    def get_test_ratings(self):
        return [(u, i, r) for u in self.testSet_u for i, r in self.testSet_u[u].items()]

class Recommender:
    def __init__(self, k, batch_size=64):
        self.config = ConfigX()
        self.rg = RatingGetter(k)
        self.batch_size = batch_size
        self.social_users = None
        self.U = None
        self.V = np.random.rand(self.config.factor, len(self.rg.item)) / (self.config.factor ** 0.5)
        self.loss = 0.0
        self.prev_loss = 0.0
        self.item_user_matrix = self.build_item_user_matrix()

    def build_item_user_matrix(self):
        """
        Build a sparse item-user rating matrix.
        Returns: scipy.sparse.csr_matrix
        """
        n_items = len(self.rg.item)
        n_users = len(self.rg.user)
        rows, cols, data = [], [], []
        for u in self.rg.trainSet_u:
            for i, r in self.rg.trainSet_u[u].items():
                rows.append(self.rg.item[i])
                cols.append(self.rg.user[u])
                data.append(r)
        return sp.csr_matrix((data, (rows, cols)), shape=(n_items, n_users), dtype=np.float32)

    def compute_item_similarity(self, i, j):
        """
        Compute cosine similarity between items i and j on demand.
        """
        vec_i = self.item_user_matrix[i].toarray().flatten()
        vec_j = self.item_user_matrix[j].toarray().flatten()
        common_users = np.logical_and(vec_i != 0, vec_j != 0)
        if not np.any(common_users):
            return 0.0
        vec_i, vec_j = vec_i[common_users], vec_j[common_users]
        dot_product = np.sum(vec_i * vec_j)
        norm_i = np.sqrt(np.sum(vec_i ** 2))
        norm_j = np.sqrt(np.sum(vec_j ** 2))
        if norm_i == 0 or norm_j == 0:
            return 0.0
        sim = dot_product / (norm_i * norm_j)
        return max(0, sim)

    def compute_ils(self, pred_ratings, n=10):
        """
        Compute Intra-List Similarity (ILS) as a diversity metric.
        pred_ratings: predicted ratings dictionary
        n: length of the recommendation list (Top-N)
        Returns: average ILS value
        """
        ils_sum = 0.0
        user_count = 0
        for u in pred_ratings:
            pred_u = [(i, pred_ratings[u].get(i, 0)) for i in range(len(self.rg.item))]
            pred_u.sort(key=lambda x: x[1], reverse=True)
            top_n = [i for i, _ in pred_u[:n]]
            if len(top_n) < 2:
                continue
            ils = 0.0
            pairs = 0
            for i in range(len(top_n)):
                for j in range(i + 1, len(top_n)):
                    sim = self.compute_item_similarity(top_n[i], top_n[j])
                    ils += sim
                    pairs += 1
            if pairs > 0:
                ils /= pairs
            ils_sum += ils
            user_count += 1
        return ils_sum / user_count if user_count > 0 else 0.0

    def compute_ndcg(self, pred_ratings, true_ratings, n=10):
        ndcg_sum = 0.0
        user_count = 0
        for u in true_ratings:
            pred_u = [(i, pred_ratings[u].get(i, 0)) for i in range(len(self.rg.item))]
            true_u = {i: denormalize(r, self.config.min_val, self.config.max_val) for i, r in true_ratings[u].items()}
            pred_u.sort(key=lambda x: x[1], reverse=True)
            top_n_pred = [i for i, _ in pred_u[:n]]
            dcg = 0.0
            for i, item in enumerate(top_n_pred):
                if item in true_u:
                    dcg += (2**true_u[item] - 1) / np.log2(i + 2)
            ideal = sorted(true_u.values(), reverse=True)[:n]
            idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
            ndcg_sum += dcg / idcg if idcg > 0 else 0
            user_count += 1
        return ndcg_sum / user_count if user_count > 0 else 0.0

    def train_model(self, sock):
        if self.social_users is None:
            print("Requesting social users from platform B...")
            send_with_length(sock, ("GET_SOCIAL_USERS", None))
            self.social_users = receive_with_length(sock)
            print(f"Received social users: {self.social_users[:5]}... (total: {len(self.social_users) if self.social_users else 0})")
            
            if not self.social_users:
                raise ValueError("No social users received")
                
            self.U = np.random.rand(self.config.factor, len(self.social_users)) / (self.config.factor ** 0.5)
            self.social_user_map = {u: i for i, u in enumerate(self.social_users)}
            print(f"Initialized U shape: {self.U.shape}, factor: {self.config.factor}")

        iteration = 0
        while iteration < self.config.maxIter:
            start_time = time.time()
            batch = self.rg.get_batch(self.batch_size, self.social_users)
            batch_users = sorted(set(u for u, _, _ in batch))
            batch_items = sorted(set(i for _, i, _ in batch))
            u_idx = {u: i for i, u in enumerate(batch_users)}
            v_idx = {i: j for j, i in enumerate(batch_items)}

            R_B = np.zeros((len(batch_users), len(batch_items)))
            I_B = np.zeros((len(batch_users), len(batch_items)))
            for u, i, r in batch:
                R_B[u_idx[u], v_idx[i]] = r
                I_B[u_idx[u], v_idx[i]] = 1

            U_B = self.U[:, [self.social_user_map[u] for u in batch_users]]
            V_B = self.V[:, [self.rg.item[i] for i in batch_items]]

            send_with_length(sock, ("SSMM_D", batch_users))
            U_B_D_B_T = ssmm_a(U_B, sock)

            send_with_length(sock, ("SSMM_S", batch_users))
            U_S_B_T = ssmm_a(self.U[:, [self.social_user_map[u] for u in batch_users]], sock)

            send_with_length(sock, ("SSMM_E", batch_users))
            U_B_E_B_T = ssmm_a(U_B, sock)

            min_cols = min(U_B_D_B_T.shape[1], U_S_B_T.shape[1], U_B_E_B_T.shape[1], U_B.shape[1])
            print(f"SSMM results shapes: U_B_D_B_T={U_B_D_B_T.shape}, U_S_B_T={U_S_B_T.shape}, U_B_E_B_T={U_B_E_B_T.shape}, U_B={U_B.shape}")
            
            U_B_D_B_T = U_B_D_B_T[:, :min_cols]
            U_S_B_T = U_S_B_T[:, :min_cols]
            U_B_E_B_T = U_B_E_B_T[:, :min_cols]
            U_B_adjusted = U_B[:, :min_cols]

            pred = U_B_adjusted.T @ V_B
            error = I_B * (R_B - pred)
            loss_basic = 0.5 * np.sum(error ** 2)
            loss_social = (0.5 * self.config.gamma * np.sum(U_B_D_B_T) -
                           self.config.gamma * np.sum(U_S_B_T) +
                           0.5 * self.config.gamma * np.sum(U_B_E_B_T))
            loss_reg = 0.5 * self.config.lambdaP * (np.sum(U_B_adjusted ** 2) + np.sum(V_B ** 2))
            self.loss = loss_basic + loss_social + loss_reg

            n_ratings = np.sum(I_B)
            rmse_train = denormalize(np.sqrt(np.sum(error ** 2) / n_ratings), self.config.min_val, self.config.max_val) if n_ratings > 0 else 0.0

            test_ratings = self.rg.get_test_ratings()
            pred_ratings = defaultdict(dict)
            true_ratings = defaultdict(dict)
            for u, i, r in test_ratings:
                true_ratings[u][i] = r
                if u in self.social_user_map and i in self.rg.item:
                    pred_ratings[u][i] = self.U[:, self.social_user_map[u]] @ self.V[:, self.rg.item[i]]
            rmse_test = 0.0
            if test_ratings:
                error_sum = sum((pred_ratings[u].get(i, 0) - r) ** 2 for u, i, r in test_ratings)
                rmse_test = denormalize(np.sqrt(error_sum / len(test_ratings)), self.config.min_val, self.config.max_val)
            ndcg_test = self.compute_ndcg(pred_ratings, true_ratings, n=10)
            ils_test = self.compute_ils(pred_ratings, n=10)
            diversity_test = 1 - ils_test

            term1 = -V_B @ (error * I_B).T
            term2 = 0.5 * self.config.gamma * U_B_D_B_T
            term3 = -self.config.gamma * U_S_B_T
            term4 = 0.5 * self.config.gamma * U_B_E_B_T
            term5 = self.config.lambdaP * U_B_adjusted

            min_cols = min(term.shape[1] for term in [term1, term2, term3, term4, term5])
            term1 = term1[:, :min_cols]
            term2 = term2[:, :min_cols]
            term3 = term3[:, :min_cols]
            term4 = term4[:, :min_cols]
            term5 = term5[:, :min_cols]

            grad_U_B = term1 + term2 + term3 + term4 + term5
            grad_V_B = -U_B_adjusted @ (error * I_B) + self.config.lambdaP * V_B

            U_B_adjusted -= self.config.lr * grad_U_B
            V_B -= self.config.lr * grad_V_B

            for u, idx in u_idx.items():
                if idx < min_cols:
                    self.U[:, self.social_user_map[u]] = U_B_adjusted[:, idx]
            for i, idx in v_idx.items():
                self.V[:, self.rg.item[i]] = V_B[:, idx]

            iteration += 1
            print(f"Iteration {iteration}: Loss = {self.loss:.4f}, RMSE_train = {rmse_train:.4f}, "
                  f"RMSE_test = {rmse_test:.4f}, NDCG@10 = {ndcg_test:.4f}, "
                  f"ILS = {ils_test:.4f}, Diversity = {diversity_test:.4f}, "
                  f"Time = {time.time() - start_time:.2f}s")
            if iteration > 1 and abs(self.loss - self.prev_loss) < self.config.threshold:
                print("Converged: Loss change below threshold")
                break
            self.prev_loss = self.loss

        print("Training completed, sending TRAIN_DONE")
        send_with_length(sock, "TRAIN_DONE")
        return rmse_test, ndcg_test, ils_test, diversity_test

if __name__ == "__main__":
    # Run only one fold (fold 0)
    k = 0
    print(f"Running experiment with fold {k}")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', 12345))
    server_socket.listen(1)
    print(f"P0 Server started for fold {k}, waiting for P1 connection...")
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    recommender = Recommender(k, batch_size=64)
    with conn:
        while True:
            request = receive_with_length(conn)
            if not request:
                print("Connection closed by client")
                break
            if isinstance(request, tuple) and len(request) >= 1:
                command = request[0]
            else:
                command = request
            if command == "GET_RATINGS":
                user = request[1]
                ratings = recommender.rg.get_row(user)
                send_with_length(conn, ratings)
            elif command == "TRAIN":
                print("Starting training...")
                rmse_test, ndcg_test, ils_test, diversity_test = recommender.train_model(conn)
                print(f"Experiment with fold {k}: RMSE_test = {rmse_test:.4f}, NDCG@10 = {ndcg_test:.4f}, "
                      f"ILS = {ils_test:.4f}, Diversity = {diversity_test:.4f}")
                break
    server_socket.close()

    result = (f"RMSE: {rmse_test:.4f}, NDCG@10: {ndcg_test:.4f}, "
              f"ILS: {ils_test:.4f}, Diversity: {diversity_test:.4f}")
    print(result)
    with open("result.txt", "a") as result_file:
        result_file.write(result + "\n")