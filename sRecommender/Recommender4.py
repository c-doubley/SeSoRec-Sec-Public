"""
文件名称: Recommender4.py

描述:
    SeSoRec的安全版本SeSoRec-Sec
    PPMM替换SSMM
    使用 TrustFilm 数据集
    对应Social4.py
功能:

用法:
    python Recommender4.py

作者: chenyuyue
日期: 2025/4/17
"""
# 使用了PPMM的SoSeRec ft数据集
# Recommender.py
import sys
import socket
import pickle
import numpy as np
from collections import defaultdict
from configx import ConfigX
import time

sys.path.append("..")

def normalize(rating, min_val=0.5, max_val=4.0):
    return (rating - min_val) / (max_val - min_val)

def denormalize(rating, min_val=0.5, max_val=4.0):
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

def ppmm_a(A, sock):
    d1, d2 = A.shape
    print(f"PPMM_A input shape: {A.shape}")
    E = np.random.rand(d1, d2)
    E_0 = np.random.rand(d1, d2)
    E_1 = E - E_0
    send_with_length(sock, E_1)
    F_0, F = receive_with_length(sock)
    F_0 = np.array(F_0, dtype=np.float64)
    F = np.array(F, dtype=np.float64)
    EF = E @ F
    EF_0 = np.random.rand(d1, EF.shape[1])
    EF_1 = EF - EF_0
    send_with_length(sock, EF_1)
    A_hat = A - E
    send_with_length(sock, A_hat)
    B_hat = receive_with_length(sock)
    B_hat = np.array(B_hat, dtype=np.float64)
    C_0 = E_0 @ B_hat + A_hat @ F_0 + EF_0
    C_1 = receive_with_length(sock)
    C_1 = np.array(C_1, dtype=np.float64)
    result = C_0 + C_1
    print(f"PPMM_A output shape: {result.shape}")
    return result

class RatingGetter:
    def __init__(self, k):
        self.config = ConfigX()
        self.config.rating_cv_path = "./data/cv/"
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

    def compute_ndcg(self, pred_ratings, true_ratings, n=10):
        ndcg_sum = 0.0
        user_count = 0
        for u in true_ratings:
            pred_u = [(i, pred_ratings[u].get(i, 0)) for i in range(len(self.rg.item))]
            true_u = true_ratings[u]
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

            # 使用 PPMM 替换 SSMM
            send_with_length(sock, ("PPMM_D", batch_users))
            U_B_D_B_T = ppmm_a(U_B, sock)

            send_with_length(sock, ("PPMM_S", batch_users))
            U_S_B_T = ppmm_a(self.U, sock)

            send_with_length(sock, ("PPMM_E", batch_users))
            U_B_E_B_T = ppmm_a(U_B, sock)

            pred = U_B.T @ V_B
            error = I_B * (R_B - pred)
            loss_basic = 0.5 * np.sum(error ** 2)
            loss_social = (0.5 * self.config.gamma * np.sum(U_B_D_B_T) -
                           self.config.gamma * np.sum(U_S_B_T) +
                           0.5 * self.config.gamma * np.sum(U_B_E_B_T))
            loss_reg = 0.5 * self.config.lambdaP * (np.sum(U_B ** 2) + np.sum(V_B ** 2))
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

            term1 = -V_B @ (error * I_B).T
            term2 = 0.5 * self.config.gamma * U_B_D_B_T
            term3 = -self.config.gamma * U_S_B_T
            term4 = 0.5 * self.config.gamma * U_B_E_B_T
            term5 = self.config.lambdaP * U_B

            grad_U_B = term1 + term2 + term3 + term4 + term5
            grad_V_B = -U_B @ (error * I_B) + self.config.lambdaP * V_B

            U_B -= self.config.lr * grad_U_B
            V_B -= self.config.lr * grad_V_B

            for u, idx in u_idx.items():
                self.U[:, self.social_user_map[u]] = U_B[:, idx]
            for i, idx in v_idx.items():
                self.V[:, self.rg.item[i]] = V_B[:, idx]

            iteration += 1
            print(f"Iteration {iteration}: Loss = {self.loss:.4f}, RMSE_train = {rmse_train:.4f}, "
                  f"RMSE_test = {rmse_test:.4f}, NDCG@10 = {ndcg_test:.4f}, Time = {time.time() - start_time:.2f}s")
            if iteration > 1 and abs(self.loss - self.prev_loss) < self.config.threshold:
                print("Converged: Loss change below threshold")
                break
            self.prev_loss = self.loss

        print("Training completed, sending TRAIN_DONE")
        send_with_length(sock, "TRAIN_DONE")
        return rmse_test, ndcg_test

if __name__ == "__main__":
    rmse_folds, ndcg_folds = [], []
    for k in range(5):
        print(f"Running fold {k}")
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
                if request[0] == "GET_RATINGS":
                    user = request[1]
                    ratings = recommender.rg.get_row(user)
                    send_with_length(conn, ratings)
                elif request[0] == "TRAIN":
                    rmse_test, ndcg_test = recommender.train_model(conn)
                    rmse_folds.append(rmse_test)
                    ndcg_folds.append(ndcg_test)
                    print(f"Fold {k}: RMSE_test = {rmse_test:.4f}, NDCG@10 = {ndcg_test:.4f}")
                    break
        server_socket.close()
    print(f"Average RMSE: {np.mean(rmse_folds):.4f}, Average NDCG@10: {np.mean(ndcg_folds):.4f}")