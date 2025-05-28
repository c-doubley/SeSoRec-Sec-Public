"""
File Name: SeSoRec_tf_R.py

Description:
    Reproduction version of SeSoRec, including the SSMM protocol
    Uses the TrustFilm dataset
    Corresponds to SeSoRec_tf_S.py
Functionality:

Usage:
    cd recommended_quality_experiments
    python SeSoRec_tf_R.py
    python SeSoRec_tf_S.py

Author:  
Date: 2025/4/17
"""

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


def ssmm_a(P, sock):
    """
    Secure Matrix Multiplication Implementation for Platform A
    P: Matrix held by Platform A
    sock: Socket for communication with Platform B
    Returns: M + N = P * Q, where Q is the matrix held by Platform B
    """
    # Save the original shape to return the result with the correct size
    original_shape = P.shape
    print(f"SSMM_A input shape: {original_shape}")
    
    x, y = P.shape
    pad_col = y % 2  # 1 if y is odd, 0 if even
    if pad_col:
        P = np.pad(P, ((0, 0), (0, 1)), 'constant')  # If y is odd, add a column of zeros
        y += 1
    
    # Step 1: Generate random matrix P'
    P_prime = np.random.rand(x, y)
    
    # Step 2: Extract even and odd columns of P'
    P_e = P_prime[:, ::2]  # Even columns
    P_o = P_prime[:, 1::2]  # Odd columns
    
    # Step 4: Compute P1 and P2
    P1 = P + P_prime
    P2 = P_e + P_o
    
    # Send P1 and P2 to Platform B
    send_with_length(sock, (P1, P2))
    
    # Step 5: Receive Q1 and Q2
    response = receive_with_length(sock)
    if not isinstance(response, tuple) or len(response) != 2:
        raise TypeError(f"Expected tuple of length 2, got {type(response)}")
    Q1, Q2 = response
    
    # Ensure Q1 and Q2 are numpy arrays and of floating-point type
    Q1 = np.array(Q1, dtype=np.float64)
    Q2 = np.array(Q2, dtype=np.float64)
    
    # Step 6: Compute M
    M = (P + 2 * P_prime) @ Q1 + (P2 + P_o) @ Q2
    
    # Step 8: Receive N and compute M + N
    N = receive_with_length(sock)
    N = np.array(N, dtype=np.float64)  # Ensure N is a numpy array and of floating-point type
    
    # Ensure M and N have consistent shapes
    if M.shape != N.shape:
        print(f"Warning: M shape {M.shape} does not match N shape {N.shape}, trying to adjust...")
        # Truncate to the smaller dimension
        min_rows = min(M.shape[0], N.shape[0])
        min_cols = min(M.shape[1], N.shape[1])
        M = M[:min_rows, :min_cols]
        N = N[:min_rows, :min_cols]
    
    result = M + N
    
    # Ensure the result matches the original shape of P
    if result.shape != original_shape:
        print(f"Warning: Result shape {result.shape} does not match original shape {original_shape}, trying to adjust...")
        # If the result has fewer columns than the original, pad with zero columns
        if result.shape[1] < original_shape[1]:
            result = np.pad(result, ((0, 0), (0, original_shape[1] - result.shape[1])), 'constant')
        # If the result has more columns than the original, truncate
        elif result.shape[1] > original_shape[1]:
            result = result[:, :original_shape[1]]
    
    print(f"SSMM_A output shape: {result.shape}")
    return result

class RatingGetter:
    def __init__(self, k):
        self.config = ConfigX()
        self.config.rating_cv_path = "./data/cv/TrustFilm/"
        self.k = k
        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.trainSet_u = defaultdict(dict)
        self.testSet_u = defaultdict(dict)
        self._load_data()

    def _load_data(self):
        # Load training set
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

        # Load test set
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
            pred_u.sort(key=lambda x: x[1], reverse=True)  # Sort by predicted score
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

            # Use SSMM to compute U_B * D_B^T
            send_with_length(sock, ("SSMM_D", batch_users))
            U_B_D_B_T = ssmm_a(U_B, sock)

            # Use SSMM to compute U * S_B^T
            send_with_length(sock, ("SSMM_S", batch_users))
            U_S_B_T = ssmm_a(self.U, sock)

            # Use SSMM to compute U_B * E_B^T
            send_with_length(sock, ("SSMM_E", batch_users))
            U_B_E_B_T = ssmm_a(U_B, sock)

            # Ensure all SSMM results have consistent shapes
            min_cols = min(U_B_D_B_T.shape[1], U_S_B_T.shape[1], U_B_E_B_T.shape[1], U_B.shape[1])
            print(f"SSMM results shapes: U_B_D_B_T={U_B_D_B_T.shape}, U_S_B_T={U_S_B_T.shape}, U_B_E_B_T={U_B_E_B_T.shape}, U_B={U_B.shape}")
            
            # Truncate all matrices to the same number of columns
            U_B_D_B_T = U_B_D_B_T[:, :min_cols]
            U_S_B_T = U_S_B_T[:, :min_cols]
            U_B_E_B_T = U_B_E_B_T[:, :min_cols]
            U_B_adjusted = U_B[:, :min_cols]  # Adjust U_B to match other matrices

            # Use adjusted U_B for prediction
            pred = U_B_adjusted.T @ V_B
            error = I_B * (R_B - pred)
            loss_basic = 0.5 * np.sum(error ** 2)
            
            # Modify social loss calculation using SSMM results
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

            term1 = -V_B @ (error * I_B).T
            term2 = 0.5 * self.config.gamma * U_B_D_B_T
            term3 = -self.config.gamma * U_S_B_T
            term4 = 0.5 * self.config.gamma * U_B_E_B_T
            term5 = self.config.lambdaP * U_B_adjusted  # Use adjusted U_B

            # Print shapes of all terms
            shapes = [term.shape for term in [term1, term2, term3, term4, term5]]
            print(f"Term shapes before adjustment: {shapes}")
            
            # Ensure all terms have consistent shapes - find the smallest number of columns
            min_cols = min(term.shape[1] for term in [term1, term2, term3, term4, term5])
            
            # Truncate all terms to the same number of columns
            term1 = term1[:, :min_cols]
            term2 = term2[:, :min_cols]
            term3 = term3[:, :min_cols]
            term4 = term4[:, :min_cols]
            term5 = term5[:, :min_cols]
            
            print(f"Term shapes after adjustment: {[term.shape for term in [term1, term2, term3, term4, term5]]}")

            # Compute gradients and update parameters
            grad_U_B = term1 + term2 + term3 + term4 + term5
            grad_V_B = -U_B_adjusted @ (error * I_B) + self.config.lambdaP * V_B

            # Update parameters
            U_B_adjusted -= self.config.lr * grad_U_B
            V_B -= self.config.lr * grad_V_B

            # Copy updated parameters back to the original matrices
            for u, idx in u_idx.items():
                if idx < min_cols:  # Ensure index is within valid range
                    self.U[:, self.social_user_map[u]] = U_B_adjusted[:, idx]
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
        
        # Create a new socket for each fold
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind(('localhost', 12345))
            server_socket.listen(1)
            print(f"P0 Server started for fold {k}, waiting for P1 connection...")
            
            # Set timeout to avoid infinite waiting
            server_socket.settimeout(60)  # 60-second timeout
            
            try:
                conn, addr = server_socket.accept()
                print(f"Connected by {addr}")
                
                recommender = Recommender(k, batch_size=64)
                
                with conn:
                    # Handle requests from the client
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
                            rmse_test, ndcg_test = recommender.train_model(conn)
                            rmse_folds.append(rmse_test)
                            ndcg_folds.append(ndcg_test)
                            print(f"Fold {k}: RMSE_test = {rmse_test:.4f}, NDCG@10 = {ndcg_test:.4f}")
                            break
                        else:
                            print(f"Unknown request: {command}")
                
            except socket.timeout:
                print(f"Timeout waiting for connection in fold {k}")
            
        except Exception as e:
            print(f"Error in fold {k}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Ensure socket is closed
            server_socket.close()
            print(f"Server socket for fold {k} closed")
            time.sleep(2)  # Wait 2 seconds to ensure port is fully released
    
    if rmse_folds:
        print(f"Average RMSE: {np.mean(rmse_folds):.4f}, Average NDCG@10: {np.mean(ndcg_folds):.4f}")
    else:
        print("No results collected")