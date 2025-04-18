"""
文件名称: Recommender1.py

描述:
    SeSoRec的不安全版本，不包含SSMM协议，直接传输需要的矩阵数据
    对应Social1.py
    和 Recommender2.py的区别是这个版本还没加入五折交叉验证
功能:

用法:
    python Recommender1.py

作者: chenyuyue
日期: 2025/4/17
"""
# Recommender.py
# encoding:utf-8
import sys
import socket
import pickle
import numpy as np
from collections import defaultdict
from configx import ConfigX

sys.path.append("..")

def normalize(rating, min_val=0.5, max_val=4.0):
    return (rating - min_val) / (max_val - min_val)

def send_with_length(sock, data):
    serialized_data = pickle.dumps(data)
    length = len(serialized_data)
    sock.sendall(length.to_bytes(8, byteorder='big'))  # 发送 8 字节长度
    sock.sendall(serialized_data)  # 发送实际数据

def receive_with_length(sock):
    length_bytes = sock.recv(8)  # 接收长度
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
        self.trainSet_i = defaultdict(dict)
        self.userMeans = {}
        self.itemMeans = {}
        self.globalMean = 0
        self._load_data()

    def _load_data(self):
        path = f"{self.config.rating_cv_path}{self.config.dataset_name}-{self.k}-train.txt"
        with open(path, 'r') as f:
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
                self.trainSet_i[i][u] = r

        total_rating, total_length = 0.0, 0
        for u in self.user:
            u_total = sum(self.trainSet_u[u].values())
            u_length = len(self.trainSet_u[u])
            total_rating += u_total
            total_length += u_length
            self.userMeans[u] = u_total / u_length
        for i in self.item:
            self.itemMeans[i] = sum(self.trainSet_i[i].values()) / len(self.trainSet_i[i])
        self.globalMean = total_rating / total_length if total_length > 0 else 0

    def get_row(self, u):
        return self.trainSet_u[u]

    def containsUser(self, u):
        return u in self.user

    def containsItem(self, i):
        return i in self.item

    def get_train_size(self):
        return len(self.user), len(self.item)

class Recommender:
    def __init__(self, k):
        self.config = ConfigX()
        self.rg = RatingGetter(k)
        self.P = np.random.rand(self.rg.get_train_size()[0], self.config.factor) / (self.config.factor ** 0.5)
        self.Q = np.random.rand(self.rg.get_train_size()[1], self.config.factor) / (self.config.factor ** 0.5)
        self.loss = 0.0
        self.prev_loss = 0.0

    def predict(self, user, item):
        if self.rg.containsUser(user) and self.rg.containsItem(item):
            return self.P[self.rg.user[user]].dot(self.Q[self.rg.item[item]])
        elif self.rg.containsUser(user):
            return self.rg.userMeans[user]
        elif self.rg.containsItem(item):
            return self.rg.itemMeans[item]
        else:
            return self.rg.globalMean

    def train_model(self, sock):
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            for user in self.rg.trainSet_u:
                for item, rating in self.rg.trainSet_u[user].items():
                    u = self.rg.user[user]
                    i = self.rg.item[item]
                    error = rating - self.predict(user, item)
                    self.loss += 0.5 * error ** 2
                    p, q = self.P[u], self.Q[i]

                    send_with_length(sock, ("GET_SOCIAL", user, self.P))
                    social_term_p, social_term_m, social_term_loss = receive_with_length(sock)

                    update_p = self.config.lr * (
                        error * q - self.config.alpha * (social_term_p + social_term_m) - self.config.lambdaP * p)
                    update_q = self.config.lr * (error * p - self.config.lambdaQ * q)
                    self.P[u] += np.clip(update_p, -1, 1)
                    self.Q[i] += np.clip(update_q, -1, 1)

                    self.loss += 0.5 * self.config.alpha * social_term_loss

            self.loss += 0.5 * self.config.lambdaP * (self.P * self.P).sum() + 0.5 * self.config.lambdaQ * (
                self.Q * self.Q).sum()

            iteration += 1
            print(f"Iteration {iteration}: Loss = {self.loss:.4f}")
            if iteration > 1 and abs(self.loss - self.prev_loss) < self.config.threshold:
                break
            self.prev_loss = self.loss

def server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 12345))
    server_socket.listen(1)
    print("P0 Server started, waiting for P1 connection...")
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    recommender = Recommender(0)
    with conn:
        while True:
            request = receive_with_length(conn)
            if not request:
                break
            if request[0] == "GET_RATINGS":
                user = request[1]
                ratings = recommender.rg.get_row(user)
                send_with_length(conn, ratings)
            elif request[0] == "TRAIN":
                recommender.train_model(conn)
                send_with_length(conn, "TRAIN_DONE")

if __name__ == "__main__":
    server()