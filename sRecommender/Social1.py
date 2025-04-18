"""
文件名称: Social1.py

描述:
    SeSoRec的不安全版本，不包含SSMM协议，直接传输需要的矩阵数据
    对应Recommender1.py
    和 Social2.py的区别是这个版本还没加入五折交叉验证
功能:

用法:
    python Social1.py

作者: chenyuyue
日期: 2025/4/17
"""

# Social.py
# encoding:utf-8
import sys
import socket
import pickle
import numpy as np
from collections import defaultdict
from configx import ConfigX

sys.path.append("..")

def pearson_sp(x1, x2):
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
                u_from, u_to, t = map(float, line.strip().split(self.config.sep))
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

    def init_user_sim(self, sock):
        print("Constructing user-user similarity matrix...")
        for u in self.tg.user:
            for f in self.tg.get_followees(u):
                key = f"{u}-{f}"
                if key not in self.user_sim:
                    send_with_length(sock, ("GET_RATINGS", u))
                    u_ratings = receive_with_length(sock)
                    send_with_length(sock, ("GET_RATINGS", f))
                    f_ratings = receive_with_length(sock)
                    sim = (pearson_sp(u_ratings, f_ratings) + 1.0) / 2.0
                    self.user_sim[key] = sim

    def get_social_terms(self, user, P):
        social_term_p = np.zeros(self.config.factor)
        social_term_m = np.zeros(self.config.factor)
        social_term_loss = 0.0

        followees = self.tg.get_followees(user)
        for followee in followees:
            key = f"{user}-{followee}"
            if key in self.user_sim:
                s = self.user_sim[key]
                uf = P[self.tg.user[followee]]
                p = P[self.tg.user[user]]
                social_term_p += s * (p - uf)
                social_term_loss += s * ((p - uf).dot(p - uf))

        followers = self.tg.get_followers(user)
        for follower in followers:
            key = f"{user}-{follower}"
            if key in self.user_sim:
                s = self.user_sim[key]
                ug = P[self.tg.user[follower]]
                p = P[self.tg.user[user]]
                social_term_m += s * (p - ug)

        return social_term_p, social_term_m, social_term_loss

def client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))
    social = Social()

    with client_socket:
        social.init_user_sim(client_socket)
        send_with_length(client_socket, ("TRAIN", None))
        while True:
            request = receive_with_length(client_socket)
            if not request:
                break
            if request[0] == "GET_SOCIAL":
                user, P = request[1], request[2]
                social_term_p, social_term_m, social_term_loss = social.get_social_terms(user, P)
                send_with_length(client_socket, (social_term_p, social_term_m, social_term_loss))
            elif request[0] == "TRAIN_DONE":
                print("Training completed")
                break

if __name__ == "__main__":
    client()