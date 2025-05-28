#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os

# 强制设置 stdout 编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# 输入文件路径
epinions_ratings_path = "./data/Epinions/ratings.txt"
epinions_trust_path = "./data/Epinions/user_rating.txt"
output_dir = "./data/cv/Epinions/"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取 Epinions 数据
ratings_df = pd.read_csv(epinions_ratings_path, sep="\s+", header=None,
                         names=["OBJECT_ID", "MEMBER_ID", "RATING", "STATUS", "CREATION", 
                                "LAST_MODIFIED", "TYPE", "VERTICAL_ID"])
trust_df = pd.read_csv(epinions_trust_path, sep="\s+", header=None,
                       names=["MY_ID", "OTHER_ID", "VALUE", "CREATION"])

# 数据清洗
# 1. 处理评分数据
ratings_df = ratings_df[["MEMBER_ID", "OBJECT_ID", "RATING"]]
ratings_df["RATING"] = ratings_df["RATING"].apply(lambda x: 5 if x > 5 else x)
ratings_df = ratings_df.dropna()

# 2. 过滤交互次数小于 20 的用户和项目
user_counts = ratings_df["MEMBER_ID"].value_counts()
item_counts = ratings_df["OBJECT_ID"].value_counts()
valid_users = user_counts[user_counts >= 20].index
valid_items = item_counts[item_counts >= 20].index
ratings_df = ratings_df[ratings_df["MEMBER_ID"].isin(valid_users) & 
                        ratings_df["OBJECT_ID"].isin(valid_items)]

# 3. 如果用户数超过 5000，保留交互次数最多的 5000 个用户
if len(valid_users) > 1000:
    top_users = user_counts.loc[valid_users].nlargest(1000).index
    ratings_df = ratings_df[ratings_df["MEMBER_ID"].isin(top_users)]
    valid_users = top_users

print(f"Filtered users: {len(valid_users)}, items: {len(valid_items)}, ratings: {len(ratings_df)}")

# 4. 处理信任数据（只保留信任关系，且用户在 valid_users 中）
trust_df = trust_df[trust_df["VALUE"] == 1][["MY_ID", "OTHER_ID", "VALUE"]]
trust_df = trust_df[trust_df["MY_ID"].isin(valid_users) & trust_df["OTHER_ID"].isin(valid_users)]

# 5. 用户和物品 ID 映射
user_ids = np.unique(np.concatenate([ratings_df["MEMBER_ID"], trust_df["MY_ID"], trust_df["OTHER_ID"]]))
item_ids = np.unique(ratings_df["OBJECT_ID"])
user_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
item_map = {old_id: new_id for new_id, old_id in enumerate(item_ids)}

ratings_df["MEMBER_ID"] = ratings_df["MEMBER_ID"].map(user_map)
ratings_df["OBJECT_ID"] = ratings_df["OBJECT_ID"].map(item_map)
trust_df["MY_ID"] = trust_df["MY_ID"].map(user_map)
trust_df["OTHER_ID"] = trust_df["OTHER_ID"].map(user_map)

# 保存映射关系
np.save("./data/cv/Epinions/user_map.npy", user_map)
np.save("./data/cv/Epinions/item_map.npy", item_map)

# 五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(ratings_df)):
    train_data = ratings_df.iloc[train_idx]
    test_data = ratings_df.iloc[test_idx]
    train_data.to_csv(f"{output_dir}Epinions-{fold}-train.txt", sep="\t", 
                      header=False, index=False, columns=["MEMBER_ID", "OBJECT_ID", "RATING"])
    test_data.to_csv(f"{output_dir}Epinions-{fold}.txt", sep="\t", 
                     header=False, index=False, columns=["MEMBER_ID", "OBJECT_ID", "RATING"])

trust_df.to_csv(f"{output_dir}trust.txt", sep="\t", 
                header=False, index=False, columns=["MY_ID", "OTHER_ID", "VALUE"])

print("数据清洗和五折分割完成！")