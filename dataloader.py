
"""

    - 读取数据集
    - 生成验证集（包括验证集的采样）
    - 包含了采样器，用于每个epoch生成相应的训练数据。
    -
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sp
from config import config
from time import time
import scipy.io as spio
import json
import random
from sampler import get_sampler
# from interface import baseDataset
from tqdm import tqdm
import torch.utils.data as Data

class MyDataset:
    def __init__(self, dataset_path="", dataset_name="gowalla", sampler="uniform", train_way="per_user", device=None):
        """
        """
        super(MyDataset, self).__init__()
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.train_way = train_way
        self.device = device

        self.train_mat, self.val_data = self.get_train_val_data()
        # meta_info
        self.users_num, self.items_num = self.train_mat.shape
        self.has_sideinfo, self.users_info, self.items_info = self.get_sideinfo()

        # self.all_pos = self.get_all_user_pos(self.users_num, self.train_mat)   # list版 的用户正物品列表
        self.test_data = self.get_test_data()
        self.sampler = get_sampler(config["sampler"], self.users_num, self.items_num, 0, 0, config["num_neg"], device,
                                   mat=self.train_mat)
        self.dataset_print()

        self.X = None  # 用于测试BCE
        self.y = None

    def dataset_print(self):
        print("dataset_name: ", self.dataset_name)
        print("users_num: ", self.users_num)
        print("items_num: ", self.items_num)
        print("has_sideinfo", self.has_sideinfo)

    def get_train_val_data(self) -> [csr_matrix, pd.DataFrame]:
        """
        用于将train 划分为  train 和 val。
        其中val需要进行负采样采样，并缓存。
        """
        try:
            train_mat = spio.loadmat(os.path.join(self.dataset_path, self.dataset_name, "_cache_train.mat"))["train_mat"]
            val_data = pd.read_csv(os.path.join(self.dataset_path, self.dataset_name, "_cache_val.csv"))
            return train_mat, val_data
        except:
            mat = spio.loadmat(os.path.join(self.dataset_path, self.dataset_name, "train.mat"))['train_mat']
            mat = mat.tocsr()  # 按行读取，即每一行为一个用户
            m, n = mat.shape
            train_indices_data = []
            train_indptr = [0] * (m + 1)
            # val_indices_data = []
            # val_indptr = [0] * (m + 1)
            val_data = []
            with tqdm(range(m)) as temp:
                for u in temp:
                    # 拆分每行
                    u_pos_list = mat[u].nonzero()[1]    # 非零列坐标
                    # 采一个正样本用于验证
                    val_idx = random.sample(range(len(u_pos_list)), 1)
                    u_val_data = [u_pos_list[val_idx[0]]]
                    for _ in range(99):  # 99个负样本
                        while True:
                            neg_idx = random.randint(0, n - 1)
                            if neg_idx not in u_pos_list:
                                u_val_data.append(neg_idx)
                                break
                    val_data.append(u_val_data)

                    val_binary_idx = np.full(len(u_pos_list), False)
                    val_binary_idx[val_idx] = True
                    train_idx = (~val_binary_idx).nonzero()[0]  # 这一行啥意思？
                    # 拼接成csr
                    for idx in train_idx:
                        train_indices_data.append(u_pos_list[idx])
                    train_indptr[u + 1] = len(train_indices_data)

            train_indices = train_indices_data
            train_mat = csr_matrix(([1] * len(train_indices), train_indices, train_indptr), (m, n))
            val_data = pd.DataFrame(np.array(val_data))

            assert val_data.shape[0] == m
            assert val_data.shape[1] == 100
            # test
            # for i in range(val_data.shape[0]):
            #     assert val_data.iloc[i, 0] in mat[i].A

            # save
            spio.savemat(os.path.join(self.dataset_path, self.dataset_name, "_cache_train.mat"),
                         {"train_mat": train_mat})
            val_data.to_csv(os.path.join(self.dataset_path, self.dataset_name, "_cache_val.csv"), index=False)
            del mat
            return train_mat, val_data

    def get_sideinfo(self):
        try:
            users_info = pd.read_csv(os.path.join(self.dataset_path, self.dataset_name, "users_info.csv"))
            items_info = pd.read_csv(os.path.join(self.dataset_path, self.dataset_name, "news_info.csv"))
            users_info = users_info[["userInt", "phoneName", "OS", "province", "city", "age", "gender"]]
            items_info = items_info[["newsInt"]]
            return True, users_info, items_info
        except:
            return False, None, None

    def get_test_data(self):
        test_data = pd.read_csv(os.path.join(self.dataset_path, self.dataset_name, "test.csv"))
        return test_data

    def generate_epoch_data(self):
        """
        用于在每个epoch，获得负采样之后的训练样本，包括三种形式：
        - (u, i, label)
        - (u, i, j)
        - (u, i1, i2, .., j1, j2, ..)
        """
        if self.train_way == "per_user":
            print("sample_num: ", self.users_num)
            return Data.TensorDataset(torch.IntTensor(range(self.users_num)))
        else:
            raise NotImplementedError


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")


def get_dataset(dataset_path, dataset_name, sampler, train_way, device="cpu"):
    if dataset_name in ['citeulike-a', "gowalla", "yidian"]:
        return MyDataset(dataset_path, dataset_name, sampler, train_way=train_way, device=device)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    pass
