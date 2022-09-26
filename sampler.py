"""
refer to AdaSIR

给用户序号集，返回对应用户的负样本序号集

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

class base_sampler(nn.Module):
    """
    Uniform sampler
    """

    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super(base_sampler, self).__init__()
        self.num_items = num_items
        self.num_neg = num_neg   # 负样本比例，默认是1比1
        self.device = device

    def update_pool(self, model, **kwargs):
        pass

    def forward(self, u_pos, **kwargs):  # 单个用户的正样本  一维 tensor
        # todo: 增加随机因子
        u_neg = []
        for _ in range(u_pos.shape[0]):  # 99个负样本
            while True:
                neg_idx = random.randint(0, self.num_items - 1)
                if neg_idx not in u_pos:
                    u_neg.append(neg_idx)
                    break
        return torch.IntTensor(u_neg)



all_samplers = {
    "uniform": base_sampler,
    # "pop": base_sampler_pop,
    # "adaptive": adaptive_sampler_per_user,
}


def check_sampler(sampler_name):
    if sampler_name not in all_samplers:
        raise NotImplementedError(f"Haven't supported {sampler_name} yet!, try {all_samplers.keys()}")


def get_sampler(sampler_name, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
    if sampler_name in all_samplers:
        return all_samplers[sampler_name](num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
    else:
        raise NotImplementedError
