
"""

TRAIN(data, model, loss_fun, config):
    model.train()
    for epoch in config["epoch_num"]:
        total_loss = 0
        for batch_idx, batch_data in enumerate(train_data):
            # 负样本的采样，边信息的拼接
            batch_data.to(device)
            loss = loss_fun(batch_data, model)
            loss.backward()
            total_loss += loss
        # verbose
        print(total_loss/sample_num)

TEST():

"""

import time
import math
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import nni
from utils import timer, shuffle
import multiprocessing
from metrics2 import batch_metrics_at_Ks
from dataloader import MyDataset
from utils import EarlyStopping, getFileName
from loss import SAUC_for_user

try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList

CORES = multiprocessing.cpu_count() // 2

class Procedure:
    def __init__(self, dataset, model, loss_fun, opt, config):
        self.dataset = dataset
        self.model = model
        self.loss_fun = loss_fun
        self.opt = opt
        self.config = config
        self.model_path = getFileName(self.config)

    def fit(self, epoch_num, batch_size):

        opt = self.opt
        loss_fun = self.loss_fun
        model = self.model
        dataset = self.dataset
        dataset: MyDataset

        early_stopping = EarlyStopping(patience=self.config["early_stopping_patience"],
                                       verbose=self.config["is_use_early_stop"], path=self.model_path)
        for epoch in range(epoch_num):
            epoch_data = dataset.generate_epoch_data()  # 按用户训练，这儿仅获取uid的数据 TensorDataset
            train_dataloader = DataLoader(epoch_data, batch_size=batch_size,
                                          num_workers=self.config["num_workers"],
                                          pin_memory=True, shuffle=True)
            epoch_loss = 0.0
            print("Epoch %d" % epoch)
            model.train()
            with tqdm(train_dataloader, disable=not self.config["tqdm_verbose"]) as t:
                for batch_data in t:

                    opt.zero_grad()

                    batch_loss = loss_fun(batch_data)
                    batch_loss.backward()
                    epoch_loss += batch_loss.cpu().item()
                    opt.step()

            epoch_loss = epoch_loss / len(train_dataloader)
            print(f"--loss :{epoch_loss:.2f}")

            results = self.val(self.dataset.val_data)
            nni.report_intermediate_result(results['auc'][0])
            early_stopping(results['auc'], model)
            if self.config["is_use_early_stop"] and early_stopping.early_stop:
                print("Early stopping")
                break


    def val_one_batch(self, batch_rating):   # x表示每个batch里的item_score  shape(batch_size, item_num)
        batch_ground_truth = np.zeros_like(batch_rating)
        batch_ground_truth[:, 0] = 1

        pred_rank_index = np.argsort(batch_rating)[:, ::-1]
        batch_ground_truth = batch_ground_truth[np.arange(pred_rank_index.shape[0])[:, None], np.argsort(pred_rank_index)]

        recall, ap, ndcg, auc, mrr = batch_metrics_at_Ks(batch_ground_truth, batch_rating, self.config["topks"])

        return {'recall': recall,
                'ap': ap,
                'ndcg': ndcg,
                'auc': auc,
                'mrr': mrr
                }

    def val(self, data):
        """
        供 验证 和 测试 使用
        :param dataset:
        :param model:
        :return:
        """
        self.model.eval()

        multicore = 0

        if multicore == 1:
            pool = multiprocessing.Pool(CORES)

        tensor_data = Data.TensorDataset(torch.Tensor(range(self.dataset.users_num)))
        test_loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=100, num_workers=0, pin_memory=True)

        pred_ans = []
        with torch.no_grad():
            for _, batch_user in enumerate(test_loader):
                batch_user = batch_user[0]
                items = data.iloc[batch_user, :].values
                batch_user = batch_user.to(self.config["device"])
                items = torch.from_numpy(items).to(self.config["device"])  # u1_pos u1_neg1 ... u1_neg99 | u2_pos ...
                items_length = items.shape[1]  # 每个用户的物品数量 默认为100

                if self.config["model_name"] == "deepfm":
                    # 拼接用户物品
                    x = torch.cat(
                        (batch_user.reshape(-1, 1).repeat(1, items_length).reshape(-1, 1),
                         items.reshape(-1, 1)),
                        dim=1)
                    x = x.float()
                    y_pred = self.model(x).cpu().data.numpy().reshape(-1, items_length)
                else:
                    raise NotImplementedError

                pred_ans.append(y_pred)

            if multicore == 1:
                pre_results = pool.map(self.val_one_batch, pred_ans)
            else:
                pre_results = []
                for x in pred_ans:  # x表示每个batch里的item_score  shape(batch_size, item_num)
                    pre_results.append(self.val_one_batch(x))

        results = {
            'ap': np.zeros(len(self.config["topks"])),
            'recall': np.zeros(len(self.config["topks"])),
            # 'precision': np.zeros(len(self.config["topks"])),
            'ndcg': np.zeros(len(self.config["topks"])),
            'auc': np.zeros(1),
            'mrr': np.zeros(1),
        }

        for batch_result in pre_results:
            results['recall'] += batch_result['recall']
            # results['precision'] += result['precision']
            results['ap'] += batch_result['ap']
            results['ndcg'] += batch_result['ndcg']
            results['auc'] += batch_result['auc']
            results['mrr'] += batch_result['mrr']
        results['recall'] /= float(self.dataset.users_num)
        # results['precision'] /= float(self.dataset.users_num)
        results['ap'] /= float(self.dataset.users_num)
        results['ndcg'] /= float(self.dataset.users_num)
        results['auc'] /= float(self.dataset.users_num)
        results['mrr'] /= float(self.dataset.users_num)

        if multicore == 1:
            pool.close()
        for name in ["ndcg", "recall", "ap", "auc", "mrr"]:
            print(f"{name:6}:", end="  ")
            for res in results[name]:
                print(f"{str(res):9.7}", end="  ")
            print("")
        return results


    def test(self):
        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_path))
        print("============= test =============")
        results = self.val(self.dataset.test_data)
        nni.report_final_result(results["auc"][0])

if __name__ == '__main__':
    pass


