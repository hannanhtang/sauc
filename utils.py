
import os
import argparse
import pickle
from collections import deque
import random
from time import time
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
import logging
# from metrics import AUC_LOO, AP_MRR
# from metrics import normalized_discounted_cumulative_gain_matrix as NDCG


def logCof(logger, log_path="./log/", log_file_name="test.log"):
    '''
        CRITICAL, ERROR, WARNING, INFO, DEBUG
        从左到右，日志级别一次下降
        打印的时候只会打印比设置的级别高的日志
        logger设置的级别是最低级别，它的handler比他低也没用，不会输出更低的

    '''
    logger.setLevel(level=logging.DEBUG)

    if log_file_name == "test.log":
        log_file_name = "test_" + datetime.now().strftime('%Y%m%d%H%M%S') + ".log"
    os.makedirs(log_path, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_path, log_file_name))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # console = logging.StreamHandler()
    # console.setLevel(logging.WARNING)
    # console.setFormatter(formatter)

    logger.addHandler(handler)
    # logger.addHandler(console)
    return logger



class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}s|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}s|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(int(timer.time() - self.start))



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)



def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def getFileName(config):
    if not os.path.exists(config["checkpoints_path"]):
        os.makedirs(config["checkpoints_path"], exist_ok=True)
    dataset_name = config['dataset_name']
    sample_way = config["sampler"]
    model_name = config["model_name"]
    loss = config["loss"]
    current_time = config["current_time"]
    # if config["model"] == 'mf':
    #     model_name = "mf"
    # elif config["model"] == 'lgn':
    #     model_name = f"lgn-{config['dataset']}-{sample_way}-{loss}-{config['lightGCN_n_layers']}-{config['latent_dim_rec']}.pth.tar"
    # elif config["model"] == 'lgn_hash':
    #     model_name = f"lgn_hash-{config['dataset']}-{sample_way}-{loss}-{config['lightGCN_n_layers']}-{config['latent_dim_rec']}.pth.tar"
    # else:
    #     model_name = f"othermodel-{config['dataset']}.pth.tar"
    file_name = "-".join([dataset_name, sample_way, model_name, loss, current_time]) + ".pth.tar"
    return os.path.join(config["checkpoints_path"], file_name)


class EarlyStopping:
    '''
    Early stop the training if validation metric doesn't improve after
    a given patience
    from: https://github.com/shaheerzaman/Earlystopping_Pytorch/blob/master/pytool.py

    Example:
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=5, verbose=True)
        for epoch in xxx:
            xxx
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
    '''

    def __init__(self, patience=7, verbose=False,
                 delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, metric, model):
        score = metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''
        save model when validation loss decrease
        '''
        # if self.verbose:
        #     print(f'validation loss decrease ({self.val_loss_min:.6f})')
        # print("model_print", model)
        torch.save(model.state_dict(), self.path)
        # torch.save(model, self.path)