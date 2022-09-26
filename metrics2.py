import numpy as np
import math
import torch
from sklearn.metrics import auc
'''
评测指标：
- 每个指标分开写，会清楚很多
- 要求 input output 都为 np.array

Recall
Precision
AUC
AP
NDCG
MRR
'''


def recall_at_k(binary_gt: np.ndarray, y_pred: np.ndarray, k=float("inf")):
    assert len(binary_gt) == len(y_pred)
    assert type(binary_gt[0]) != np.ndarray
    assert type(y_pred[0]) != np.ndarray
    k = min(int(k), len(y_pred))
    # 排序
    pred_rank_index = np.argsort(y_pred)[::-1][:k]
    binary_gt = binary_gt[pred_rank_index]

    num_pos = np.count_nonzero(binary_gt)
    num_pos = num_pos if num_pos else 1
    recall = np.sum(binary_gt) / num_pos
    return recall


def precision_at_k(binary_gt: np.ndarray, y_pred: np.ndarray, k=float("inf")):
    assert len(binary_gt) == len(y_pred)
    assert type(binary_gt[0]) != np.ndarray
    assert type(y_pred[0]) != np.ndarray
    k = min(int(k), len(y_pred))
    # 排序
    pred_rank_index = np.argsort(y_pred)[::-1][:k]
    binary_gt = binary_gt[pred_rank_index]

    precision = np.sum(binary_gt) / k
    return precision


def ap_at_k(binary_gt: np.ndarray, y_pred: np.ndarray, k=float("inf")):
    assert len(binary_gt) == len(y_pred)
    assert type(binary_gt[0]) != np.ndarray
    assert type(y_pred[0]) != np.ndarray
    k = min(int(k), len(y_pred))
    # 排序
    pred_rank_index = np.argsort(y_pred)[::-1][:k]
    binary_gt = binary_gt[pred_rank_index]

    p_list = (binary_gt * np.cumsum(binary_gt)) / (1 + np.arange(k))
    batch_nonzero = np.count_nonzero(p_list)
    batch_nonzero = max(batch_nonzero, 1)  # 防止分母为0
    ap = np.sum(p_list) / batch_nonzero
    return ap

def auc_at_k(binary_gt: np.ndarray, y_pred: np.ndarray, k=float("inf")):
    assert len(binary_gt) == len(y_pred)
    assert type(binary_gt[0]) != np.ndarray
    assert type(y_pred[0]) != np.ndarray
    k = min(int(k), len(y_pred))
    # 排序
    pred_rank_index = np.argsort(y_pred)[::-1][:k]
    binary_gt = binary_gt[pred_rank_index]

    num_pos = sum(binary_gt)
    num_great_pair = num_pos*(k - 1) - np.sum(binary_gt * np.arange(k)) - num_pos*(num_pos - 1)/2
    num_all_pair = num_pos*(k - num_pos)
    num_all_pair = num_all_pair if num_all_pair else 1  # 考虑只有负样本或只有正样本的情况
    auc = num_great_pair / num_all_pair
    return auc


def ndcg_at_k(binary_gt: np.ndarray, y_pred: np.ndarray, k=float("inf")):
    assert len(binary_gt) == len(y_pred)
    assert type(binary_gt[0]) != np.ndarray
    assert type(y_pred[0]) != np.ndarray
    k = min(int(k), len(y_pred))
    # 排序
    pred_rank_index = np.argsort(y_pred)[::-1][:k]
    binary_gt = binary_gt[pred_rank_index]

    ideal_gt = np.zeros(k)
    ideal_gt[:sum(binary_gt)] = 1
    dcg = np.sum((1 / np.log2(np.arange(k) + 2)) * binary_gt)
    idcg = np.sum((1 / np.log2(np.arange(k) + 2)) * ideal_gt)  # 默认 binary_gt 取值为 1/0
    idcg[idcg == 0.] = 1 # 防止没有正样本时，分母为0
    # print(binary_gt, ideal_gt, dcg, idcg)
    return dcg / idcg


def mrr_at_k(binary_gt: np.ndarray, y_pred: np.ndarray, k=float("inf")):
    '''
    Mean Reciprocal Rank  考虑每个用户topk中的mrr值
    :param binary_gt:
    :param y_pred:
    :param k:
    :return:
    '''
    assert len(binary_gt) == len(y_pred)
    assert type(binary_gt[0]) != np.ndarray
    assert type(y_pred[0]) != np.ndarray
    k = min(int(k), len(y_pred))
    # 排序
    pred_rank_index = np.argsort(y_pred)[::-1][:k]
    binary_gt = binary_gt[pred_rank_index]

    mrr = max(binary_gt / np.arange(1, k + 1), default=0)
    return mrr


def metrics_at_Ks(binary_gt: np.ndarray, y_pred: np.ndarray, Ks=None):  # Ks是一个递增的整数数组
    '''

    :param binary_gt:
    :param y_pred:
    :param Ks:
    :return:
    '''
    if Ks is None:
        Ks = [float("inf")]
    assert len(binary_gt) == len(y_pred)
    assert type(binary_gt[0]) != np.ndarray
    assert type(y_pred[0]) != np.ndarray
    n = len(y_pred)
    # 开辟空间
    recall = np.zeros(len(Ks))
    ap = np.zeros(len(Ks))
    ndcg = np.zeros(len(Ks))
    # 排序
    # pred_rank_index = np.argsort(y_pred)[::-1]
    # binary_gt = binary_gt[pred_rank_index]
    num_pos = np.sum(binary_gt)

    for i, k in enumerate(Ks):
        k = min(int(k), n)
        binary_gt_k = binary_gt[:k]
        # recall
        tmp = num_pos if num_pos else 1   # 考虑没有正样本的情况
        recall[i] = np.sum(binary_gt_k) / tmp
        # ap
        p_list = (binary_gt * np.cumsum(binary_gt)) / (1 + np.arange(k))
        batch_nonzero = np.count_nonzero(p_list)
        batch_nonzero = max(batch_nonzero, 1)  # 防止分母为0
        ap[0] = np.sum(p_list) / batch_nonzero
        # ndcg
        ideal_gt = np.zeros(k)
        ideal_gt[:sum(binary_gt_k)] = 1
        dcg = np.sum((1 / np.log2(np.arange(k) + 2)) * binary_gt_k)
        idcg = np.sum((1 / np.log2(np.arange(k) + 2)) * ideal_gt)  # 默认 binary_gt 取值为 1/0
        ndcg[i] = dcg/idcg if idcg != 0. else 0     # 防止没有正样本时，分母为0
    mrr = max(binary_gt / np.arange(1, n + 1))
    # auc
    num_great_pair = num_pos*(n - 1) - np.sum(binary_gt * np.arange(n)) - num_pos*(num_pos - 1)/2
    num_all_pair = num_pos*(n - num_pos)
    num_all_pair = num_all_pair if num_all_pair else 1  # 考虑只有负样本或只有正样本的情况
    auc = num_great_pair / num_all_pair
    return recall, ap, ndcg, auc, mrr


def batch_recall_at_k(binary_gt: np.ndarray, y_pred: np.ndarray, k=float("inf")):
    assert len(binary_gt) == len(y_pred)
    assert len(binary_gt[0]) == len(y_pred[0])
    k = min(int(k), len(y_pred[0]))

    num_pos_k = binary_gt[:, :k].sum(1)
    num_pos = binary_gt.sum(1)
    num_pos[num_pos == 0.] = 1.       # 考虑没有正样本的特殊情况
    recall = num_pos_k / num_pos
    return np.sum(recall)


def batch_precision_at_k(binary_gt: np.ndarray, y_pred: np.ndarray, k=float("inf")):
    assert len(binary_gt) == len(y_pred)
    assert len(binary_gt[0]) == len(y_pred[0])
    k = min(int(k), len(y_pred[0]))

    num_pos_k = binary_gt[:, :k].sum(1)
    precision = num_pos_k / k
    return np.sum(precision)


def batch_ap_at_k(binary_gt: np.ndarray, y_pred: np.ndarray, k=float("inf")):
    assert len(binary_gt) == len(y_pred)
    assert len(binary_gt[0]) == len(y_pred[0])
    k = min(int(k), len(y_pred[0]))
    binary_gt = binary_gt[:, :k]

    p_list = binary_gt * np.cumsum(binary_gt, axis=1) / (1 + np.arange(k))
    batch_nonzero = np.count_nonzero(p_list, axis=1)
    batch_nonzero[batch_nonzero == 0] = 1
    ap = np.sum(p_list, axis=1) / batch_nonzero
    return np.sum(ap)


def batch_auc_at_k(binary_gt: np.ndarray, y_pred: np.ndarray, k=float("inf")):
    assert len(binary_gt) == len(y_pred)
    assert len(binary_gt[0]) == len(y_pred[0])
    k = min(int(k), len(y_pred[0]))
    binary_gt = binary_gt[:, :k]

    num_pos = np.count_nonzero(binary_gt, axis=1)
    num_great_pair = num_pos*(k - 1) - np.sum(binary_gt * np.arange(k), axis=1) - num_pos*(num_pos - 1)/2
    num_all_pair = num_pos*(k - num_pos)    # 如果只有负样本或只有正样本
    auc = num_great_pair / num_all_pair
    auc[np.isnan(auc)] = 0
    return np.sum(auc)


def batch_ndcg_at_k(binary_gt: np.ndarray, y_pred: np.ndarray, k=float("inf")):
    '''
    Normalized Discounted Cumulative Gain
    '''
    assert len(binary_gt) == len(y_pred)
    assert len(binary_gt[0]) == len(y_pred[0])
    k = min(int(k), len(y_pred[0]))
    batch_size = len(binary_gt)
    binary_gt = binary_gt[:, :k]

    ideal_gt = np.zeros((batch_size, k))
    for i, num_pos in enumerate(np.sum(binary_gt, axis=1)):
        ideal_gt[i, :num_pos] = 1
    dcg = np.sum((1 / np.log2(np.arange(k) + 2)) * binary_gt, axis=1)
    idcg = np.sum((1 / np.log2(np.arange(k) + 2)) * ideal_gt, axis=1)  # 默认 binary_gt 取值为 1/0
    idcg[idcg == 0.] = 1    # 防止没有正样本时，分母为0
    # print(binary_gt, ideal_gt, dcg, idcg)
    return np.sum(dcg / idcg)



def batch_mrr_at_k(binary_gt: np.ndarray, y_pred: np.ndarray, k=float("inf")):
    assert len(binary_gt) == len(y_pred)
    assert len(binary_gt[0]) == len(y_pred[0])
    k = min(int(k), len(y_pred))

    binary_gt = binary_gt[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    mrr = (binary_gt/scores).max(1)
    return np.sum(mrr)




def batch_metrics_at_Ks(binary_gt: np.ndarray, y_pred: np.ndarray, Ks=None):  # Ks是一个递增的整数数组
    '''
    :param binary_gt: shape (test_batch, k), should be a list? cause users may have different amount of pos items.
    :param y_pred: shape (test_batch, k)
    :param Ks: top-k list
    :return: 按 batch 求和之后的结果
    '''
    if Ks is None:
        Ks = [float("inf")]
    assert len(binary_gt) == len(y_pred)
    assert len(binary_gt[0]) == len(y_pred[0])
    num_pos = np.count_nonzero(binary_gt, axis=1)
    batch_size = len(binary_gt)
    num_sample_each_user = len(binary_gt[0])

    # 开辟空间
    recall = np.zeros(len(Ks))
    ap = np.zeros(len(Ks))
    ndcg = np.zeros(len(Ks))

    for i, k in enumerate(Ks):
        k = min(int(k), num_sample_each_user)
        binary_gt_k = binary_gt[:, :k]
        num_pos_k = np.count_nonzero(binary_gt[:, :k], axis=1)
        # recall
        tmp = num_pos.copy()
        tmp[tmp == 0.] = 1                   # 考虑num_pos == 0 的情况
        recall[i] = np.sum(num_pos_k / tmp)
        # ap
        p_list = binary_gt_k * np.cumsum(binary_gt_k, axis=1) / (1 + np.arange(k))
        batch_nonzero = np.count_nonzero(p_list, axis=1)
        batch_nonzero[batch_nonzero == 0] = 1
        ap[i] = np.sum(np.sum(p_list, axis=1) / batch_nonzero)
        # ndcg
        ideal_gt = np.zeros((batch_size, k))
        for j, num_pos_batch_j in enumerate(num_pos):
            ideal_gt[j, :num_pos_batch_j] = 1
        dcg = np.sum((1 / np.log2(np.arange(k) + 2)) * binary_gt_k, axis=1)
        idcg = np.sum((1 / np.log2(np.arange(k) + 2)) * ideal_gt, axis=1)  # 默认 binary_gt 取值为 1/0
        idcg[idcg == 0.] = 1  # 防止没有正样本时，分母为0
        ndcg[i] = np.sum(dcg/idcg)

    mrr = np.sum(np.max(binary_gt / np.arange(1, num_sample_each_user + 1), 1))
    # auc
    num_great_pair = num_pos*(num_sample_each_user - 1) - np.sum(binary_gt * np.arange(num_sample_each_user), axis=1)\
                     - num_pos*(num_pos - 1)/2
    num_all_pair = num_pos*(num_sample_each_user - num_pos)
    num_all_pair[num_all_pair == 0.] = 1        # 考虑只有负样本或只有正样本
    auc = num_great_pair / num_all_pair
    auc = np.sum(auc)
    return recall, ap, ndcg, auc, mrr


if __name__ == "__main__":

    y = np.array([0, 1, 1, 0, 0, 0])
    y_hat = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    # print(recall_at_k(y, y_hat, 3), "true is: ", 1)
    # print(precision_at_k(y, y_hat, 3), "true is: ", 2/3)
    # print(ap_at_k(y, y_hat, 3), "true is: ", 7/12)
    # print(auc_at_k(y, y_hat, 4), "true is: ", 0.5)
    # print(ndcg_at_k(y, y_hat, 3), "true is: ", 0.6934264036172708)
    # print(mrr_at_k(y, y_hat, 3), "true is: ", 1/2)
    # print(metrics_at_Ks(y, y_hat, [1, 2, 3, 4, 10, 15]))

    # batch
    y = np.array([[0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    y_hat = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                      [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
    # print(batch_recall_at_k(y, y_hat, 3), "true is: ", 1)
    # print(precision_at_k(y, y_hat, 3), "true is: ", 2/3)
    # print(ap_at_k(y, y_hat, 3), "true is: ", 7/12)
    # print(batch_auc_at_k(y, y_hat, 4), "true is: ", 0.5)
    # print(batch_ndcg_at_k(y, y_hat, 3), "true is: ", 0.6934264036172708)
    # print(mrr_at_k(y, y_hat, 3), "true is: ", 1/2)
    print(batch_metrics_at_Ks(y, y_hat, [0, 1, 2, 3]))
    pass

