
import torch
import torch.nn as nn
from dataloader import MyDataset
from sampler import base_sampler
from deepctr_torch.models import deepfm

class SAUC_for_user(nn.Module):

    def __init__(self, model, config, dataset):
        super().__init__()
        self.model: deepfm = model
        self.config = config
        self.dataset: MyDataset = dataset
        self.sampler: base_sampler = dataset.sampler
        self.mat = dataset.train_mat.tocsr()

    def forward(self, batch_data: torch.int32):
        """

        :param batch_data:一个batch的用户
        :return:
        """
        batch_uid = batch_data[0]
        batch_size = batch_uid.shape[0]
        # 按每个用户，进行负采样，并拼接成为batch_sample
        idx = 0
        u_idx = [0]  # batch内每个用户的正样本数
        all_pos_num = self.mat[batch_uid].sum()
        batch_sample = torch.empty((all_pos_num*2, 2))
        for uid in batch_uid:
            u_pos = torch.from_numpy(self.mat[uid].tocoo().col)
            u_neg = self.sampler(u_pos)
            pos_num = u_pos.shape[0]
            u_sample = torch.cat((torch.Tensor([uid] * 2*pos_num).reshape(-1, 1),
                                 torch.cat((u_pos, u_neg)).reshape(-1, 1)), dim=1)
            batch_sample[idx: idx+2*pos_num] = u_sample
            idx += 2*pos_num
            u_idx.append(idx)

        # to gpu
        batch_sample = batch_sample.to(self.config["device"])
        u_idx = torch.LongTensor(u_idx).to(self.config["device"])
        loss_set = torch.empty((batch_size)).to(self.config["device"])

        batch_scores = self.model(batch_sample)  # shape(batch_sample_num, 1)

        # 按用户计算loss
        for i in range(u_idx.shape[0]-1):
            start, end = u_idx[i], u_idx[i + 1]
            m = torch.div(start + end, 2, rounding_mode="floor")
            scores_pos = batch_scores[start: m, :]
            scores_neg = batch_scores[m: end, :].reshape(1, -1)

            # 验证是否都属于同一个用户
            # temp = None
            # cnt = 0
            # for x in batch_sample[start: end, 0]:
            #     if x != temp:
            #         cnt += 1
            #         temp = x
            # if cnt != 1:
            #     print("assert error")
            # assert cnt == 1

            pred = scores_pos - scores_neg
            sauc_weight = 1 / (m-start)**2
            loss_set[i] = 1 - sauc_weight * torch.sum(torch.sigmoid(pred / self.config["tau"]))

        if self.config["reduction"]:
            loss = loss_set.mean(-1)
        else:
            loss = loss_set.sum(-1)

        reg_loss = self.model.get_regularization_loss()
        return loss + self.config["weight_decay"] * reg_loss



def getLoss(lossName, model, config, dataset):
    if lossName == "bce":
        return BCE(model, config)
    # elif lossName == "bpr":
    #     return BPR(model, config)
    # elif lossName == "bpr_for_sample":
    #     return BPR_for_sample(model, config, dataset)
    # elif lossName == "bpr_for_user":
    #     return BPR_for_user(model, config, dataset)
    # elif lossName == "sauc_for_sample":
    #     return SAUC_for_sample(model, config, dataset)
    if lossName == "sauc_for_user":
        return SAUC_for_user(model, config, dataset)
    else:
        raise NotImplementedError(f"loss {lossName} haven't been implemented!!!")
