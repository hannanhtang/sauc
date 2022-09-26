
"""
    check_dataset
    check_sampler
    check_model
    check_loss

    Procedure.TRAIN(dataset, model, loss)
    model.load_static_dict(file)
    Procedure.TEST(dataset, model, )

"""
# import sys
# sys.path.append("..")
from datetime import datetime
from utils import logCof, set_seed
from deepctr_torch.inputs import SparseFeat, DenseFeat

from pprint import pprint
from dataloader import get_dataset
from sampler import check_sampler, get_sampler
from Procedure import Procedure
from torch.optim import Adam
from loss import SAUC_for_user
from config import config_print
from loss import getLoss
import torch
from models import get_models
import os


def main(config):
    config["device"] = "cpu" if config["use_gpu"] and not torch.cuda.is_available() else "cuda:" + str(config["cuda"])
    config_print(config)

    set_seed(config["seed"])
    print(">>SEED:", config["seed"])

    check_sampler(config["sampler"])
    # check_model(config["model"])

    dataset = get_dataset(config["dataset_path"], config["dataset_name"],
                          config["sampler"], config["train_way"], config["device"])

    sparse_features = {"userInt": dataset.users_num, "newsInt": dataset.items_num}
    print("sparse_features\n", sparse_features)
    # dense_features = [str(x) for x in range(100)]
    dense_features = []
    fixlen_feature_columns = [SparseFeat(feat, sparse_features[feat], embedding_dim=config["embedding_dim"])
                              for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]
    model = get_models(config, fixlen_feature_columns, dataset.users_num, dataset.items_num)
    model.to(config["device"])
    loss_fun = getLoss(config["loss"], model, config, dataset)
    opt = Adam(model.parameters(), lr=config['lr'])

    # operation
    procedure = Procedure(dataset, model, loss_fun, opt, config)
    procedure.fit(config["epoch_num"], config["batch_size"])
    procedure.test()


if __name__ == "__main__":
    from config import config
    import logging
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    main(config)
