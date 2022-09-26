import torch
import torch.nn as nn
import torch.nn.functional as F
from deepctr_torch.models import DeepFM


def get_models(config, fixlen_feature_columns, num_user, num_item):
    if config["model_name"] == "deepfm":
        return DeepFM(linear_feature_columns=fixlen_feature_columns,
                      dnn_feature_columns=fixlen_feature_columns,
                      lr=config["lr"],
                      l2_reg_embedding=config["l2_reg_embedding"],
                      l2_reg_dnn=config["l2_reg_dnn"],
                      device=config["device"],
                      dnn_hidden_units=(config["layer_size1"], config["layer_size2"]),  #
                      # dnn_hidden_units=(),  #
                      dnn_dropout=config["dropout"],
                      dnn_use_bn=True
                      )
    else:
        raise NotImplementedError
