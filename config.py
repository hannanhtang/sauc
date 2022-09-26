from datetime import datetime
from os.path import join

current_time = datetime.now().strftime('%Y%m%d%H%M%S')
ROOT = "/data/tshuang/Projects/sample_sauc/"
config_pre = {
    "config_base": {
        "current_time": current_time,
        "fix_seed": True,
        "seed": 2022,
        "use_gpu": True,
        "cuda": 2,
        "checkpoints_path": join(ROOT, "saved_models"),
        "log_path": "log_gowalla",
        "topks": [2, 4, 8, 10, 20, 50, 100],
    },
    "config_dataset": {
        "dataset_path": join(ROOT, "datasets/"),
        "dataset_name": 'citeulike-a',
        "num_neg": 5,  # 5， 20
        "sampler": "uniform",  # uniform, pop
    },
    "config_model": {
        "model_name": "deepfm",
        "dropout": 0.35,
        "embedding_dim": 8,
        "layer_size1": 32,
        "layer_size2": 8,
        "l2_reg_embedding": 0.0001,
        "l2_reg_dnn": 0.00001,
    },
    "config_loss": {
        "loss": 'sauc_for_user',  # bpr_for_user, sauc_for_user
        "reduction": False,
        "weight_decay": 0.001,
        "tau": 0.02,
        "pos_rank_weighted": False,
    },
    "config_train": {
        "optim": "adam",
        "lr": 0.02,
        "batch_size": 1000,
        "epoch_num": 1000,
        "num_workers": 0,
        "train_way": "per_user",  # per_sample, per_user
        "tqdm_verbose": False,
        "is_use_early_stop": True,
        "early_stopping_patience": 30,
        "only_test": True,
    }
}
config = {}
for key in config_pre.keys():
    config.update(config_pre[key])

important_config_list = [
    "cuda",
    "seed",

    "dataset_name",
    "sampler",

    "model_name",

    "loss",
    "tau",

    "train_way",
    "lr",
    "batch_size",
]

def config_print(config):
    print('===========  config  =================')
    print(f"--------------------------- important config ------")
    for key_name in important_config_list:
        print(f"{key_name}: {config[key_name]}")

    for config_name in config_pre.keys():
        print(f"--------------------------- {config_name} ------")
        for key_name in config_pre[config_name].keys():
            print(f"{key_name}: {config[key_name]}")
    print('===========  end  =================')


if __name__ == '__main__':
    pass


