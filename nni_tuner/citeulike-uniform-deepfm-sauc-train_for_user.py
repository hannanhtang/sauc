import sys

sys.path.append("..")

from config import config
import nni


def get_default_parameters():
    # 关键参数，包括要调参的参数
    params = \
        {
            # base
            "device": 'cuda:2',
            "seed": 2022,

            # dataset
            "dataset_name": 'citeulike-a',

            # sampler
            "sampler": "uniform",
            "num_neg": 5,  # 5， 20

            # model
            "model_name": "deepfm",
            "dropout": 0.75,
            "embedding_dim": 4,
            "layer_size1": 16,
            "layer_size2": 16,

            # loss
            "tau": 0.02,
            "weighted": False,
            "loss": 'sauc_for_user',


            # train
            "batch_size": 500,
            "num_workers": 0,
            "train_way": "per_user",
            "epoch_num": 5000,
            "is_use_early_stop": True,
            "lr": 0.1,
            "early_stopping_patience": 100,
        }
    return params


if __name__ == "__main__":
    '''
    仅供nni测试使用
    '''
    from main import main

    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        config.update(get_default_parameters())
        config.update(tuner_params)
        # log_file_name = params["project_name"] + "_" + current_time + ".log"
        # logger = logCof(logger, "../log/", log_file_name)
        # logger.info(params)
        main(config)
    except Exception as exception:
        # logger.exception(exception)
        raise
