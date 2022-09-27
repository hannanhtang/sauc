import os
import random
import pickle
import argparse
from time import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def main(args):

    path = "./"
    t_start = t = time()
    train_cols = ["uid", "iid"]
    items_cols = ["iid"] + [str(x) for x in range(100)]
    data = pd.read_csv(os.path.join(path, "raw/citeulike.csv"), names=train_cols)
    print(f"加载交互数据完成。cost: {time() - t:.2f}s, total cost:{time() - t_start:.2f}s.")
    t = time()

    random.seed(2022)

    # 提取交互数量多于20的用户的所有交互
    d = data.uid.value_counts()
    users = d[d > 20].keys().tolist()
    df = data.set_index("uid").loc[users, :].reset_index()
    del data
    print(f"删除交互数量少于20的用户，完成！！！ cost: {time() - t:.2f}s, total cost:{time() - t_start:.2f}s.")
    t = time()
    usersIdSet = df.uid.unique()
    users_size = max(usersIdSet)
    itemsIdSet = df.iid.unique()
    items_size = max(itemsIdSet)
    interact_size = df.shape[0]

    items_info = pd.read_csv(os.path.join(path, "raw/feature_citeulike.txt"), names=items_cols)
    print("*****************数据组合完毕，接下来进行归一化**********************")
    mms = MinMaxScaler(feature_range=(0, 1))
    items_info[items_cols] = mms.fit_transform(items_info[items_cols])
    items_info.to_csv("items_info.csv", index=False)
    del items_info
    print(f"保留有交互的物品， 完成！！！ cost: {time() - t:.2f}s, total cost:{time() - t_start:.2f}s.")
    t = time()

    label = pd.DataFrame([1]*df.shape[0], columns=["label"])
    df = pd.concat([df, label], axis=1)

    # 造负样本
    neg_uid_set = []
    neg_iid_set = []
    print("total iterator nums: ", len(usersIdSet))
    with tqdm(enumerate(usersIdSet)) as temp:
        for _, uid in temp:
            pos_data = df[df.uid == uid].iid.values
            neg_data = list(set(range(items_size)) - set(pos_data))
            random.shuffle(neg_data)
            pos_len = len(pos_data)
            neg_data = neg_data[:200]  # 100 + 100
            neg_uid_set.extend([uid]*len(neg_data))
            neg_iid_set.extend(neg_data)
    df_neg = pd.DataFrame(list(zip(neg_uid_set, neg_iid_set, [0]*len(neg_iid_set))), columns=["uid", "iid", "label"])
    df = pd.concat([df, df_neg], axis=0)  # 正负样本集合 上下拼接
    del df_neg
    df.reset_index(drop=True, inplace=True)  # 负样本的index是乱的
    print(f"测试集和验证集的负样本采样完成！！！ cost: {time() - t:.2f}s, total cost:{time() - t_start:.2f}s.")
    t = time()
    # 获取三种训练集划分，验证集和测试集的划分
    train_data1_index, train_data2_index, train_data3_user_list = [], [], []
    val_data_index, test_data_index = [], []
    start = 0
    for uid in usersIdSet:
        u_label = df.label[df.uid == uid]
        pos_index = u_label[u_label == 1].index.tolist()
        neg_index = u_label[u_label == 0].index.tolist()
        assert len(pos_index) >= 3, "val 和 test 至少有1个正样本"
        assert len(neg_index) == 200

        # leave-one-out 划分正样本
        random.shuffle(pos_index)
        train_pos_index = pos_index[:-2]
        val_pos_index = [pos_index[-2]]
        test_pos_index = [pos_index[-1]]
        train_pos_len = len(train_pos_index)

        # 划分负样本
        random.shuffle(neg_index)
        neg_len = len(neg_index)

        # train_neg_index = neg_index[200:]
        val_neg_index = neg_index[:100]
        test_neg_index = neg_index[100:]
        train_data1_index.extend(train_pos_index)          # train1  pos_set
        for i in range(len(train_pos_index)):
            train_data2_index.extend([train_pos_index[i]])  # train2 (pos)

        train_data3_user_list.append((uid, start, start + train_pos_len))  # uid, start, end
        start += train_pos_len

        val_data_index.extend(val_pos_index + val_neg_index)
        test_data_index.extend(test_pos_index + test_neg_index)
        assert not len(val_data_index) % 101
        assert not len(test_data_index) % 101
    assert start == len(train_data1_index)
    print(f"交互数据的index划分完成！！！ cost: {time() - t:.2f}s, total cost:{time() - t_start:.2f}s.")
    t = time()

    # meta-json
    meta_info = {"user_size": users_size, "item_size": items_size, "interact_size": interact_size}
    with open('meta_info.pickle', 'wb') as f:
        pickle.dump(meta_info, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    # train1
    df.iloc[train_data1_index, :].to_csv("train_data1.csv", index=False)
    print(f"train1完成 cost: {time() - t:.2f}s, total cost:{time() - t_start:.2f}s.")
    t = time()

    # train2
    df.iloc[train_data2_index, :].to_csv("train_data2.csv", index=False)
    print(f"train2完成 cost: {time() - t:.2f}s, total cost:{time() - t_start:.2f}s.")
    t = time()

    # train3
    dataset3 = {'train_data3_user_list': train_data3_user_list}
    with open('train3.pickle', 'wb') as f:
        pickle.dump(dataset3, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    del dataset3
    print(f"train3完成 cost: {time() - t:.2f}s, total cost:{time() - t_start:.2f}s.")
    t = time()

    # val
    df.iloc[val_data_index, :].to_csv("val_data.csv", index=False)
    print(f"val完成 cost: {time() - t:.2f}s, total cost:{time() - t_start:.2f}s.")
    t = time()

    # test
    df.iloc[test_data_index, :].to_csv("test_data.csv", index=False)
    print(f"test完成 cost: {time() - t:.2f}s, total cost:{time() - t_start:.2f}s.")

if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset',
    #                     type=str,
    #                     default='citeulike')
    # parser.add_argument('--data_dir',
    #                     type=str,
    #                     default='/data/Datasets/CiteUlike/Origin/',
    #                     help="File path for raw data")
    # parser.add_argument('--outputdir',
    #                     type=str,
    #                     default='./',
    #                     help="Proportion for training and testing split")
    args = parser.parse_args()
    main(args)

