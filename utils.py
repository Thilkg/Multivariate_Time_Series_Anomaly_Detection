import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import sys

def normalize_data(data, scaler=None):  #数据归一化 
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")  
    # print(data)  #这里测试一下  1   都是一些列表数据

    return data, scaler


def get_data_dim(dataset):
    """
    :param dataset: Name of dataset
    :return: Number of dimensions in data
    :param dataset: 数据集的名称
    :return: 数据中的维数
    """
    if dataset == "SMAP":
        return 12 #修改25-》12
    elif dataset == "MSL":
        return 55
    elif str(dataset).startswith("machine"):
        return 38
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_target_dims(dataset):
    """
    :param dataset: Name of dataset
    :return: index of data dimension that should be modeled (forecasted and reconstructed),
                     returns None if all input dimensions should be modeled

    :param dataset: 数据集的名称
    :return: 应该建模（预测和重构）的数据维度的索引，
                      如果所有输入维度都应该建模，则返回 None
    """
    if dataset == "SMAP":
        return None
    elif dataset == "MSL":
        return [0]
    elif dataset == "SMD":
        return None
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_data(dataset, max_train_size=None, max_test_size=None,
             normalize=False, spec_res=False, train_start=0, test_start=0):
    """
    Get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    返回形状：(([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    prefix = "datasets"
    if str(dataset).startswith("machine"):
        prefix += "/ServerMachineDataset/processed"
    elif dataset in ["MSL", "SMAP"]:
        prefix += "/data/processed"
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    train_start=0
    train_end=478880
    test_start=0
    test_end=569695
    print("load data of:", dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)  #12维度
    f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
    # train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]  #reshape()函数可以对数组的结构进行改变。  不知道几行，改成12列
    # f.close()
    train_data=pd.read_csv("datasets/data/SMAP/train/SMAP_train.csv")
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
        # test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        test_data=pd.read_csv("datasets/data/SMAP/test/SMAP_test.csv")
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        # test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        test_label=pd.read_csv('datasets/data/SMAP/test_label/SMAP_test_label.csv')
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None

    if normalize:
        train_data, scaler = normalize_data(train_data, scaler=None)  #数据归一化
        test_data, _ = normalize_data(test_data, scaler=scaler)

    #--------------
    # pd1=train_data
    pd1=pd.read_csv("datasets/data/SMAP/train/SMAP_train.csv")  
    # pd2=test_data
    pd2=pd.read_csv("datasets/data/SMAP/test/SMAP_test.csv")
    # pd3=test_label
    pd3=pd.read_csv('datasets/data/SMAP/test_label/SMAP_test_label.csv')
    #------------

    print("train set shape: ", pd1.shape)
    print("test set shape: ", pd2.shape)
    print("test set label shape: ", None if pd3 is None else pd3.shape)
    return (pd1, None), (pd2, pd3)


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim 
        self.horizon = horizon  # 预测窗口大小

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window


def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):  #数据
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
        # print("-----------!!!!")
        print(f"train_size: {len(train_indices)}")  #430902
        print(f"validation_size: {len(val_indices)}")   #47878  验证

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")  #569595

    return train_loader, val_loader, test_loader


def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    :param loss: 有损失的字典
     :param save_path: 保存绘图的路径
    """

    plt.plot(losses["train_forecast"], label="Forecast loss")
    plt.plot(losses["train_recon"], label="Recon loss")
    plt.plot(losses["train_total"], label="Total loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_forecast"], label="Forecast loss")
    plt.plot(losses["val_recon"], label="Recon loss")
    plt.plot(losses["val_total"], label="Total loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()


def load(model, PATH, device="cpu"):
    """
    Loads the model's parameters from the path mentioned
    :param PATH: Should contain pickle file
    """
    model.load_state_dict(torch.load(PATH, map_location=device))


def get_series_color(y):
    if np.average(y) >= 0.95:
        return "black"
    elif np.average(y) == 0.0:
        return "black"
    else:
        return "black"


def get_y_height(y):
    if np.average(y) >= 0.95:
        return 1.5
    elif np.average(y) == 0.0:
        return 0.1
    else:
        return max(y) + 0.1


def adjust_anomaly_scores(scores, dataset, is_train, lookback):
    """
    Method for MSL and SMAP where channels have been concatenated as part of the preprocessing
    :param scores: anomaly_scores
    :param dataset: name of dataset
    :param is_train: if scores is from train set
    :param lookback: lookback (window size) used in model
    在预处理过程中连接通道的 MSL 和 SMAP 方法
     :param 分数: anomaly_scores
     :param dataset: 数据集名称
     :param is_train: 如果分数来自训练集
     :param lookback: 模型中使用的回溯（窗口大小）
    """

    # Remove errors for time steps when transition to new channel (as this will be impossible for model to predict)
    #消除转换到新通道时的时间步错误（因为模型无法预测）
    if dataset.upper() not in ['SMAP', 'MSL']:  #.upper()是大写字母的意思
        return scores

    adjusted_scores = scores.copy()  #预测来的scores
    if is_train:
        md = pd.read_csv(f'./datasets/data/2{dataset}_train_md.csv')  #读取数据集  训练集无标签
    else:
        md = pd.read_csv('./datasets/data/{dataset.lower()}/test_label/SMAP_test_label.csv')  #测试集  有标签
        # md = md[md['spacecraft'] == dataset.upper()]

    # md = md[md['chan_id'] != 'P-2']

    # Sort values by channel 按通道对值进行排序
    md = md.sort_values(by=['chan_id'])

    # Getting the cumulative start index for each channel
    #获取每个通道的累积起始索引
    sep_cuma = np.cumsum(md['num_values'].values) - lookback  
    sep_cuma = sep_cuma[:-1]  #从位置0到位置-1之前的数,除了最后一个数
    buffer = np.arange(1, 20)
    i_remov = np.sort(np.concatenate((sep_cuma, np.array([i+buffer for i in sep_cuma]).flatten(),
                                      np.array([i-buffer for i in sep_cuma]).flatten())))
    i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
    i_remov = np.sort(np.unique(i_remov))
    if len(i_remov) != 0:
        adjusted_scores[i_remov] = 0

    # Normalize each concatenated part individually
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    s = [0] + sep_cuma.tolist()
    for c_start, c_end in [(s[i], s[i+1]) for i in range(len(s)-1)]:
        e_s = adjusted_scores[c_start: c_end+1]

        e_s = (e_s - np.min(e_s))/(np.max(e_s) - np.min(e_s))
        adjusted_scores[c_start: c_end+1] = e_s

    return adjusted_scores