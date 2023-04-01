###测试

# import torch
# from torch import nn
# print(torch.cuda.is_available()) # true 查看GPU是否可用
# print(torch.cuda.device_count()) #GPU数量， 4
# torch.cuda.current_device() #当前GPU的索引， 0
# torch.cuda.get_device_name(0) #输出GPU名称

import json
from datetime import datetime
import torch.nn as nn

from args import get_parser
from utils import *
from GCN_GAT import MTAD_GAT
from prediction import Predictor
from training import Trainer
import os


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

    adjusted_scores = scores.copy()
    if is_train:
        md = pd.read_csv(f'./datasets/data/{dataset.lower()}_train_md.csv')
    else:
        md = pd.read_csv('./datasets/data/labeled_anomalies.csv')
        md = md[md['spacecraft'] == dataset.upper()]

    # md = md[md['chan_id'] != 'P-2']

    # Sort values by channel
    md = md.sort_values(by=['chan_id'])

    # Getting the cumulative start index for each channel
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    sep_cuma = sep_cuma[:-1]
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


predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )
predictor.predict_anomalies(x_train, x_test, label)


