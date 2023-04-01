from ast import literal_eval
from csv import reader
from os import listdir, makedirs, path
from pickle import dump
import numpy as np
import pandas as pd

from args import get_parser


def load_and_save(category, filename, dataset, dataset_folder, output_folder):
    temp = np.genfromtxt(
        path.join(dataset_folder, category, filename),
        dtype=np.float32,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        # 将obj对象序列化存入已经打开的file中。
        dump(temp, file)



def load_data(dataset):
    """ Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly) """

    if dataset == "SMD":
        dataset_folder = "datasets/ServerMachineDataset"
        output_folder = "datasets/ServerMachineDataset/processed"
        makedirs(output_folder, exist_ok=True)
        file_list = listdir(path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith(".txt"):
                load_and_save(
                    "train",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test_label",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )

    elif dataset == "SMAP" or dataset == "MSL":
        dataset_folder = "datasets/data/SMAP/test_label"
        output_folder = "datasets/data/processed"
        makedirs(output_folder, exist_ok=True)
        # with open(path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
        with open(path.join(dataset_folder, "SMAP_test_label.csv"), "r") as file:  #这里是测试集的标签 应该是1048575 个

            csv_reader = reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
            # print(res)
        # res = sorted(res, key=lambda k: k[0])
        # print(res)
        data_info = []
        data_info=res
        # print(data_info)
        labels = [ True if row=='1' else False for row in res  ]
        # print(labels)
        # for row in data_info:
        #     # 判断需要计算的内容计算后是不是合法的 python 类型，如果是则进行运算，否则就不进行运算
        #     # 数据结构转换，将字符串转换为原来的2维列表
        #     # anomalies = literal_eval(row[2])
        #     # print(anomalies)
        #     # 异常值
        #     # length = int(row[-1])
            
        #     # label = np.zeros([length], dtype=np.bool_)
             
        #     # for anomaly in anomalies:
        #     #     label[anomaly[0] : anomaly[1] + 1] = True
            
        #     labels.extend(label)
        labels = np.asarray(labels)
        # print(labels)  #弄成列表形式

        print(dataset, "test_label", labels.shape)
        # print(data_info)  #还是个空的
        with open(path.join(output_folder, dataset + "_" + "test_label" + ".pkl"), "wb") as file:
            # 将obj对象序列化存入已经打开的file中。
            dump(labels, file)    #存入processed中

        # def concatenate_and_save(category):
        #     data = []
        #     for row in data_info:
        #         filename = row[0]
        #         temp = np.load(path.join(dataset_folder, category, filename + ".npy"))
        #         data.extend(temp)
        #     data = np.asarray(data)
        #     print(dataset, category, data.shape)
        #     with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        #         dump(data, file)

        # for c in ["train", "test"]:
        #     concatenate_and_save(c)
        pd1=pd.read_csv("datasets/data/SMAP/train/SMAP_train.csv")
        print('SMAP', 'train', pd1.shape)
        with open(path.join(output_folder, 'SMAP' + "_" + "train" + ".pkl"), "wb") as file:
                dump(pd1, file)
        pd2=pd.read_csv("datasets/data/SMAP/test/SMAP_test.csv")
        # print(len(pd2))
        # pd2=pd2*0.5
        # print(pd2)
        print('SMAP', 'test', pd2.shape)
        with open(path.join(output_folder, 'SMAP' + "_" + "test" + ".pkl"), "wb") as file:
                dump(pd2, file)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()  #读SMAP
    ds = args.dataset.upper()  #字符串放大：
    # print(ds)  # SMAP
    load_data(ds)
