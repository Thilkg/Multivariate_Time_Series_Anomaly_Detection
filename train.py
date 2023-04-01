import json
from datetime import datetime
import torch.nn as nn

from args import get_parser
from utils import *
from GCN_GAT import MTAD_GAT
from prediction import Predictor
from training import Trainer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
if __name__ == "__main__":

    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args() 

    dataset = args.dataset
    window_size = args.lookback    #100
    spec_res = args.spec_res
    normalize = args.normalize  #标准化
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split     
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2:]
    args_summary = str(args.__dict__)
    print(args_summary)   #打印一些参数

    if dataset == 'SMD':
        output_path = f'output/SMD/{args.group}'
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)  
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)  #在这里进行了数据归一化  标准化
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"
# -----------------以上数据归一化数据预处理 ----------
    # print(x_train)

    x_train = torch.from_numpy(x_train.values).float()
    x_test = torch.from_numpy(x_test.values).float()
    # print("------------")
    # print(x_train.shape)  #torch.Size([478880, 12])
    # print("------------")
    n_features = x_train.shape[1]  #12个特征 12维

    target_dims = get_target_dims(dataset)  #返回的是NOne
    # print("------------")
    # print(target_dims) #none
    # print("------------")
    # print(type(target_dims))  #<class 'list'>
    # print("------------")
    if target_dims is None:  #SMD/SMAP
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")

    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)  #滑动窗口数据集
    # print("----------")
    # print(train_dataset)
    # print("----------")
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims) 

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )
    # print("-----------!!!")  
    #train_size: 430902
    #validation_size: 47878
    #test_size: 569595
 
# --------------9.6  从这开始看 ------------------------
#定义模型
    model = GCN_GAT(
        n_features, #输入特征的数量  12
        window_size,#输入序列的长度 默认 100
        out_dim, #要输出的特征数 12
        kernel_size=args.kernel_size,  #在一维卷积中使用的内核大小  7
        use_gatv2=args.use_gatv2,  #是否使用 GATv2 修改后的注意力机制而不是标准 GAT  trUE
        feat_gat_embed_dim=args.feat_gat_embed_dim,  #在面向功能的 GAT 层  默认DOWN
        time_gat_embed_dim=args.time_gat_embed_dim,  #在面向时间的 GAT 层  默认DOWN
        gru_n_layers=args.gru_n_layers,  #等于1    GRU 层的层数
        gru_hid_dim=args.gru_hid_dim,  #150      GRU 层中的隐藏维度
        forecast_n_layers=args.fc_n_layers,  #1  #基于FC的预测模型中的层数
        forecast_hid_dim=args.fc_hid_dim, #150  # 基于FC的预测模型中的隐藏维度
        recon_n_layers=args.recon_n_layers,  #1  #基于 GRU 的重建模型中的层数
        recon_hid_dim=args.recon_hid_dim,  #150  #基于 GRU 的重建模型中的隐藏维度
        dropout=args.dropout,  #0.2   #在全连接层加dropout层，防止模型过拟合
        alpha=args.alpha  #0.2
  
    )
#Adam 是一种可以替代传统随机梯度下降（SGD）过程的一阶优化算法，它能基于训练数据迭代地更新神经网络权重
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)  #lr=args.init_lr 学习率  default=1e-3  0.001
    forecast_criterion = nn.MSELoss()   #均方差损失函数
    # print(forecast_criterion)  #MSELoss()
    recon_criterion = nn.MSELoss()
    # print(recon_criterion) #MSELoss()
    #训练模型
    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,  
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )
    print('-------------')
    trainer.fit(train_loader, val_loader)   #这里
    print('-------------')
    plot_losses(trainer.losses, save_path=save_path, plot=False)

    # Check test loss
    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test reconstruction loss: {test_loss[1]:.5f}")
    print(f"Test total loss: {test_loss[2]:.5f}")

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001)
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]

    trainer.load(f"{save_path}/model.pt")  #都有
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
    }
    best_model = trainer.model
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )
    print("-!!!-----")
    label = y_test[window_size:] if y_test is not None else None
    predictor.predict_anomalies(x_train, x_test, label)

    # Save config  保存配置
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
