# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))  # 设置图像大小
names = ['5', '10', '30', '50', '80', '100']
x = range(len(names))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字

plt.ylim(0.5, 1.0)  # 设置x轴的范围
y_1 = [0.70, 0.840, 0.730, 0.721, 0.720, 0.711]
y_2 = [0.84, 0.855, 0.901, 0.899, 0.897, 0.902]   
# y_3 = [4, 5, 6, 1, 2, 3]



plt.rcParams['savefig.dpi'] = 800 #图片像素
# #
plt.rcParams['figure.dpi'] = 800 #分辨率
# #
# # # plt.axis('off')可以去坐标轴


plt.plot(x, y_1, color='blue', marker='o', linestyle='-', label='OmniAnomaly')
plt.plot(x, y_2, color='red', marker='D', linestyle='-', label='Ous')
# plt.plot(x, y_3, color='green', marker='*', linestyle=':', label='C')
plt.legend()  # 显示图例
plt.legend(fontsize=13)
plt.xticks(x, names, rotation=45,fontsize=13)
plt.xlabel("window sizes",fontproperties='Times New Roman',fontsize=15)  # X轴标签
plt.ylabel("F1 scores",fontproperties='Times New Roman',fontsize=15)  # Y轴标签
plt.title('F1 scores of different models with different sliding window sizes', fontproperties='Times New Roman',fontsize=15)
plt.savefig("1")
plt.show()

# plt.figure(dpi=300,figsize=(24,8))
# # 改变文字大小参数-fontsize
# plt.xticks(fontsize=10)


# #画图2
# # -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt
# plt.figure(figsize=(7, 5))  # 设置图像大小
# names = ['CBLOF', 'FBagging', 'KNN', 'HBOS', 'IForest', 'OCSVM','LOF','MCD','Ours']
# x = range(len(names))
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字

# plt.ylim(0.0, 1.0)  # 设置x轴的范围
# y_1 = [0.425, 0.424, 0.431, 0.397, 0.413, 0.421,0.426,0.422,0.877]
# y_2 = [0.900, 0.899, 0.902, 0.896, 0.895, 0.899,0.900,0.901,0.822]   
# y_3 = [0.277, 0.279, 0.284, 0.258, 0.270, 0.276,0.281,0.276,0.940]

# plt.plot(x, y_1, color='blue', marker='o', linestyle='-', label='F1-scores')
# plt.plot(x, y_2, color='red', marker='D', linestyle='-', label='Precision')
# plt.plot(x, y_3, color='green', marker='^', linestyle='-', label='Recall')
# plt.legend()  # 显示图例
# plt.legend(fontsize=11)
# plt.xticks(x, names, rotation=45,fontsize=11)
# plt.xlabel("window sizes",fontsize=13)  # X轴标签
# plt.ylabel("F1 scores",fontsize=13)  # Y轴标签
# plt.title('Comparison with traditional methods', fontsize=12)
# plt.savefig("2")
# plt.show()

# import json
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# f1_1 = [0.843,0.813,0.729,0.877]
# f1_2 = [0.942,0.805,0.856,0.822]
# f1_3 = [0.978,0.821,0.643,0.940]
 
# x = ['OmniAnomaly','MAD-GAN','LSTM-VAE','Ours']
# x_len = np.arange(len(x))
# total_width, n = 0.8, 4
# width = 0.2
# xticks = x_len - (total_width - width) / 2
# plt.figure(figsize=(7, 5))
 
# ax = plt.axes()
# plt.grid(axis="y", c='#d2c9eb', linestyle = '--',zorder=0)
# plt.bar(xticks, f1_1, width=0.8*width, label="F1-score", color="#7e728c",linewidth = 2,  zorder=10)
# plt.bar(xticks + width, f1_2, width=0.8*width, label="Precision", color="#60c49a",linewidth = 2, zorder=10)
# plt.bar(xticks + width + width, f1_3, width=0.8*width, label="Recall", color="#c48f60",linewidth = 2, zorder=10)

# #数字
# # plt.text(xticks[0], f1_1[0] + 0.2, f1_1[0], ha='center',fontproperties='Times New Roman',  fontsize=9,  zorder=5)
# # plt.text(xticks[1], f1_1[1] + 0.2, f1_1[1], ha='center', fontproperties='Times New Roman', fontsize=9,  zorder=5)
# # plt.text(xticks[2], f1_1[2] + 0.2, f1_1[2], ha='center', fontproperties='Times New Roman', fontsize=9,  zorder=5)
# # plt.text(xticks[3], f1_1[3] + 0.2, f1_1[3], ha='center', fontproperties='Times New Roman', fontsize=9,  zorder=5)

 
# # plt.text(xticks[0] + width, f1_2[0] + 0.2, f1_2[0], ha='center',fontproperties='Times New Roman', fontsize=9,  zorder=5)
# # plt.text(xticks[1] + width, f1_2[1] + 0.2, f1_2[1], ha='center',fontproperties='Times New Roman', fontsize=9,  zorder=5)
# # plt.text(xticks[2] + width, f1_2[2] + 0.2, f1_2[2], ha='center', fontproperties='Times New Roman',fontsize=9,  zorder=5)
# # plt.text(xticks[3] + width, f1_2[3] + 0.2, f1_2[3], ha='center', fontproperties='Times New Roman',fontsize=9,  zorder=5)

 
# # plt.text(xticks[0] + width + width, f1_2[0] + 0.2, f1_2[0], ha='center',fontproperties='Times New Roman', fontsize=9,  zorder=5)
# # plt.text(xticks[1] + width + width, f1_2[1] + 0.2, f1_2[1], ha='center',fontproperties='Times New Roman', fontsize=9,  zorder=5)
# # plt.text(xticks[2] + width + width, f1_2[2] + 0.2, f1_2[2], ha='center', fontproperties='Times New Roman',fontsize=9,  zorder=5)
# # plt.text(xticks[3] + width + width, f1_2[3] + 0.2, f1_2[3], ha='center', fontproperties='Times New Roman',fontsize=9,  zorder=5)

# plt.legend(prop={'family' : 'Times New Roman', 'size': 10}, ncol = 2)
# x_len = [-0.1,0.9,1.9,2.9]
# x_len = np.array(x_len)
# plt.xticks(x_len, x, fontproperties='Times New Roman',fontsize = 10)
# plt.yticks(fontproperties='Times New Roman',fontsize = 10)
# plt.ylim(ymin=0.5)
# plt.xlabel("Methods", fontproperties='Times New Roman',fontsize=15)
# plt.ylabel("Value",fontproperties='Times New Roman', fontsize=15)

# # ax.spines['bottom'].set_linewidth('1.0')#设置边框线宽为2.01
# # # ax.spines['bottom'].set_color('black')
# # ax.spines['top'].set_linewidth('2.0')#设置边框线宽为2.0
# # # ax.spines['top'].set_color('black')
# # ax.spines['right'].set_linewidth('2.0')#设置边框线宽为2.0
# # # ax.spines['right'].set_color('black')
# # ax.spines['left'].set_linewidth('2.0')#设置边框线宽为2.0
# # # ax.spines['left'].set_color('black')

# plt.title('Comparison with classical methods', fontsize=12)
# plt.savefig("3")

# plt.show()
 