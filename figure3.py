import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f1_1 = [0.843,0.813,0.729,0.852,0.877]
f1_2 = [0.942,0.805,0.856,0.959,0.822]
f1_3 = [0.978,0.821,0.643,0.696,0.940]
 
x = ['OmniAnomaly','MAD-GAN','LSTM-VAE','GDN','Ours']
x_len = np.arange(len(x))
total_width, n = 0.8, 5
width = 0.2
xticks = x_len - (total_width - width) / 2
plt.figure(dpi=300,figsize=(10, 8)) 
 
plt.rcParams['savefig.dpi'] = 800 #图片像素
# #
plt.rcParams['figure.dpi'] = 800 #分辨率
ax = plt.axes()
plt.grid(axis="y", c='#d2c9eb', linestyle = '--',zorder=0)
plt.bar(xticks, f1_1, width=0.8*width, label="F1-score", color="#7e728c",linewidth = 2,  zorder=10)
plt.bar(xticks + width, f1_2, width=0.8*width, label="Precision", color="#60c49a",linewidth = 2, zorder=10)
plt.bar(xticks + width + width, f1_3, width=0.8*width, label="Recall", color="#c48f60",linewidth = 2, zorder=10)


#数字
# plt.text(xticks[0], f1_1[0] + 0.2, f1_1[0], ha='center',fontproperties='Times New Roman',  fontsize=9,  zorder=5)
# plt.text(xticks[1], f1_1[1] + 0.2, f1_1[1], ha='center', fontproperties='Times New Roman', fontsize=9,  zorder=5)
# plt.text(xticks[2], f1_1[2] + 0.2, f1_1[2], ha='center', fontproperties='Times New Roman', fontsize=9,  zorder=5)
# plt.text(xticks[3], f1_1[3] + 0.2, f1_1[3], ha='center', fontproperties='Times New Roman', fontsize=9,  zorder=5)

 
# plt.text(xticks[0] + width, f1_2[0] + 0.2, f1_2[0], ha='center',fontproperties='Times New Roman', fontsize=9,  zorder=5)
# plt.text(xticks[1] + width, f1_2[1] + 0.2, f1_2[1], ha='center',fontproperties='Times New Roman', fontsize=9,  zorder=5)
# plt.text(xticks[2] + width, f1_2[2] + 0.2, f1_2[2], ha='center', fontproperties='Times New Roman',fontsize=9,  zorder=5)
# plt.text(xticks[3] + width, f1_2[3] + 0.2, f1_2[3], ha='center', fontproperties='Times New Roman',fontsize=9,  zorder=5)

 
# plt.text(xticks[0] + width + width, f1_2[0] + 0.2, f1_2[0], ha='center',fontproperties='Times New Roman', fontsize=9,  zorder=5)
# plt.text(xticks[1] + width + width, f1_2[1] + 0.2, f1_2[1], ha='center',fontproperties='Times New Roman', fontsize=9,  zorder=5)
# plt.text(xticks[2] + width + width, f1_2[2] + 0.2, f1_2[2], ha='center', fontproperties='Times New Roman',fontsize=9,  zorder=5)
# plt.text(xticks[3] + width + width, f1_2[3] + 0.2, f1_2[3], ha='center', fontproperties='Times New Roman',fontsize=9,  zorder=5)

plt.legend(prop={'family' : 'Times New Roman', 'size': 10}, ncol = 2)
x_len = [-0.1,0.9,1.9,2.9,3.9]
x_len = np.array(x_len)
plt.xticks(x_len, x, fontproperties='Times New Roman',fontsize = 12)
plt.yticks(fontproperties='Times New Roman',fontsize = 12)
plt.ylim(ymin=0.5)
plt.xlabel("Methods", fontproperties='Times New Roman',fontsize=15)
plt.ylabel("Value",fontproperties='Times New Roman', fontsize=15)

# ax.spines['bottom'].set_linewidth('1.0')#设置边框线宽为2.01
# # ax.spines['bottom'].set_color('black')
# ax.spines['top'].set_linewidth('2.0')#设置边框线宽为2.0
# # ax.spines['top'].set_color('black')
# ax.spines['right'].set_linewidth('2.0')#设置边框线宽为2.0
# # ax.spines['right'].set_color('black')
# ax.spines['left'].set_linewidth('2.0')#设置边框线宽为2.0
# # ax.spines['left'].set_color('black')

plt.title('Comparison with classical methods', fontsize=15)
plt.savefig("3")

plt.show()
 