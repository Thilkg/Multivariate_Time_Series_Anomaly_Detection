#画图2
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))   # 设置图像大小
names = ['CBLOF', 'FBagging', 'KNN', 'HBOS', 'IForest', 'OCSVM','LOF','MCD','Ours']
x = range(len(names))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字

plt.rcParams['savefig.dpi'] = 800 #图片像素
# #
plt.rcParams['figure.dpi'] = 800 #分辨率

plt.ylim(0.0, 1.0)  # 设置x轴的范围
y_1 = [0.425, 0.424, 0.431, 0.397, 0.413, 0.421,0.426,0.422,0.877]
y_2 = [0.900, 0.899, 0.902, 0.896, 0.895, 0.899,0.900,0.901,0.822]   
y_3 = [0.277, 0.279, 0.284, 0.258, 0.270, 0.276,0.281,0.276,0.940]

plt.plot(x, y_1, color='blue', marker='o', linestyle='-', label='F1-scores')
plt.plot(x, y_2, color='red', marker='D', linestyle='-', label='Precision')
plt.plot(x, y_3, color='green', marker='^', linestyle='-', label='Recall')
plt.legend()  # 显示图例
plt.legend(fontsize=13)
plt.xticks(x, names,fontproperties='Times New Roman', rotation=45,fontsize=12)
# plt.xlabel("window sizes",fontsize=15)  # X轴标签
plt.xlabel("Methods", fontproperties='Times New Roman',fontsize=15)
plt.ylabel("F1 scores",fontproperties='Times New Roman',fontsize=15)  # Y轴标签
plt.title('Comparison with traditional methods',fontproperties='Times New Roman', fontsize=15)
plt.savefig("2")
plt.show()