import matplotlib.pyplot as plt
import numpy as np
# size = 5
# # 返回size个0-1的随机数
# a = np.random.random(size)
# b = np.random.random(size)
# c = np.random.random(size)
#
# # x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
# x = np.arange(size)
#
# # 有a/b/c三种类型的数据，n设置为3
# total_width, n = 0.8, 3
# # 每种类型的柱状图宽度
# width = total_width / n
#
# # 重新设置x轴的坐标
# x = x - (total_width - width) / 2
# print(x)
#
# # 画柱状图
#
# plt.bar(x, a, width=width, label="a")
# plt.bar(x + width, b, width=width, label="b")
# plt.bar(x + 2*width, c, width=width, label="c")
# # 显示图例
# plt.legend()
# # 功能1
# x_labels = ["第1组", "第2组", "第3组", "第4组", "第5组"]
# # 用第1组...替换横坐标x的值
# plt.xticks(x, x_labels)
# # 显示柱状图
# plt.show()


# def drawBar(gmt, vit, path, scale, size):
#     plt.figure(figsize=(8, 8))
#     total_width, n = 0.4, 2
#     x = np.arange(n)
#
#     width = total_width / n
#     x = x - (total_width - width) / 2
#     # plt.bar(x - width / 2 - 0.05, gmt, width=width, label="GMTr", color=['#63B2EE'])
#     # plt.bar(x + width / 2 + 0.05, vit, width=width, label='ViT', color=['#F88B88'])
#     plt.bar(x - width / 2 - 0.05, gmt, width=width, label="GMTr")
#     plt.bar(x + width / 2 + 0.05, vit, width=width, label='ViT')
#     #{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
#     plt.grid(linestyle='--')
#
#     plt.ylabel('Avg Acc (%)', fontsize=size)
#     plt.xlabel('Model Size', fontsize=size)
#     plt.yticks(size=size)
#     plt.ylim(scale[0], scale[1])
#     x_labels = ["Base", "Small"]
#     # 用第1组...替换横坐标x的值
#     plt.xticks(x, x_labels, size=size)
#     plt.legend(fontsize=size)
#     # plt.show()
#     plt.savefig(path,format='pdf')#输出
#
# gmts = [[83.52, 81.17], [84.5, 82.15], [82.46, 79.47], [83.91, 80.39]]
# vits = [[82.96, 80.70], [83.59, 81.37], [80.8, 78.95], [82.98, 80.18]]
# # scale = [[79, 84], [79, 85], [78, 83], [79, 84]]
# scale = [[75, 85], [75, 85], [75, 85], [75, 85]]
# path = ['ngmvoc.pdf', 'bbgmvoc.pdf', 'ngm71k.pdf', 'bbgm71k.pdf']
# size=24
# for i in range(len(gmts)):
#     drawBar(gmts[i], vits[i], 'images/' + path[i], scale[i], size)

# plt.figure(figsize=(8, 8))
# total_width, n = 0.4, 2
# x = np.arange(n)
#
# width = total_width / n
# x = x - (total_width - width) / 2
# # plt.bar(x - width / 2 - 0.05, gmt, width=width, label="GMTr", color=['#63B2EE'])
# # plt.bar(x + width / 2 + 0.05, vit, width=width, label='ViT', color=['#F88B88'])
# plt.bar(x - width / 2 - 0.05, gmt, width=width, label="GMTr")
# plt.bar(x + width / 2 + 0.05, vit, width=width, label='ViT')
# #{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
# plt.grid(linestyle='--')
#
# plt.ylabel('Avg Acc (%)', fontsize=size)
# plt.xlabel('Model Size', fontsize=size)
# plt.yticks(size=size)
# # plt.ylim(scale[0], scale[1])
# x_labels = ["Base", "Small"]
# # 用第1组...替换横坐标x的值
# plt.xticks(x, x_labels, size=size)
# plt.legend(fontsize=size)
# # plt.show()
# plt.savefig(path,format='pdf')#输出

import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码

labels = ['NGMv2', 'BBGM']
# [gmtb vitb gmts vits]
a = [[83.52, 84.5], [82.46, 83.91]]
b = [[82.96, 83.59], [80.8, 82.98]]
c = [[81.17, 82.15], [79.47, 80.39]]
d = [[80.7, 81.37], [78.95, 80.18]]
path = ['voc', '71k']
legend = [True, False]
size=25
# e = [80, 95]
# marks = ["o", "X", "+", "*", "O"]

def draw(a, b, c, d, size, path, legend):
    x = np.arange(len(labels))  # 标签位置
    width = 0.1  # 柱状图的宽度
    # plt.figure(figsize=(8, 8))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'


    fig, ax = plt.subplots(figsize=(16, 6))
    # axes = plt.gca()
    # axes.spines['top'].set_color('k')
    # rects1 = ax.bar(x - width * 2, a, width, label='GMT base', hatch="...", color='w', edgecolor="b")
    # rects2 = ax.bar(x - width + 0.01, b, width, label='ViT base', hatch="o", color='w', edgecolor="y")
    # rects3 = ax.bar(x + 0.02, c, width, label='GMT small', hatch="**", color='w', edgecolor="g")
    # rects4 = ax.bar(x + width + 0.03, d, width, label='ViT small', hatch="XX", color='w', edgecolor="r")
    rects1 = ax.bar(x - width * 2, a, width, label='QueryTrans base', color=['#3399ff'], edgecolor="k")
    rects2 = ax.bar(x - width + 0.01, b, width, label='ViT base', color=['#3366ff'], edgecolor="k")
    rects3 = ax.bar(x + 0.02, c, width, label='QueryTrans small', color=['#3333ff'], edgecolor="k")
    rects4 = ax.bar(x + width + 0.03, d, width, label='ViT small', color=['#0000cc'], edgecolor="k")
    # rects5 = ax.bar(x + width * 2 + 0.04, e, width, label='e', hatch="+", color='w', edgecolor="k")

    # 为y轴、标题和x轴等添加一些文本。
    ax.set_ylabel('Avg acc (%)', fontsize=size)
    # ax.set_xlabel('Framework', fontsize=20)
    # ax.set_title('标题')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=size)
    ax.tick_params(top=True, right=True)
    plt.ylim(77, 85)
    plt.yticks(fontsize=size)
    if legend:
        ax.legend(ncol=4, loc='center', bbox_to_anchor=(0.5, 1.097), edgecolor='none', fontsize=size)
    plt.grid(which='both', axis='y', ls='--')

    # plt.show()
    plt.savefig(r'images/' + path + '.pdf', format='pdf')

for i in range(2):
    draw(a[i], b[i], c[i], d[i], size, path[i], legend[i])