import matplotlib.pyplot as plt
import numpy as np


def draw_size(y1, y2, y, legend, path, size=25):
    plt.figure(figsize=(12, 8))
    n = 19
    x = np.arange(n)
    y = [y] * n

    plt.plot(x, y, c='r', label=legend)
    plt.plot(x, y1, marker='x', markersize=10, c='b', label='QueryTrans base')
    plt.plot(x, y2, marker='.', markersize=10, c='b', label='QueryTrans small')
    plt.grid(ls='--')
    if legend is not None:
        plt.legend(fontsize=size, ncol=3, bbox_to_anchor=(1.13, 1.157), edgecolor='none')
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], fontsize=size)
    plt.yticks(fontsize=size)
    plt.ylabel('Avg acc (%)', fontsize=size)
    plt.xlabel('Epoch', fontsize=size - 1)
    # plt.show()
    plt.savefig(path, format='pdf')

# legend = ['NGMv2', 'BBGM', 'NGMv2', 'BBGM']
# y=[80.1, 79, 80.58, 82.15]
# #ngmvoc bbgmvoc ngm71k bbgm71k
# y1 = [
#        [77.73, 79.03, 80.26, 81.59, 82.21, 82.53, 82.76, 82.98, 83.10, 83.23, 83.43, 83.47, 83.46, 83.50, 83.52, 83.49, 83.50, 83.45, 83.51],
#        [78.12, 79.96, 81.76, 82.43, 82.98, 83.37, 83.72, 84.08, 84.42, 84.47, 84.46, 84.50, 84.45, 84.39, 84.35, 84.41, 84.32, 84.40, 84.45],
#        [77.43, 78.38, 79.24, 79.91, 80.57, 81.06, 81.37, 81.68, 81.98, 82.16, 82.42, 82.44, 82.47, 82.41, 82.39, 82.40, 82.43, 82.38, 82.42],
#        [78.32, 79.78, 80.52, 81.38, 82.27, 82.87, 83.21, 83.56, 83.76, 83.91, 83.84, 83.87, 83.82, 83.90, 83.84, 83.85, 83.92, 83.91, 83.88]
#      ]
# y2 = [
#        [76.83, 78.03, 78.89, 79.52, 80.46, 81.09, 81.12, 81.17, 81.15, 81.13, 81.14, 81.16, 81.08, 81.13, 81.17, 81.01, 81.09, 81.14, 81.15],
#        [77.53, 79.17, 80.34, 80.89, 81.45, 81.99, 82.12, 82.15, 82.14, 82.13, 82.10, 82.08, 82.11, 82.07, 82.03, 82.10, 82.01, 82.12, 82.14],
#        [75.39, 76.52, 77.61, 78.09, 78.59, 79.00, 79.39, 79.41, 79.42, 79.38, 79.40, 79.35, 79.37, 79.34, 79.40, 79.31, 79.35, 79.38, 79.40],
#        [76.48, 77.52, 78.71, 79.08, 79.62, 80.00, 80.32, 80.39, 80.34, 80.35, 80.37, 80.31, 80.38, 80.39, 80.32, 80.28, 80.35, 80.34, 80.38]
#      ]
# path = [r'images/' + 'ngmvocsize.pdf', r'images/' + 'bbgmvocsize.pdf', r'images/' + 'ngm71ksize.pdf', r'images/' + 'bbgm71ksize.pdf']
#
# for i in range(4):
#     draw_size(y1[i], y2[i], y[i], legend[i], path[i])

def drawlayers(trans, gcn, path, size=25):
    plt.figure(figsize=(12, 8))
    n = 6
    x = [2, 3, 4, 5, 6, 7]

    plt.plot(x, gcn, ls='--', marker='*', markersize=16, linewidth=2.5,  c='r', label='GCN')
    plt.plot(x, trans, marker='s', markersize=16, linewidth=2.5, label='TransformerConv')
    # plt.plot(x, spair, marker='.', markersize=10, label='GMTr on Spair-71k')
    plt.grid(ls='--')
    plt.legend(fontsize=size)
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    plt.ylabel('Avg acc (%)', fontsize=size)
    plt.xlabel('Number of layers', fontsize=size)
    # plt.show()
    plt.savefig(path, format='pdf')

trans=[[82.8, 83.6, 83.4, 82.4, 82., 81.5], [82.6, 83.18, 83.0, 82.76, 82.24, 81.7]]
gcn=[[83.2] * 6, [82.46] * 6]
path= ['images/vocnumlayer.pdf', 'images/71knumlayer.pdf']

for i in range(2):
    drawlayers(trans[i], gcn[i], path[i])