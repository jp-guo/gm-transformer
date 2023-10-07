import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

def extract(img: Tensor, ps: Tensor, ns: Tensor, extract_size: int=16):
    """

    :param img: shape [b, c, h, w]
    :param ps: shape [b, n, 2]
    :param ns: shape [b]
    :param extract_size: size of the patch
    :return: patch set [b, n, c, size, size]
    """
    b, c = img.shape[0], img.shape[1]
    n = ps.shape[1]
    qs = []

    for i in range(n):
        patch = torch.zeros((b, c, extract_size, extract_size))
        for j in range(b):
            if ns[j] > i:
                patch[j] = imageExtract(img[j], ps[j, i, 0], ps[j, i, 1], extract_size)
        qs.append(patch)
    return qs


def imageExtract(img: Tensor, x: Tensor, y: Tensor, extract_size: int=16) -> Tensor:
    '''

    :param x: x coordinate, scaler
    :param y: y coordinate, scaler
    :param img: [c, h, w], tensor
    :param extract_size: size of the extracted image, scale
    :return: extract_image: [c, extract_size, extract_size]
    '''
    c, img_size = img.shape[0], img.shape[1]
    x = int(torch.clamp(x, 0, img_size - 1))
    y = int(torch.clamp(y, 0, img_size - 1))


    left = max(0, extract_size // 2 - 1 - y)
    ori_left = max(0, y - extract_size // 2 + 1)
    right = min(extract_size // 2 - 1 + img_size -y -1, extract_size - 1)
    ori_right = min(y + extract_size // 2, img_size - 1)
    buttom = max(0, extract_size // 2 - 1 - x)
    ori_buttom = max(0, x - extract_size // 2 + 1)
    top = min(extract_size // 2 - 1 + img_size - x - 1, extract_size - 1)
    ori_top = min(x + extract_size // 2, img_size - 1)

    padding = torch.zeros((c, extract_size, extract_size))

    padding[:, left: right + 1, buttom: top + 1] = img[:, ori_left: ori_right + 1, ori_buttom: ori_top + 1].detach()

    return padding


if __name__ == '__main__':
    img = torch.zeros((3, 3, 8, 8))
    cnt = 0
    for i in range(8):
        for j in range(8):
            cnt += 1
            img[:, 0, i, j] = cnt
            img[:, 1, i, j] = cnt * 2
            img[:, 2, i, j] = cnt * 3

    print(img)

    ps = torch.tensor([[[1, 2], [2, 1]], [[1, 1], [0, 0]], [[0, 0], [0, 0]]]).reshape(3, 2, 2)
    ns = torch.tensor([2, 1, 0]).reshape(-1)
    print(extract(img, ps, ns, 6))
    #
    # ps = torch.tensor([1, 1]).reshape(1, 1, 2)
    # ns = torch.tensor(1).reshape(-1)
    # print(extract(torch.unsqueeze(img [0], 0), ps, ns, 6))
    #
    # ps = torch.tensor([1, 2]).reshape(1, 1, 2)
    # ns = torch.tensor(1).reshape(-1)
    # print(extract(torch.unsqueeze(img[0], 0), ps, ns, 6))

    # ps = torch.tensor([7, 7]).reshape(1, 1, 2)
    # ns = torch.tensor(1).reshape(-1)
    # print(extract(torch.unsqueeze(img[0], 0), ps, ns, 6))



# a = torch.arange(3)
# b = torch.arange(3, 5)
#
# ein_out = torch.einsum('i,j->ij', a, b).numpy()  # ein_out = torch.einsum('i,j', [a, b]).numpy()
# org_out = torch.outer(a, b).numpy()
#
# print("input:\n", a, b,  sep='\n')
# print("ein_out: \n", ein_out.shape)
# print("org_out: \n", org_out.shape)
# print("is org_out == ein_out ?", np.allclose(ein_out, org_out))
