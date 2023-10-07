from torch import nn
from src.lap_solvers.sinkhorn import Sinkhorn
import torch
import math
import numpy as np
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=4,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.apply(_init_vit_weights)


    def forward(self, x, K=None, n1=None, n2=None):
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches]
        # @: multiply -> [batch_size, num_heads, num_patches, num_patches]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # with open('123.txt', 'a') as f:
        #     torch.set_printoptions(threshold=np.inf)
        #     print(x[0, :, :5], file=f)
        #     print('before:', attn[0][0][0], file=f)
        # attn = (F.normalize(q) @ F.normalize(k).transpose(-2, -1)) * self.scale
        for i in range(B):
            n_row, n_col = n1[i], n2[i]
            attn[i, :, n_row * n_col:] = -1e10
            attn[i, :, : n_row * n_col, n_row * n_col:] = -1e10

        if K is not None:
            K = K.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn = attn + K

        attn = attn.softmax(dim=-1)
        # with open('123.txt', 'a') as f:
        #     torch.set_printoptions(threshold=np.inf)
        #     # print('x:', x[0][0], file=f)
        #     print('K:', K[0][0][0], file=f)
        #     print('after:', attn[0][0][0], file=f)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


    def filter_forward(self, x, A, K=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        A = A.unsqueeze(1).repeat(1, self.num_heads, 1, 1)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.where(A > 0, attn, torch.tensor(-1e10, device=A.device))

        attn = attn.softmax(dim=-1)
        if K is not None:
            K = K.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn = attn + K
        attn = F.normalize(attn, p=1, dim=2)
        # with open('123.txt', 'a') as f:
        #     torch.set_printoptions(threshold=np.inf)
            # print('x:', x[0][0], file=f)
            # print('K:', K[0][0][0], file=f)
            # print('after:', attn[0][0][0], file=f)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LocallyEnhancedFeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 kernel_size=3, with_bn=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # pointwise
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        # depthwise
        self.conv2 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=hidden_features
        )
        # pointwise
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        # self.drop = nn.Dropout(drop)

        self.with_bn = with_bn
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.bn3 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        b, n, k = x.size()
        x = x.reshape(b, int(math.sqrt(n)), int(math.sqrt(n)), k).permute(0, 3, 1, 2)
        if self.with_bn:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act(x)
            x = self.conv3(x)
            x = self.bn3(x)
        else:
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.conv3(x)

        x = x.flatten(2).permute(0, 2, 1)
        return x


class AssociationLayer(nn.Module):
    def __init__(self,
                 in_node_features,
                 out_node_features,
                 forard_type='mlp',
                 sk_channel=1,
                 sk_iter=20,
                 sk_tau=0.05,
                 mlp_ratio=4,
                 head=4,
                 norm=True,
                 proj=False):
        super(AssociationLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.out_nfeat = out_node_features

        self.proj = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            nn.LeakyReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            # nn.LeakyReLU(),
            # nn.LayerNorm(self.out_nfeat)
        )if proj else None

        self.attn = Attention(self.out_nfeat, num_heads=head)
        self.norm1 = nn.LayerNorm(self.out_nfeat) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(self.out_nfeat) if norm else nn.Identity()
        if forard_type == 'leff':
            self.mlp = LocallyEnhancedFeedForward(self.out_nfeat, hidden_features = self.out_nfeat * mlp_ratio, out_features=self.out_nfeat)
        elif forard_type == 'mlp':
            self.mlp = Mlp(in_features=self.out_nfeat, hidden_features=self.out_nfeat * mlp_ratio)
        else:
            raise 'Unsupported forward type'
        self.apply(_init_vit_weights)


    def filter_forward(self, x, A, K):
        if self.proj is not None:
            x = self.proj(x)

        x = x + self.attn.filter_forward(self.norm1(x), A, K)
        x = x + self.mlp(self.norm2(x))

        return x


    def forward(self, x, K=None, n1=None, n2=None):
        # x1 = self.proj(x)
        # x1 = x1 + self.attn(self.norm1(x1))
        # x2 = x1 + self.mlp(self.norm2(x1))

        # if self.classifier is not None:
        #     assert n1.max() * n2.max() == x.shape[1]
        #     x3 = self.classifier(x2)
        #     n1_rep = torch.repeat_interleave(n1, self.sk_channel, dim=0)
        #     n2_rep = torch.repeat_interleave(n2, self.sk_channel, dim=0)
        #     x4 = x3.permute(0, 2, 1).reshape(x.shape[0] * self.sk_channel, n2.max(), n1.max()).transpose(1, 2)
        #     x5 = self.sk(x4, n1_rep, n2_rep, dummy_row=True).transpose(2, 1).contiguous()
        #     x6 = x5.reshape(x.shape[0], self.sk_channel, n1.max() * n2.max()).permute(0, 2, 1)
        #     x_new = torch.cat((x2, x6), dim=-1)
        # else:
        #     x_new = x2
        #
        # return x_new
            # vec = torch.zeros(x.shape, device=x.device)
            # n_total = n1 * n2
            # for i in range(x.shape[0]):
            #     vec[i, : n_total[i]] = find_nobs(x[i, : n_total[i]].transpose(-1, -2)).transpose(-1, -2)
            # # for i in range(x.shape[0]):
            # #     vec[i] = find_nobs(x[i].transpose(-1, -2)).transpose(-1, -2)
            # x = vec
        # vec = torch.rand(x.shape, device=x.device)
        if self.proj is not None:
            # with open('proj.txt', 'a') as f:
            #     torch.set_printoptions(threshold=np.inf)
            #     print(x[0, :], file=f)
                x = self.proj(x)
                # print(x[0, :, :5], file=f)
        x = x + self.attn(self.norm1(x), K, n1, n2)
        x = x + self.mlp(self.norm2(x))
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


