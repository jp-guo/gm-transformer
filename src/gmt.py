from functools import partial
from collections import OrderedDict
from src.feature_align import feature_align
import torch
import torch.nn as nn
import numpy as np
import math
from src.extract import imageExtract
from src.utils.config import cfg


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_one(patches, position):
    # patches: 3 * b * head * l2 * sqrt(l1) * sqrt(l1) * dim // head
    # position: b * l_2 * 2
    # l1 ** 2 is the number of patches, l2 is the number of keypoints
    _, batch, head, l2, length, __, dim = patches.size()
    l1 = int(length ** 2)
    pos_ = position[:, :, 0].unsqueeze(0).unsqueeze(2).unsqueeze(-1). \
        unsqueeze(-1).unsqueeze(-1).repeat(3, 1, head, 1, 1, int(l1 ** 0.5), dim) # 3 * b * head * l2 * 1 * sqrt(l1) * dim // head
    patches = torch.gather(patches, dim=4, index=pos_)

    pos_ = position[:, :, 1].unsqueeze(0).unsqueeze(2).unsqueeze(-1). \
        unsqueeze(-1).unsqueeze(-1).repeat(3, 1, head, 1, 1, 1, dim) # 3 * b * head * l2 * 1 * 1 * dim // head
    patches = torch.gather(patches, dim=5, index=pos_)
    return patches.squeeze(-2)

def get_single_filter_matrix(patches, keypoint, position):
    """
    :param patches: 3 * b * head * l_1 * dim//head  (l_1 is the number of patches)
    :param keypoint: 3 * b * head * l_2 * dim//head (l_2 is the number of keypoints)
    :param position: b * l_2 * 2 (height_position (value), weight_position (value))
    """
    _, batch, head, l1, dim = patches.size()
    l2 = keypoint.size(3)

    query = keypoint[0]     # b * head * l_2 * dim//head

    patches = patches.reshape(_, batch, head, int(l1 ** 0.5), int(l1 ** 0.5), dim) # 3 * b * head * sqrt(l1) * sqrt(l1) * dim // head
    patches = patches.unsqueeze(3).repeat(1, 1, 1, l2, 1, 1, 1) # 3 * b * head * l2 * sqrt(l1) * sqrt(l1) * dim // head

    bottom_x = torch.floor(position[:, :, 0]).long().unsqueeze(-1) # b * l_2 * 2
    top_x = torch.ceil(position[:, :, 0]).long().unsqueeze(-1)
    bottom_y = torch.floor(position[:, :, 1]).long().unsqueeze(-1)
    top_y = torch.ceil(position[:, :, 1]).long().unsqueeze(-1)

    # edge position
    top_x = torch.clamp(top_x, 0, int(l1 ** 0.5) - 1)
    top_y = torch.clamp(top_y, 0, int(l1 ** 0.5) - 1)
    bottom_x = torch.clamp(bottom_x, 0, int(l1 ** 0.5) - 1)
    bottom_y = torch.clamp(bottom_y, 0, int(l1 ** 0.5) - 1)

    top_left = torch.cat([bottom_y, bottom_x], dim=-1)
    top_right = torch.cat([top_y, bottom_x], dim=-1)
    bottom_left = torch.cat([bottom_y, top_x], dim=-1)
    bottom_right = torch.cat([top_y, top_x], dim=-1)

    patch1 = get_one(patches, top_left)
    patch2 = get_one(patches, top_right)
    patch3 = get_one(patches, bottom_left)
    patch4 = get_one(patches, bottom_right)
    patches = torch.cat([patch1, patch2, patch3, patch4], dim=-2)      # 3 * batch * head * l2 * 4 * dim
    attn = torch.einsum('bhld, bhlkd -> bhlk', query, patches[1])
    attn = torch.softmax(attn, dim=-1)
    keypoint = torch.einsum('bhlk, bhlkd -> bhld', attn, patches[2])
    keypoint = keypoint.transpose(1, 2).flatten(2) # b l d

    return keypoint


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

    def _forward(self, img, ps, ns):
        B, C, H, W = img.shape
        n_max = ps.shape[1]

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(img).flatten(2).transpose(1, 2)
        x = self.norm(x)

        patch_vec = torch.zeros((B, n_max, x.shape[-1])).to(x.device)
        for i in range(B):
            for j in range(ns[i]):
                vec = imageExtract(img[i], ps[i, j, 0], ps[i, j, 1]).unsqueeze(0).to(img.device)
                vec = self.proj(vec).flatten(2).transpose(1, 2)
                vec = self.norm(vec)
                patch_vec[i, j] = vec

        x = torch.cat((x, patch_vec), dim=1)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
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

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def global_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def local_global_forward(self, cls, x, query, ps, ns):
        B, N, C = x.shape
        M = query.shape[1]
        # [3, B, NUM_HEADS, 1 + N + M, C // NUM_HEADS]
        qkv = self.qkv(torch.cat([cls, x, query], dim=1)).reshape(B, 1 + N + M, 3, self.num_heads,
                                                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cls and x forward
        attn = (q[:, :, :1 + N] @ k[:, :, :1 + N].transpose(-2, -1)) * self.scale
        # with open('gmt.txt', 'a') as f:
        #     torch.set_printoptions(threshold=np.inf)
        #     print(x[0, :5, 16], file=f)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v[:, :, :1 + N]).transpose(1, 2).reshape(B, N + 1, C)

        # # query forward
        # for i in range(M):
        #     # idx = [j for j in range(1, 197)] + [197 + i]
        #     idx = [j for j in range(197)] + [197 + i]
        #     attn = (q[:, :, [197 + i]] @ k[:, :, idx].transpose(-2, -1)) * self.scale
        #     attn = attn.softmax(dim=-1)
        #     attn = self.attn_drop(attn)
        #     query = (attn @ v[:, :, idx]).transpose(1, 2).reshape(B, 1, C)
        #     x = torch.cat([x, query], dim=1)

        # qkv: [3, B, NUM_HEADS, 1 + N + M, C // NUM_HEADS]
        query = get_single_filter_matrix(qkv[:, :, :, 1: 197], qkv[:, :, :, 197:], ps / 16.)
        x = torch.cat([x, query.to(x.device)], dim=1)

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


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, input_dict):
        x, ps, ns = input_dict[0], input_dict[1], input_dict[2]

        if x.shape[1] > 197:
            y = self.attn.local_global_forward(self.norm1(x[:, :1]),
                                               self.norm1(x[:, 1: 197]),
                                               self.norm1(x[:, 197:]), ps, ns)
        elif x.shape[1] == 197:
            y = self.attn.global_forward(self.norm1(x))
        else:
            raise(f'Unsupported input shape!')

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return {0: x, 1: ps, 2: ns}


class Gmt(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, extract_layer=-1):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(Gmt, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.img_size = img_size
        self.depth = depth
        self.extract_layer = extract_layer
        self.patch_size = patch_size
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # self.l2norm = nn.LocalResponseNorm(embed_dim, alpha=embed_dim, beta=0.5, k=0)
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.fc_norm = norm_layer(embed_dim)

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def prepare_local_tokens(self, x, ps, ns):
        """
        extrcat features in blocks
        :param x:
        :param ps:
        :param ns:
        :return:
        """
        cls, patch = x[:, [0]], x[:, 1:]
        num_patches = 14

        patch = patch.transpose(1, 2)
        patch = patch.reshape((patch.shape[0], patch.shape[1], num_patches, num_patches))

        local_token = feature_align(patch, ps, ns, (self.img_size, self.img_size))
        local_token = local_token.transpose(1, 2)

        token = patch.flatten(2).transpose(1, 2)

        x = torch.cat((cls, token, local_token), dim = 1)

        return x

    def forward(self, x, ps, ns):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed._forward(x, ps, ns)  # [B, 197+n, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]


        x[:, : 197] += self.pos_embed
        x = self.pos_drop(x)

        for i in range(self.depth):
            if i == self.extract_layer:
                x = self.prepare_local_tokens(x, ps, ns)

            x = self.blocks[i]({0: x, 1: ps, 2: ns})[0]

            if i == cfg.GMT.FILTER_LOW - 1:
                filter_nodes = self.fc_norm(x)[:, 197:]
            if i == cfg.GMT.FILTER_HIGH - 1:
                filter_edges = self.fc_norm(x)[:, 197:]
            if i == cfg.GMT.BILINEAR_LOW - 1:
                bilinear_nodes = self.fc_norm(x)[:, 1: 197]
            if i == cfg.GMT.BILINEAR_HIGH - 1:
                bilinear_edges = self.fc_norm(x)[:, 1: 197]

            # if i == self.depth - 2:
            #     filter_nodes = self.fc_norm(x)[:, 197:]
            #     # bilinear_nodes = self.fc_norm(x)[:, 1: 197]
            #     pass
            # elif i == self.depth - 1:
            #     filter_edges = self.fc_norm(x)[:, 197:]
            #     bilinear_edges = self.fc_norm(x)[:, 1: 197]

        x = self.fc_norm(x)

        glb = x[:, 0]

        bilinear_nodes = bilinear_nodes.transpose(1, 2)
        filter_nodes = filter_nodes.transpose(1, 2)
        bilinear_edges = bilinear_edges.transpose(1, 2)
        filter_edges = filter_edges.transpose(1, 2)

        num_patches = self.img_size // self.patch_size

        bilinear_nodes = bilinear_nodes.reshape((x.shape[0], self.embed_dim, num_patches, num_patches))
        bilinear_edges = bilinear_edges.reshape((x.shape[0], self.embed_dim, num_patches, num_patches))
        bilinear_nodes = feature_align(bilinear_nodes, ps, ns, (self.img_size, self.img_size))
        bilinear_edges = feature_align(bilinear_edges, ps, ns, (self.img_size, self.img_size))

        if cfg.GMT.MODE == 'node filter + edge filter':
            nodes = filter_nodes
            edges = filter_edges
        elif cfg.GMT.MODE == 'node filter + edge bilinear':
            nodes = filter_edges
            edges = bilinear_edges
        elif cfg.GMT.MODE == 'node bilinear + edge filter':
            nodes = bilinear_edges
            edges = filter_edges
        elif cfg.GMT.MODE == 'node bilinear + edge bilinear':
            nodes = bilinear_nodes
            edges = bilinear_edges
        else:
            raise 'Unknown graph matching transformer mode.'

        return nodes, edges, glb

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)