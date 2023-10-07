import torch
import torch.nn as nn
from torchvision import models
from src.ceit import *
from src.vit import *
from src.xcit import *
from src.gmt import *

class VGG16_base(nn.Module):
    r"""
    The base class of VGG16. It downloads the pretrained weight by torchvision API, and maintain the layers needed for
    deep graph matching models.
    """
    def __init__(self, batch_norm=True, final_layers=False):
        super(VGG16_base, self).__init__()
        self.node_layers, self.edge_layers, self.final_layers = self.get_backbone(batch_norm)
        if not final_layers: self.final_layers = None
        self.backbone_params = list(self.parameters())

    def forward(self, *input):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def get_backbone(batch_norm):
        """
        Get pretrained VGG16 models for feature extraction.

        :return: feature sequence
        """
        if batch_norm:
            model = models.vgg16_bn(pretrained=True)
        else:
            model = models.vgg16(pretrained=True)

        conv_layers = nn.Sequential(*list(model.features.children()))

        conv_list = node_list = edge_list = []

        # get the output of relu4_2(node features) and relu5_1(edge features)
        cnt_m, cnt_r = 1, 0
        for layer, module in enumerate(conv_layers):
            if isinstance(module, nn.Conv2d):
                cnt_r += 1
            if isinstance(module, nn.MaxPool2d):
                cnt_r = 0
                cnt_m += 1
            conv_list += [module]

            #if cnt_m == 4 and cnt_r == 2 and isinstance(module, nn.ReLU):
            if cnt_m == 4 and cnt_r == 3 and isinstance(module, nn.Conv2d):
                node_list = conv_list
                conv_list = []
            #elif cnt_m == 5 and cnt_r == 1 and isinstance(module, nn.ReLU):
            elif cnt_m == 5 and cnt_r == 2 and isinstance(module, nn.Conv2d):
                edge_list = conv_list
                conv_list = []

        assert len(node_list) > 0 and len(edge_list) > 0

        # Set the layers as a nn.Sequential module
        node_layers = nn.Sequential(*node_list)
        edge_layers = nn.Sequential(*edge_list)
        final_layers = nn.Sequential(*conv_list, nn.AdaptiveMaxPool2d((1, 1), return_indices=False)) # this final layer follows Rolink et al. ECCV20

        return node_layers, edge_layers, final_layers


class ResNet_base(nn.Module):
    r"""
    The base class of ResNet. It downloads the pretrained weight by torchvision API, and maintain the layers needed for
    deep graph matching models.
    """
    def __init__(self, resnet_name, final_layers=False):
        super(ResNet_base, self).__init__()
        self.node_layers, self.edge_layers, self.final_layers = self.get_backbone(resnet_name)
        if not final_layers: self.final_layers = None
        self.backbone_params = list(self.parameters())

    def forward(self, *input):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def get_backbone(name):
        """
        Get pretrained ResNet34 models for feature extraction.

        :return: feature sequence
        """
        module = getattr(models, name)
        model = module(pretrained=True)

        node_list = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4[0]]
        edge_list = [model.layer4[1:]]

        # Set the layers as a nn.Sequential module
        node_layers = nn.Sequential(*node_list)
        edge_layers = nn.Sequential(*edge_list)
        final_layers = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1), return_indices=False))

        return node_layers, edge_layers, final_layers


class VGG16_bn_final(VGG16_base):
    r"""
    VGG16 with batch normalization and final layers.
    """
    def __init__(self):
        super(VGG16_bn_final, self).__init__(True, True)


class VGG16_bn(VGG16_base):
    r"""
    VGG16 with batch normalization, without final layers.
    """
    def __init__(self):
        super(VGG16_bn, self).__init__(True, False)


class VGG16_final(VGG16_base):
    r"""
    VGG16 without batch normalization, with final layers.
    """
    def __init__(self):
        super(VGG16_final, self).__init__(False, True)


class VGG16(VGG16_base):
    r"""
    VGG16 without batch normalization or final layers.
    """
    def __init__(self):
        super(VGG16, self).__init__(False, False)


class ResNet34(ResNet_base):
    r"""
    ResNet34 without final layers.
    """
    def __init__(self):
        super(ResNet34, self).__init__('resnet34', False)


class ResNet34_final(ResNet_base):
    r"""
    ResNet34 with final layers.
    """
    def __init__(self):
        super(ResNet34_final, self).__init__('resnet34', True)


class ResNet50(ResNet_base):
    r"""
    ResNet50 without final layers.
    """
    def __init__(self):
        super(ResNet50, self).__init__('resnet50', False)


class ResNet50_final(ResNet_base):
    r"""
    ResNet50 with final layers.
    """
    def __init__(self):
        super(ResNet50_final, self).__init__('resnet50', True)


class ResNet101(ResNet_base):
    r"""
    ResNet101 without final layers.
    """
    def __init__(self):
        super(ResNet101, self).__init__('resnet101', False)


class ResNet101_final(ResNet_base):
    r"""
    ResNet101 with final layers.
    """
    def __init__(self):
        super(ResNet101_final, self).__init__('resnet101', True)


class Ceit_base(nn.Module):
    def __init__(self):
        super(Ceit_base, self).__init__()
        self.transformer = CeIT(
                hybrid_backbone=Image2Tokens(),
                patch_size=4, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))

        # self.default_cfg = _cfg()
        #--------------------------load parameters - -----------------------------------------------
        state_dict = torch.load('./experiments/ThinkMatchPretrained/ceit_base_patch16_224_150epochs/checkpoint.pth')
        print(self.transformer.load_state_dict(state_dict['model'], strict=False))
        # -----------------------------------------------------------------------------------------
        self.backbone_params = list(self.transformer.parameters())

    @property
    def device(self):
        return next(self.parameters()).device

class Xcit_medium(nn.Module):
    def __init__(self):
        super(Xcit_medium, self).__init__()
        self.transformer = XCiT(
        patch_size=16, embed_dim=512, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1e-5, tokens_norm=True)

        # --------------------------load parameters for base ViT version 2--------------------------
        weights_dict = torch.load('./experiments/ThinkMatchPretrained/xcit_medium_24_p16_224.pth')

        print(self.transformer.load_state_dict(weights_dict['model'], strict=False))
        # ------------------------------------------------------------------------------------------

        self.backbone_params = list(self.transformer.parameters())

    @property
    def device(self):
        return next(self.parameters()).device


class Vit_small(nn.Module):
    def __init__(self):
        super(Vit_small, self).__init__()
        self.transformer = ViT(img_size=224,
                             patch_size=16,
                             embed_dim=384,
                             depth=12,
                             num_heads=6
                              )
        self.backbone_params = list(self.transformer.parameters())

        # --------------------------load parameters for small ViT--------------------------
        weights_dict = torch.load('./experiments/ThinkMatchPretrained/deit_small_patch16_224.pth')
        weights_dict['model']['fc_norm.weight'] = weights_dict['model']['norm.weight']
        weights_dict['model']['fc_norm.bias'] = weights_dict['model']['norm.bias']

        items =['norm.weight', 'norm.bias', 'head.weight', 'head.bias']
        for item in items:
            del weights_dict['model'][item]

        print(self.transformer.load_state_dict(weights_dict['model'], strict=False))
        # ---------------------------------------------------------------------------------

    @property
    def device(self):
        return next(self.parameters()).device


class Vit_base(nn.Module):
    def __init__(self):
        super(Vit_base, self).__init__()
        self.transformer = ViT(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12)
        self.backbone_params = list(self.transformer.parameters())

        # --------------------------load parameters for base ViT-----------------------------------
        weights_dict = torch.load('./experiments/ThinkMatchPretrained/mae_finetuned_vit_base.pth')
        del weights_dict['model']['head.weight']
        del weights_dict['model']['head.bias']
        print(self.transformer.load_state_dict(weights_dict['model'], strict=False))
        # ------------------------------------------------------------------------------------------


    @property
    def device(self):
        return next(self.parameters()).device

class Gmt_small(nn.Module):
    def __init__(self):
        super(Gmt_small, self).__init__()
        self.gmt = Gmt(img_size=224,
                             patch_size=16,
                             embed_dim=384,
                             depth=12,
                             num_heads=6
                              )
        self.backbone_params = list(self.gmt.parameters())

        # --------------------------load parameters for small ViT--------------------------
        weights_dict = torch.load('./experiments/ThinkMatchPretrained/deit_small_patch16_224.pth')
        weights_dict['model']['fc_norm.weight'] = weights_dict['model']['norm.weight']
        weights_dict['model']['fc_norm.bias'] = weights_dict['model']['norm.bias']

        items = ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']
        for item in items:
            del weights_dict['model'][item]

        print(self.gmt.load_state_dict(weights_dict['model'], strict=False))
        # ---------------------------------------------------------------------------------
    @property
    def device(self):
        return next(self.parameters()).device


class Gmt_base(nn.Module):
    def __init__(self):
        super(Gmt_base, self).__init__()
        self.gmt= Gmt(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12)
        self.backbone_params = list(self.gmt.parameters())

        # --------------------------load parameters for base Gmt--------------------------
        weights_dict = torch.load('./experiments/ThinkMatchPretrained/mae_finetuned_vit_base.pth')
        del weights_dict['model']['head.weight']
        del weights_dict['model']['head.bias']
        print(self.gmt.load_state_dict(weights_dict['model'], strict=False))
        # ------------------------------------------------------------------------------------------

    @property
    def device(self):
        return next(self.parameters()).device


class NoBackbone(nn.Module):
    r"""
    A model with no CNN backbone for non-image data.
    """
    def __init__(self, *args, **kwargs):
        super(NoBackbone, self).__init__()
        self.node_layers, self.edge_layers = None, None

    def forward(self, *input):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

