# WF-Res

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class SignedLog(nn.Module):
    def forward(self, x):
        return torch.sign(x) * torch.log1p(torch.abs(x))


class SignedExp(nn.Module):
    def forward(self, x):
        return torch.sign(x) * torch.expm1(torch.abs(x))


class Conv2d_BN(nn.Module):
    """Convolution with BN module."""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, pad=0, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, act_layer=None):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, dilation, groups, bias=False)
        self.bn = norm_layer(out_ch)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        """foward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)
        return x


class WeberFechner(nn.Module):
    def __init__(self, dim, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)
        self.norm = nn.LayerNorm(dim)

    def forward(self, shortcut, x):
        logx = shortcut + self.drop_path(self.norm(x * self.gamma))
        return logx


class Block(nn.Module):
    r""" WFRes Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, mlp_ratio=4, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = Conv2d_BN(dim, dim, kernel_size=7, pad=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim)) # pointwise/1x1 convs, implemented with linear layers
        self.gelu = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)

        self.WF = WeberFechner(dim, drop_path)

    def forward(self, x):
        shortcut = x.permute(0, 2, 3, 1).contiguous()
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous() # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.gelu(x)
        x = self.pwconv2(x)

        x = self.WF(shortcut, x)
        x = x.permute(0, 3, 1, 2).contiguous() # (N, H, W, C) -> (N, C, H, W)
        return x

class WFRes(nn.Module):
    r""" WFRes
        A PyTorch impl of our WFRes backbone -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], mlp_ratio=4, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            Conv2d_BN(in_chans, dims[0] // 2, 3, 2, 1, act_layer=nn.Hardswish),
            Conv2d_BN(dims[0] // 2,dims[0] , 3, 2, 1,),
            SignedLog()
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                Conv2d_BN(dims[i], dims[i+1], 3, 2, 1),
                LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], mlp_ratio=mlp_ratio, drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.signed_exp = SignedExp()
        self.norm1 = nn.LayerNorm(dims[-1], eps=1e-6)
        self.norm2 = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm1(x.permute(0, 2, 3, 1).contiguous())
        x = self.signed_exp(x)
        return x.mean([1, 2]) # global average pooling, (N, H, W, C) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm2(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


@register_model
def wfres(pretrained=False, **kwargs):
    model = WFRes(depths=[3, 3, 12, 3], dims=[96, 192, 384, 768], mlp_ratio=3,  **kwargs)
    return model

