import torch
import torch.nn as nn
import torch.nn.functional as func
from typing import *


class PrintShape(nn.Module):
    def __init__(self, name: str = None):
        super(PrintShape, self).__init__()
        self.name = name

    def forward(self, x):
        if self.name is not None:
            print(self.name, x.shape)
        else:
            print(x.shape)
        return x


class ConvNormAct(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int, s: int = 1, p: int = 0, d: int = 1, g: int = 1,
                 norm: nn.Module = None, act: nn.Module = None):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=d, groups=g, bias=False)
        self.norm = norm if norm is not None else nn.BatchNorm2d(out_c)
        self.act = act if act is not None else nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DWConvNormAct(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, k: int, s: int = 1, p: int = 0, d: int = 1, g: int = 1,
                 norm: nn.Module = None, act: nn.Module = None):
        super().__init__(
            nn.Conv2d(in_c, in_c, kernel_size=k, stride=s, padding=p, dilation=d, groups=in_c),
            ConvNormAct(in_c, out_c, k=1, s=1, p=0, g=g, norm=norm, act=act)
        )



class PartialConv(nn.Module):
    """
    This is Partial Convolution Layer introduced in
    "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks"
    By
    Jierun Chen, Shiu-hong Kao, Hao He, Weipeng Zhuo, Song Wen, Chul-Ho Lee, S.-H. Gary Chan
    """
    def __init__(self, in_c: int, out_c: int, focus_c: int, k: int, s: int=1, p: int=0, d: int=1, g: int = 1):
        super(PartialConv, self).__init__()
        self.in_c = in_c
        self.conv = ConvNormAct(focus_c, focus_c + out_c - in_c, k, s, p, d, g)
        self.focus_c = focus_c

    def forward(self, x):
        x1, x2 = torch.split(x, [self.focus_c, self.in_c - self.focus_c], dim=1)
        x1 = self.conv(x1)
        return torch.cat([x1, x2], dim=1)


class FasterNetBlock(nn.Module):
    """
        This is FasterNetBlock introduced in
        "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks"
        By
        Jierun Chen, Shiu-hong Kao, Hao He, Weipeng Zhuo, Song Wen, Chul-Ho Lee, S.-H. Gary Chan
        """
    def __init__(self, in_c: int, kernel_size: int = 3, expand: int = 2):
        super().__init__()
        self.expand = expand
        self.partial_conv = PartialConv(in_c, in_c, in_c//4, k=kernel_size, s=1, p=kernel_size//2)
        self.conv1 = ConvNormAct(in_c, in_c * expand, 1)
        self.conv2 = nn.Conv2d(in_c * expand, in_c, 1)

    def forward(self, x):
        identity = x
        x = self.partial_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return func.leaky_relu(x + identity, 0.1, inplace=True)



class InvertedResidual(nn.Sequential):
    def __init__(self, in_c: int, expand_ratio: int = 2):
        super().__init__(
            ConvNormAct(in_c, in_c * expand_ratio, 1),
            nn.Conv2d(in_c * expand_ratio, in_c, 3, 1, 1)
        )

    def forward(self, x):
        return func.leaky_relu(super().forward(x) + x, 0.01, inplace=True)



def replaceLayerByType(model: nn.Module, target_layer_type: type, new_layer_type: type, *args, **kwargs):
    for name, layer in model.named_children():
        if isinstance(layer, target_layer_type):
            setattr(model, name, new_layer_type(*args, **kwargs))
            return
        else:
            replaceLayerByType(layer, target_layer_type, new_layer_type, *args, **kwargs)


def replaceLayerMatchName(model: nn.Module, target_string: str, new_layer_type: type, *args, **kwargs):
    for name, layer in model.named_children():
        if target_string in name:
            setattr(model, name, new_layer_type(*args, **kwargs))
            return
        else:
            replaceLayerMatchName(layer, target_string, new_layer_type, *args, **kwargs)




class ModelWrapper(nn.Module):
    def __init__(self, model:nn.Module, input_shapes: List[List[int]]):
        super(ModelWrapper, self).__init__()
        self.model = model

        self.input_shapes = input_shapes

    def forward(self, *args):
        x = [torch.randn(1, *input_shape) for input_shape in self.input_shapes]
        return self.model(*x)


def inferSpeedTest(model, input_shapes: List[List[int]], device="cuda", batch_size: int=1):
    from time import time
    model.to(device)
    model.eval()
    print("Start inference speed test on model %s" % model.__class__.__name__)
    with torch.no_grad():
        x = [torch.randn(batch_size, *input_shape).to(device) for input_shape in input_shapes]
        start = time()
        for _ in range(1000):
            y = model(*x)
        end = time()
    print("Inference time: %f ms" % (end - start))
    model.train()


