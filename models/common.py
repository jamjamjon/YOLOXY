"""
Common modules
"""

import math
import platform
import warnings
from copy import copy
from pathlib import Path
import rich

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.cuda import amp

from models.nms import non_max_suppression
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, CONSOLE, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())  # nn.LeakyReLU()
        # self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())  # nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class CB(nn.Module):
    # convolution + bn
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.bn(self.conv(x))


class RepConv(nn.Module):
    # rep convolution block
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()

        assert k == 3, "RepConv Block always with 3x3 Conv"

        self.conv3x3 = CB(c1, c2, k=k, s=s, p=p, g=g)
        self.conv1x1 = CB(c1, c2, k=1, s=s, p=0, g=g)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        if hasattr(self, 'fusedconv'):
            y = self.fusedconv(x)
        else:
            y = self.conv1x1(x) + self.conv3x3(x)
        return self.act(y)

    def fuse_repconv(self):
        if not hasattr(self, 'fusedconv'):   
            self.fusedconv = nn.Conv2d(self.conv3x3.conv.in_channels, 
                                        self.conv3x3.conv.out_channels, 
                                        self.conv3x3.conv.kernel_size, 
                                        self.conv3x3.conv.stride, 
                                        self.conv3x3.conv.padding, 
                                        self.conv3x3.conv.groups, 
                                        bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.fusedconv.weight.data = kernel
        self.fusedconv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'conv3x3'):
            self.__delattr__('conv3x3')
        if hasattr(self, 'conv1x1'):
            self.__delattr__('conv1x1')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv3x3)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv1x1)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std 


class AsymConv(nn.Module):
    # Asymmetric Conv
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()

        self.convkxk = CB(c1, c2, k, s)            # k x k Conv 
        self.conv1xk = CB(c1, c2, (1, k), s)       # 1 x k Conv
        self.convkx1 = CB(c1, c2, (k, 1), s)       # k x 1 Conv
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())


    def forward(self, x):
        if hasattr(self, 'fusedconv'):
            y = self.fusedconv(x)
        else:
            
            y_kxk = self.convkxk(x)
            y_kx1 = self.convkx1(x)
            y_1xk = self.conv1xk(x)
            y = y_kxk + y_kx1 + y_1xk

        return self.act(y)


    def fuse_asymconv(self):
        if not hasattr(self, 'fusedconv'):   
            self.fusedconv = nn.Conv2d(self.convkxk.conv.in_channels, 
                                        self.convkxk.conv.out_channels, 
                                        self.convkxk.conv.kernel_size, 
                                        self.convkxk.conv.stride, 
                                        self.convkxk.conv.padding, 
                                        self.convkxk.conv.groups, 
                                        bias=True
                                        )

        kernel, bias = self.get_equivalent_kernel_bias()
        self.fusedconv.weight.data = kernel
        self.fusedconv.bias.data = bias     
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'convkxk'):
            self.__delattr__('convkxk')
        if hasattr(self, 'convkx1'):
            self.__delattr__('convkx1')
        if hasattr(self, 'conv1xk'):
            self.__delattr__('conv1xk')


    def get_equivalent_kernel_bias(self):
        # fuse conv * bn
        kernelkxk, biaskxk = self._fuse_bn_tensor(self.convkxk)
        kernel1xk, bias1xk = self._fuse_bn_tensor(self.conv1xk)
        kernelkx1, biaskx1 = self._fuse_bn_tensor(self.convkx1)
        
        # fuse branch
        self._add_to_square_kernel(kernelkxk, kernel1xk)
        self._add_to_square_kernel(kernelkxk, kernelkx1)

        return kernelkxk, biaskxk + bias1xk + biaskx1
    
    # fuse conv & bn
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std 
    
    # fuse sym-kernel & asym-kernel
    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, 
                      :, 
                      square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                      square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w
                     ] += asym_kernel


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution class
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class ESE(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    Forward(640x640): 0.1591 ms
    """
    def __init__(self, c1, act=True):
        super().__init__()
        self.fc = nn.Conv2d(c1, c1, kernel_size=1, padding=0)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())  # nn.LeakyReLU()

    def forward(self, x):
        return x * self.act(self.fc(x.mean((2, 3), keepdim=True)))


class C3xESE(nn.Module):
    # CSP Bottleneck with 3 convolutions + ESE
    def __init__(self, c1, c2, n=1, ese=True, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.ese = ESE(c2) if ese is True else nn.Identity()

    def forward(self, x):
        return self.cv3(self.ese(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)))



class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)



# TODO: deprecate, to remove
class Decouple(nn.Module):
    # Decoupled head
    def __init__(self, c1, nc=80, na=1):  # ch_in, num_classes, num_anchors
        super().__init__()
        self.na = na  # number of anchors
        self.nc = nc  # number of classes

        c_ = min(c1, 256)  # min(c1, nc * na)
        # c_ = min(c1 // 2, 256)  # min(c1, nc * na)   

        self.a = Conv(c1, c_, 1)        # stem
         
        self.bc = Conv(c_, c_, 3)     # fused b,c brach 
        # self.bc = CrossConv(c_, c_, 3, 1)

        self.b1 = nn.Conv2d(c_, na * 4, 1)      # box
        self.b2 = nn.Conv2d(c_, na * 1, 1)      # obj  
        self.c = nn.Conv2d(c_, na * nc, 1)      # cls


    def forward(self, x):
        bs, nc, ny, nx = x.shape  # BCHW
        x = self.bc(self.a(x))
        b_box = self.b1(x)     # box
        b_obj = self.b2(x)     # obj
        c = self.c(x)         # cls
        
        return torch.cat((b_obj.view(bs, self.na, 1, ny, nx), 
                          b_box.view(bs, self.na, 4, ny, nx), 
                          c.view(bs, self.na, self.nc, ny, nx)), 2).view(bs, -1, ny, nx)


# TODO: different branch has different branch head
class HeadBranch(nn.Module):
    # Head Branch Conv Block Before Output
    def __init__(self, c1, c2):
        super().__init__()
        self.dwconv = DWConv(c1, c1, 3)
        self.conv = Conv(c1, c1, 1)
        # self.conv = Conv(c1, c1, 3)
        # self.conv = AsymConv(c1, c1, 3)
        self.conv2d = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        # return self.conv2d(self.conv(self.dwconv(x)))
        return self.conv2d(self.conv(x))


class HydraHead(nn.Module):
    # Decoupled Hydra Head
    def __init__(self, c1, nc=80, na=1, nk=0):  # ch_in, num_classes, num_anchors, num_keypoints
        super().__init__()
        self.na = na    # number of anchors
        self.nc = nc    # number of classes
        self.nk = nk    # number of keypoints

        c_ = min(c1, 256)  # min(c1, nc * na)
        self.cv1 = Conv(c1, c_, 1)      # stem
        # self.cv2 = Conv(c_, c_, 3)    # TODO   
        self.cv2 = AsymConv(c_, c_, 3)  #   

        # Head Branch Conv Block To replace single 1x1 conv2d
        self.conv_box = HeadBranch(c_, na * 4)      # box => x,y,w,h
        self.conv_obj = HeadBranch(c_, na * 1)      # obj  
        self.conv_cls = HeadBranch(c_, na * nc)      # cls
        if self.nk > 0:
            self.conv_kpt = HeadBranch(c_, na * nk * 3)      # kpt => x,y,conf
        
        
    def forward(self, x):
        bs, nc, ny, nx = x.shape  # BCHW

        # x = self.cv1(x)
        x = self.cv2(self.cv1(x))
        x_box = self.conv_box(x)     # box
        x_obj = self.conv_obj(x)     # obj
        x_cls = self.conv_cls(x)     # cls
        if self.nk > 0:
            x_kpt = self.conv_kpt(x)     # cls
        
        # outputs list
        xs = [x_obj.view(bs, self.na, 1, ny, nx), 
              x_box.view(bs, self.na, 4, ny, nx), 
              x_cls.view(bs, self.na, self.nc, ny, nx)]
        if self.nk > 0:
            xs.append(x_kpt.view(bs, self.na, self.nk * 3, ny, nx))

        return torch.cat(xs, 2).view(bs, -1, ny, nx)


# ------------- New Head ---------------------------------------
class BranchAttn(nn.Module):
    # head(layer) attention block
    def __init__(self, c1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))     # GAP
        self.fc = nn.Conv2d(c1, c1, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv(c1, c1, 1)
    
    def forward(self, x):
        return self.conv(x * self.sigmoid(self.fc(self.gap(x))))     # weighted x


class Branch(nn.Module):
    # xxx Branch In Head 
    def __init__(self, c1, c2, add=False):
        super().__init__()
        self.attn = BranchAttn(c1)
        self.conv2d = nn.Conv2d(c1, c2, 1)
        self.add = add

    def forward(self, x):
        return self.conv2d(x + self.attn(x)) if self.add else self.conv2d(self.attn(x))


class HydraXHead(nn.Module):
    # Hydra X Head
    def __init__(self, c1, nc=80, na=1, nk=0):  # ch_in, num_classes, num_anchors, num_keypoints
        super().__init__()
        self.na = na    # number of anchors
        self.nc = nc    # number of classes
        self.nk = nk    # number of keypoints
        c_ = min(c1, 256) 
        
        self.stem = Conv(c1, c_, 1)     # stem
        self.cv2 = AsymConv(c_, c_, 3)  # TODO: keep ?     Conv(c_, c_, 3)

        self.conv_cls = Branch(c_, nc * na, add=True)     # cls branch
        self.conv_box = Branch(c_, 4 * na, add=False)      # box branch => x,y,w,h
        self.conv_obj = Branch(c_, 1 * na, add=False)      # obj branch
        if self.nk > 0:
            self.conv_kpt = Branch(c_, 3 * nk * na, add=False)      # kpt branch => x,y,conf
        
    def forward(self, x):
        bs, nc, ny, nx = x.shape  # BCHW

        x = self.cv2(self.stem(x))
        x_box, x_obj, x_cls = self.conv_box(x), self.conv_obj(x), self.conv_cls(x)      # output => box, obj, cls
        if self.nk > 0:
            x_kpt = self.conv_kpt(x)     # output => kpt
        
        # outputs list
        xs = [x_obj.view(bs, self.na, 1, ny, nx), 
              x_box.view(bs, self.na, 4, ny, nx), 
              x_cls.view(bs, self.na, self.nc, ny, nx)]
        if self.nk > 0:
            xs.append(x_kpt.view(bs, self.na, self.nk * 3, ny, nx))

        return torch.cat(xs, 2).view(bs, -1, ny, nx)
# ------------- New Head ---------------------------------------


class DetectX(nn.Module):
    # Anchor free Detect Layer
    stride = None  # strides computed during build
    export = False  # export mode
    export_raw = False  # export raw mode, for those not support complex operators like ScatterND, GatherND, ... 

    def __init__(self, nc=80, nk=0, ch=(), inplace=True):  # detection layer
        super().__init__()
        # CONSOLE.log(log_locals=True)      # local variables
        self.nc = nc        # number of classes
        self.nk = nk        # number of keypoints
        self.no_det = self.nc + 5   # num_outputs of detection box 
        self.no_kpt = 3 * self.nk   # num_outputs of keypoints 
        self.no = self.no_det + self.no_kpt    # number of outputs per anchor,  keypoint: (xi, yi, i_conf)
        self.nl = len(ch)  # number of detection layers, => 3
        self.na = self.anchors = 1    # number of anchors 
        self.grid = [torch.zeros(1)] * self.nl    # girds for every scales
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        # self.m = nn.ModuleList(HydraHead(x, self.nc, self.na, self.nk) for x in ch)  # hydra head
        self.m = nn.ModuleList(HydraXHead(x, self.nc, self.na, self.nk) for x in ch)  # new hydra X head



    def forward(self, x):
        z = []  # inference output
        if self.export_raw:  # export raw outputs vector
            x_raw = []    

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # outputs after hydra head

            # export raw mode 
            if self.export_raw:
                x_raw.append(x[i])

            # reshape tensor
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()   # torch.Size([1, 1, 80, 80, 85])

            # inference
            if not self.training:
                # ----------------------------------------------------
                # make grid
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    d = self.stride.device
                    t = self.stride.dtype
                    if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
                        yv, xv = torch.meshgrid(torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t), indexing='ij')
                    else:
                        yv, xv = torch.meshgrid(torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t))
                    self.grid[i] = torch.stack((xv, yv), 2).view(1, self.na, ny, nx, 2).float()
                grid_x = self.grid[i][..., 0:1]   # grid x
                grid_y = self.grid[i][..., 1:2]   # grid y
                # ----------------------------------------------------
                y = x[i]    # make a copy

                # do sigmoid to box (cls, conf)
                y[..., 4: self.nc + 5] = y[..., 4: self.nc + 5].sigmoid()  # det bbox {xywh, cls, conf, kpts(optional)}
                
                # do sigmoid to kpt (conf)
                if self.nk > 0:
                    y[..., self.no_det + 2::3] = y[..., self.no_det + 2::3].sigmoid()  # kpt {x,y,conf} 

                # decode xywh, kpt(optional)
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] + self.grid[i].to(y.device)) * self.stride[i].to(y.device)  # xy
                    y[..., 2:4] = torch.exp(y[..., 2:4]) * self.stride[i].to(y.device) # wh

                    if self.nk > 0:     # has kpt
                        y[..., self.no_det::3] = (y[..., self.no_det::3] + grid_x.repeat((1,1,1,1, self.nk)).to(y.device)) * self.stride[i].to(y.device)  # x of kpt
                        y[..., self.no_det + 1::3] = (y[..., self.no_det + 1::3] + grid_y.repeat((1,1,1,1, self.nk)).to(y.device)) * self.stride[i].to(y.device)  # y of kpt

                else:
                    xy = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
                    wh = torch.exp(y[..., 2:4]) * self.stride[i]  # wh
                    if self.nk > 0:     # has kpt
                        y[..., self.no_det::3] = (y[..., self.no_det::3] + self.grid[i].repeat((1,1,1,1, self.nk)).to(y.device)) * self.stride[i].to(y.device)  # xy of kpt
                        y[..., self.no_det + 1::3] = (y[..., self.no_det + 1::3] + self.grid[i].repeat((1,1,1,1, self.nk)).to(y.device)) * self.stride[i].to(y.device)  # xy of kpt
                    y = torch.cat((xy, wh, y[..., 4:]), -1)

                z.append(y.view(bs, -1, self.no))

        return x_raw if self.export_raw else x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)  # x not do sigmoid(), while z did



class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='', device=torch.device('cpu'), dnn=False, data=None, fp16=False):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx with --dnn


        super().__init__()
        from models.experimental import attempt_load, attempt_download  # scoped to avoid circular import


        w = str(weights[0] if isinstance(weights, list) else weights)
        # pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = self.model_type(w)  # get backend
        pt, onnx, rknn = self.model_type(w)  # get backend

        w = attempt_download(w)  # download if not local
        # fp16 &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16
        fp16 &= (pt or onnx) and device.type != 'cpu'  # FP16
        stride, names = 32, [f'class{i}' for i in range(1000)]  # assign defaults
        if data:  # assign class names (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device)
            stride = max(int(model.stride.max()), 32)  # model stride
            # tag = model.tag if hasattr(model, 'tag') else 'YOLOV5' # model tag: yolov5/x
            nk = model.nk if hasattr(model, 'nk') else 0 
            kpt_kit = model.kpt_kit if hasattr(model, 'kpt_kit') else None 
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
       
        # assign all variables to self
        self.__dict__.update(locals())  


    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize)[0]
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.onnx
        if any(warmup_types) and self.device.type != 'cpu':
            im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            # for _ in range(2 if self.jit else 1):  #
            for _ in range(1):  #
                self.forward(im)  # warmup


    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        
        from export import export_formats
        suffixes = list(export_formats().Suffix)  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        # pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
        pt, onnx, rknn = (s in p for s in suffixes)
        return pt, onnx, rknn

    @staticmethod
    def _load_metadata(f='path/to/meta.yaml'):
        # Load metadata from meta.yaml if it exists
        with open(f, errors='ignore') as f:
            d = yaml.safe_load(f)
        return d['stride'], d['names']  # assign stride, names
