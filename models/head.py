"""
Head parts
"""

import math
import platform
import warnings
from copy import copy
from pathlib import Path
import rich
import yaml
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.cuda import amp

from models.nms import non_max_suppression
from models.common import *
from models.experimental import *
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, CONSOLE, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync



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
        return self.conv2d(self.conv(self.dwconv(x)))
        # return self.conv2d(self.conv(x))


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

        x = self.cv2(self.cv1(x))
        x_box, x_obj, x_cls = self.conv_box(x), self.conv_obj(x), self.conv_cls(x)     # box, obj, cls
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



class Detect(nn.Module):
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
        self.m = nn.ModuleList(HydraHead(x, self.nc, self.na, self.nk) for x in ch)  # hydra head
        # self.m = nn.ModuleList(HydraXHead(x, self.nc, self.na, self.nk) for x in ch)  # new hydra X head


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



