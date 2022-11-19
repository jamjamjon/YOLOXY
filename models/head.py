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
from collections import OrderedDict

from models.nms import non_max_suppression
from models.common import *
from models.experimental import *
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, CONSOLE, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync



class Detect(nn.Module):
    # Detect with Hydra Head
    stride = None  # strides computed during build
    export = False  # export mode

    def __init__(self, ch=(), nc=0, nk=0, inplace=False):  
        super().__init__()
        self.na = self.anchors = 1    # number of anchors 
        self.nc, self.nk, self.nl = nc, nk, len(ch)     # number of classes, keypoints, detection layers(scales)
        self.no_det, self.no_kpt = self.nc + 5, 3 * self.nk   # num_outputs of detection_box , keypoints
        self.no = self.no_det + self.no_kpt   # number of outputs per anchor,  kpts: (xi, yi, i_conf)
        self.grid = [torch.zeros(1)] * self.nl    # girds for every scales
        # self.grid_x, self.grid_y = torch.empty(0), torch.empty(0)  # grid_x, grid_y
        self.inplace = inplace  # use in-place ops (e.g. slice assignment), TODO: not used any more.
        # self.m = nn.ModuleList(Hydra(x, nc=self.nc, na=self.na, nk=self.nk) for x in ch)  # hydra head
        self.m = nn.ModuleList(HydraX(x, nc=self.nc, na=self.na, nk=self.nk) for x in ch)  # hydra head

    def forward(self, x):
        z = []  # inference output

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # outputs after hydra head
            bs, _, ny, nx = map(int, x[i].shape)
            x[i] = x[i].view(-1, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()   # reshape tensor torch.Size([1, 1, 80, 80, 85])

            # inference
            if not self.training:
                # make grid
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)
                grid_x = self.grid[i][..., 0:1]   # grid x
                grid_y = self.grid[i][..., 1:2]   # grid y

                # decode
                if self.nk <= 0:    # det => det(x, y, w, h, conf, cls)
                    xy, wh, conf = x[i].split((2, 2, self.nc + 1), 4)
                    xy = (xy + self.grid[i]) * self.stride[i]
                    wh = torch.exp(wh) * self.stride[i]
                    y = torch.cat((xy, wh, conf.sigmoid()), 4) 
                else:  # kpts, for better ONNX graph
                    xy, wh, conf, kpts_xyconf = x[i].split((2, 2, self.nc + 1, self.nk * 3), 4) 
                    xy = (xy + self.grid[i]) * self.stride[i]   # xy
                    wh = torch.exp(wh) * self.stride[i]     # wh
                    kpts_x = (kpts_xyconf[..., 0::3] + grid_x.repeat((1,1,1,1, self.nk))) * self.stride[i]  # x
                    kpts_y = (kpts_xyconf[..., 1::3] + grid_y.repeat((1,1,1,1, self.nk))) * self.stride[i]  # y
                    kpts_conf = kpts_xyconf[..., 2::3].sigmoid()  # conf
                    y = torch.cat((xy, wh, conf.sigmoid(), kpts_x, kpts_y, kpts_conf), 4) 

                    # inplace
                    # kpts_xyconf[..., 0::3] = (kpts_xyconf[..., 0::3] + grid_x.repeat((1,1,1,1, self.nk))) * self.stride[i]  # x
                    # kpts_xyconf[..., 1::3] = (kpts_xyconf[..., 1::3] + grid_y.repeat((1,1,1,1, self.nk))) * self.stride[i]  # y
                    # kpts_xyconf[..., 2::3] = kpts_xyconf[..., 2::3].sigmoid()  # conf
                    # y = torch.cat((xy, wh, conf.sigmoid(), kpts_xyconf), 4)  

                z.append(y.view(-1, self.na * nx * ny, self.no))  # concat
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)  # x not do sigmoid(), while z did


    def _make_grid(self, nx=20, ny=20):
        d, t, na = self.stride.device, self.stride.dtype, self.na
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t), indexing='ij')
        else:
            yv, xv = torch.meshgrid(torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t))
        grid = torch.stack((xv, yv), 2).expand(1, na, ny, nx, 2)
        return grid


# TODO
class HydraX(nn.Module):
    # Hydra Head
    def __init__(self, c1, nc=80, nk=0, na=1):  # ch_in, num_classes, num_anchors, num_keypoints
        super().__init__()
        self.na, self.nc, self.nk = na, nc, nk    # number of anchors, classes, keypoints, segment masks
        # c_ = min(c1, 256)  # hidden layers
        c_ = c1

        # box branch
        self.conv_box = nn.Sequential(OrderedDict([
            ('dwconv', DWConv(c_, c_, 5)),      
            ('conv2d', nn.Conv2d(c_, 4, 1)),   # 4
        ]))

        # cls branch
        self.conv_cls = nn.Sequential(OrderedDict([
            ('dwconv', DWConv(c_, c_, 5)),
            ('conv2d', nn.Conv2d(c_, self.nc, 1)),  # nc + 1
        ]))

        # obj branch
        self.conv_obj = nn.Sequential(OrderedDict([
            ('dwconv', DWConv(c_, c_, 5)),
            ('conv2d', nn.Conv2d(c_, 1, 1)),  # nc + 1
        ]))

        # kpt branch
        if self.nk > 0:
            self.conv_kpt = nn.Sequential(OrderedDict([
            ('dwconv', DWConv(c_, c_, 5)),
            ('conv2d', nn.Conv2d(c_, self.nk * 3, 1)),
        ]))

    def forward(self, x):
        bs, nc, ny, nx = x.shape  # BCHW
        x_box = self.conv_box(x).view(-1, self.na, 4, ny, nx)           # box
        x_obj = self.conv_obj(x).view(-1, self.na, 1, ny, nx)           # obj
        x_cls = self.conv_cls(x).view(-1, self.na, self.nc, ny, nx)     # cls

        if self.nk > 0:  # kpts
            x_kpt = self.conv_kpt(x).view(bs, self.na, self.nk * 3, ny, nx)  
            y = torch.cat((x_box, x_obj, x_cls, x_kpt), 2)  
            return y.view(-1, self.na * (5 + self.nc + self.nk * 3), ny, nx)   # kpts, forcing dynamic batch size
        else:
            y = torch.cat((x_box, x_obj, x_cls), 2)     
            return y.view(-1, self.na * (5 + self.nc), ny, nx)  # det, forcing dynamic batch size



class Hydra(nn.Module):
    # Hydra Head
    def __init__(self, c1, nc=80, nk=0, na=1):  # ch_in, num_classes, num_anchors, num_keypoints
        super().__init__()
        self.na, self.nc, self.nk = na, nc, nk    # number of anchors, classes, keypoints, segment masks
        # c_ = min(c1, 256)  # hidden layers
        c_ = c1

        # shared block
        self.stem = Conv(c1, c_, 1)
        # self.stem = nn.Conv2d(c1, c_, 1, 1, autopad(1, None), groups=1, bias=False)
        # self.stem = nn.Sequential(OrderedDict([
        #     ('conv2d', nn.Conv2d(c1, c_, 1, 1, autopad(1, None), groups=1, bias=False)),
        #     ('cv1', Conv(c_, c_, 3)),
        #     # ('crossconv', CrossConv(c_, c_, 3)), 
        # ]))

        # box branch
        self.conv_box = nn.Sequential(OrderedDict([
            ('dwconv', DWConv(c_, c_, 5)),  # TODO: remove or not??
            ('conv2d', nn.Conv2d(c_, 4, 1)),   # 4

        ]))

        # cls branch + obj branch
        self.conv_cls = nn.Sequential(OrderedDict([
            ('dwconv', DWConv(c_, c_, 5)),
            ('conv2d', nn.Conv2d(c_, self.nc, 1)),  # nc + 1
        ]))

        # TODO: obj branch
        self.conv_obj = nn.Sequential(OrderedDict([
            ('dwconv', DWConv(c_, c_, 5)),
            ('conv2d', nn.Conv2d(c_, 1, 1)),  # nc + 1
        ]))

        # kpt branch
        if self.nk > 0:
            self.conv_kpt = nn.Sequential(OrderedDict([
            ('dwconv', DWConv(c_, c_, 5)),
            ('conv2d', nn.Conv2d(c_, self.nk * 3, 1)),
        ]))


    def forward(self, x):
        bs, nc, ny, nx = x.shape  # BCHW
        x = self.stem(x)
        x_box = self.conv_box(x).view(-1, self.na, 4, ny, nx)   # box
        x_obj = self.conv_obj(x).view(-1, self.na, 1, ny, nx)     # obj
        x_cls = self.conv_cls(x).view(-1, self.na, self.nc, ny, nx)     # cls

        if self.nk > 0:  # kpts
            x_kpt = self.conv_kpt(x).view(bs, self.na, self.nk * 3, ny, nx)  
            y = torch.cat((x_box, x_obj, x_cls, x_kpt), 2)
            # y = torch.cat((x_box, x_cls, x_obj, x_kpt), 2)  # TODO : modify
            return y.view(-1, self.na * (5 + self.nc + self.nk * 3), ny, nx)   # kpts, forcing dynamic batch size
        else:
            y = torch.cat((x_box, x_obj, x_cls), 2)     
            # y = torch.cat((x_box, x_cls, x_obj), 2)     # TODO : modify
            return y.view(-1, self.na * (5 + self.nc), ny, nx)  # det, forcing dynamic batch size
        # return y.view(bs, -1, ny, nx)


    # TODO: remove!
    # def forward(self, x):
    #     bs, nc, ny, nx = x.shape  # BCHW
    #     x = self.stem(x)
    #     x_box = self.conv_box(x).view(-1, self.na, 4, ny, nx)   # box
    #     x_obj = self.conv_obj(x).view(-1, self.na, 1, ny, nx)     # obj
    #     x_cls = self.conv_cls(x).view(-1, self.na, self.nc, ny, nx)     # cls

    #     if self.nk > 0:  # kpts
    #         x_kpt = self.conv_kpt(x).view(bs, self.na, self.nk * 3, ny, nx)  
    #         # y = torch.cat((x_box, x_obj, x_cls, x_kpt), 2)
    #         y = torch.cat((x_box, x_cls, x_obj, x_kpt), 2)  # TODO : modify
    #         return y.view(-1, self.na * (5 + self.nc + self.nk * 3), ny, nx)   # kpts, forcing dynamic batch size
    #     else:
    #         # y = torch.cat((x_box, x_obj, x_cls), 2)     
    #         y = torch.cat((x_box, x_cls, x_obj), 2)     # TODO : modify
    #         return y.view(-1, self.na * (5 + self.nc), ny, nx)  # det, forcing dynamic batch size
    #     # return y.view(bs, -1, ny, nx)