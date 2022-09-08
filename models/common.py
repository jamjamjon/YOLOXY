"""
Common blocks & modules 
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
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, CONSOLE, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync
from utils.downloads import attempt_download


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
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())  # set nn.LeakyReLU()
        # self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())  

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
        # self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.focus = SPD() if focus is True else nn.Identity()


    def forward(self, x):
        if hasattr(self, 'fusedconv'):
            y = self.fusedconv(x)
        else:
            y = self.conv1x1(x) + self.conv3x3(x)
        return self.act(y)
        # return self.focus(self.act(y))

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
        # self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.focus = SPD() if focus is True else nn.Identity()


    def forward(self, x):
        if hasattr(self, 'fusedconv'):
            y = self.fusedconv(x)
        else:
            
            y_kxk = self.convkxk(x)
            y_kx1 = self.convkx1(x)
            y_1xk = self.conv1xk(x)
            y = y_kxk + y_kx1 + y_1xk

        return self.act(y)
        # return self.focus(self.act(y))


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
        # self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())  # nn.LeakyReLU()

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


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output



def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.yolo import Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
        ckpt = torch.load(w, map_location='cpu')  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
        model.append(ckpt.fuse().eval() if fuse else ckpt.eval())  # fused or un-fused model in eval mode

    # Compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            # if t is Detect and not isinstance(m.anchor_grid, list):
            #     delattr(m, 'anchor_grid')
            #     setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is Conv:
            m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml', 'nk', 'kpt_kit':         
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model  # return ensemble


def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class DetectMultiBackend(nn.Module):
    # YOLO MultiBackend class for python inference on various backends
    def __init__(self, weights='', device=torch.device('cpu'), dnn=False, data=None, fp16=False):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx with --dnn

        super().__init__()
        # from models.experimental import attempt_load, attempt_download  # scoped to avoid circular import

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



