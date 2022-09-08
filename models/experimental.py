"""
Experimental modules
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import *



class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)



class RepBottleneck(nn.Module):
    # -----------------------------|
    #   Bottleneck
    # -----------------------------|
    # identity(bn), c -----------------| ====> 3x3, c Conv
    # 1x1 Conv, c/2 + 3x3 Conv c, -|
    # -----------------------------|

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, e=0.5, act=True, has_identity=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        from collections import OrderedDict

        assert k % 2 != 0, 'Not support for uneven-k!'
        assert g == 1, 'Not support for group conv!'
  
        
        # 1x1 Conv + kxk Conv
        c_ = int(c2 * e)  # hidden channels
        self.pwconv_kconv = nn.Sequential(OrderedDict([
            ('pwconv', nn.Conv2d(c1, c_, 1, 1, self._autopad(k, p), groups=g, bias=False)),
            ('bn1', nn.BatchNorm2d(c_)),
            ('kconv', nn.Conv2d(c_, c2, k, s, 0, groups=g, bias=False)),
            ('bn2', nn.BatchNorm2d(c2)),
        ]))

        # identity
        self.add = has_identity
        if self.add:
            self.bn = nn.BatchNorm2d(c1)

        # act
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        # attrs
        self.c1, self.c2,  self.g, self.k, self.s, self.p = c1, c2, g, k, s, p

    def forward(self, x):
        if hasattr(self, 'fusedconv'):
            y = self.fusedconv(x)
        else:
            y = self.pwconv_kconv(x)
            y = y + self.bn(x) if self.add else y

        return self.act(y)

    def fuse(self):

        # fused kxk conv2d
        if not hasattr(self, 'fusedconv'):   
            self.fusedconv = nn.Conv2d(self.c1, 
                                        self.c2, 
                                        self.k, 
                                        self.s, 
                                        self._autopad(self.k, self.p), 
                                        self.g, 
                                        bias=True).requires_grad_(False).to(self.pwconv_kconv.pwconv.weight.device)

        # weights & bias transfer
        weight, bias = self.get_equivalent_weight_bias()
        self.fusedconv.weight.data = weight
        self.fusedconv.bias.data = bias
        for para in self.parameters():
            para.detach_()

        # del conv modules
        convs = [x for x in self._modules.keys() if 'conv' in x and x != 'fusedconv' or 'bn' in x]  # keep activation and fusedconv
        for l in convs:
            if hasattr(self, l):
                self.__delattr__(l)
        # print(self._modules.keys())


    def get_equivalent_weight_bias(self):

        # (pwconv + bn) + (kconv + bn)
        w_pwconv_, b_pwconv_ = self._fuse_conv_bn(self.pwconv_kconv.pwconv.weight, self.pwconv_kconv.bn1)   # fused pwconv
        w_kconv_, b_kconv_ = self._fuse_conv_bn(self.pwconv_kconv.kconv.weight, self.pwconv_kconv.bn2)     # fused kconv
        w_pwconv_kconv_bn, b_pwconv_kconv_bn = self._fuse_seq_pwconv_kconv(w_pwconv_, b_pwconv_, w_kconv_, b_kconv_)

        # identity
        if self.add:

            # identity to kconv
            w_identity_bn, b_identity_bn = self._identity_to_kconv_bn(k=3, bn=self.bn, device=self.pwconv_kconv.pwconv.weight.device)

            # fuse all branch kconv -> add directly
            return w_pwconv_kconv_bn + w_identity_bn, b_pwconv_kconv_bn + b_identity_bn

        # fuse all branch kconv -> add directly
        return w_pwconv_kconv_bn, b_pwconv_kconv_bn


    def _autopad(self, k, p=None):  # kernel, padding
        # Pad to 'same'
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        return p


    def _fuse_conv_bn(self, conv_weight=None, bn=None):
        # fuse conv and bn
        if conv_weight is None or bn is None:
            return 0, 0
        std = (bn.running_var + bn.eps).sqrt()
        return conv_weight * (bn.weight / std).reshape((-1, 1, 1, 1)), bn.bias - bn.running_mean * bn.weight / std 


    def _pwconv_to_kconv_padding(self, w_pwconv, k=None):
        # pad pwconv or asym-conv to kconv 
        if w_pwconv is None:
            return 0
        else:
            if k is None:
                k = w_pwconv.size()[-2:]    # get kernel_size(h, w)
            p_ = self._autopad(k)
            pad_ = [p_] * 4 if isinstance(p_, int) else [p_[0], p_[0], p_[1], p_[1]]
            return nn.functional.pad(w_pwconv, pad_)   # pad to kxk shape


    def _fuse_seq_pwconv_kconv(self, w_pwconv, b_pwconv, w_kconv, b_kconv):
        # fuse pwconv + kconv
        # x -> pwconv -> y1 -> kconv -> y2
        # pwconv: y1 = w1 * x + b1  |  kconv: y2 = w2 * y1 + b2 
        # y2 = (w2*w1)*x + (w2*b1+b2) 
        w_pwconv_kconv = F.conv2d(w_kconv, w_pwconv.permute(1, 0, 2, 3))    # weights fuse
        b_pwconv_kconv = (w_kconv * b_pwconv.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b_kconv   # bias fuse
        return w_pwconv_kconv, b_pwconv_kconv


    def _identity_to_kconv_bn(self, k, bn, device):
        # turn identity branch to kconv+bn

        # identity to kconv
        input_dim = self.c1 // self.g
        kernel = np.zeros((self.c1, input_dim, k, k), dtype=np.float32)    # empty kernel

        for i in range(self.c1):
            kernel[i, i % input_dim, 1, 1] = 1
        w_identity = torch.from_numpy(kernel).to(device)

        # fuse identity & bn
        w_identity_bn, b_identity_bn = self._fuse_conv_bn(conv_weight=w_identity, bn=bn)

        return w_identity_bn, b_identity_bn


    def _kconv_concat(w_list, b_list):
        # concat(kconv, kconv)
        return torch.cat(w_list, dim=0), torch.cat(b_list)


    def verify(self, x, verbose=True):
        '''
        Uasge:
            x = torch.rand(1, 3, 640, 640).to(device)
            repconvs = RepConvs(3, 64, 3, 2).to(device)
            # repconvs.fuse()
            repconvs.verify(x, verbose=True)
        '''

        for module in self.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                nn.init.uniform_(module.running_mean, 0, 0.1)
                nn.init.uniform_(module.running_var, 0, 0.1)
                nn.init.uniform_(module.weight, 0, 0.1)
                nn.init.uniform_(module.bias, 0, 0.1)
        self.eval()

        if verbose:
            print(f"\n{'-' * 20}\n un-fused \n{'-'*20}\n")
            print(self)
        y = self.forward(x)

        self.fuse()
        if verbose:
            print(f"\n{'-' * 20}\n fused \n{'-'*20}\n")
            print(self)
        y_fused = self.forward(x)

        print(f'>> Difference betweeen y and y_fused: {((y_fused - y) ** 2).sum()}')

import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model




class RepConvs(nn.Module):
    # ----------------------------|
    #   Inception Like Conv
    # ----------------------------|
    # 1x1 Conv -------------------| ====> kxk Conv
    # 1x1 Conv + kxk Conv --------|
    # 1x1 Conv + Avg -------------|
    # kxk Conv -------------------|
    # 1xk Conv -------------------|
    # kx1 Conv -------------------|
    # ----------------------------|


    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, e=0.5, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        from collections import OrderedDict

        assert k % 2 != 0, 'Not support for uneven-k!'
        assert g == 1, 'Not support for group conv!'

        # 1x1 Conv
        self.pwconv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(c1, c2, 1, s, 0, groups=g, bias=False)),
            ('bn', nn.BatchNorm2d(c2))
        ]))   
            
        # kxk Conv
        self.kconv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(c1, c2, k, s, self._autopad(k, p), groups=g, bias=False)),
            ('bn', nn.BatchNorm2d(c2))
        ]))  

        # 1x1 Conv + avg
        self.pwconv_avg = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(c1, c2, 1, 1, self._autopad(k, p), groups=g, bias=False)),
            ('bn1', nn.BatchNorm2d(c2)),
            ('avg', nn.AvgPool2d(k, s, 0)),
            ('bn2', nn.BatchNorm2d(c2)),
        ]))

        # 1x1 Conv + kxk Conv
        c_ = int(c2 * e)  # hidden channels
        self.pwconv_kconv = nn.Sequential(OrderedDict([
            ('pwconv', nn.Conv2d(c1, c_, 1, 1, self._autopad(k, p), groups=g, bias=False)),
            ('bn1', nn.BatchNorm2d(c_)),
            ('kconv', nn.Conv2d(c_, c2, k, s, 0, groups=g, bias=False)),
            ('bn2', nn.BatchNorm2d(c2)),
        ]))

        # AsymConv: 1xk
        self.aysmconv_1xk = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(c1, c2, (1, k), s, self._autopad((1, k), p), groups=g, bias=False)),
            ('bn', nn.BatchNorm2d(c2))

        ])) 

        # AsymConv: kx1
        self.aysmconv_kx1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(c1, c2, (k, 1), s, self._autopad((k, 1), p), groups=g, bias=False)),
            ('bn', nn.BatchNorm2d(c2))

        ])) 

        # act
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())


    def forward(self, x):
        if hasattr(self, 'fusedconv'):
            y = self.fusedconv(x)
        else:
            y = self.pwconv(x) 
            y += self.kconv(x) 
            y += self.pwconv_avg(x) 
            y += self.pwconv_kconv(x) 
            y += self.aysmconv_1xk(x) 
            y += self.aysmconv_kx1(x)
        return self.act(y)

    def fuse(self):

        # fused kxk conv2d
        if not hasattr(self, 'fusedconv'):   
            self.fusedconv = nn.Conv2d(self.kconv.conv.in_channels, 
                                        self.kconv.conv.out_channels, 
                                        self.kconv.conv.kernel_size, 
                                        self.kconv.conv.stride, 
                                        self.kconv.conv.padding, 
                                        self.kconv.conv.groups, 
                                        bias=True).requires_grad_(False).to(self.kconv.conv.weight.device)

        # weights & bias transfer
        weight, bias = self.get_equivalent_weight_bias()
        self.fusedconv.weight.data = weight
        self.fusedconv.bias.data = bias
        for para in self.parameters():
            para.detach_()

        # del conv modules
        convs = [x for x in self._modules.keys() if 'conv' in x and x != 'fusedconv']  # keep activation and fusedconv
        for l in convs:
            if hasattr(self, l):
                self.__delattr__(l)
        # print(self._modules.keys())


    def get_equivalent_weight_bias(self):

        # 1. pwconv fuse bn
        w_pwconv_bn, b_pwconv_bn = self._fuse_conv_bn(self.pwconv.conv.weight, self.pwconv.bn)  # fused first
        w_pwconv_bn = self._pwconv_to_kconv_padding(w_pwconv_bn, self.kconv.conv.kernel_size)  # pad to kxk shape

        # 2. kconv fuse bn
        w_kconv_bn, b_kconv_bn = self._fuse_conv_bn(self.kconv.conv.weight, self.kconv.bn)

        # 3. (pwconv + bn) + (kconv + bn)
        w_pwconv_, b_pwconv_ = self._fuse_conv_bn(self.pwconv_kconv.pwconv.weight, self.pwconv_kconv.bn1)   # fused pwconv
        w_kconv_, b_kconv_ = self._fuse_conv_bn(self.pwconv_kconv.kconv.weight, self.pwconv_kconv.bn2)     # fused kconv
        w_pwconv_kconv_bn, b_pwconv_kconv_bn = self._fuse_seq_pwconv_kconv(w_pwconv_, b_pwconv_, w_kconv_, b_kconv_)


        # 4. (pwconv + bn) + (avg + bn)
        w_pwconv, b_pwconv = self._fuse_conv_bn(self.pwconv_avg.conv.weight, self.pwconv_avg.bn1)   # fused pwconv
        avg_kconv_ = self._avg_to_kconv(self.pwconv_avg.conv, self.pwconv_avg.avg)   # avg -> kconv
        w_kconv, b_kconv = self._fuse_conv_bn(avg_kconv_.to(self.pwconv_avg.conv.weight.device), self.pwconv_avg.bn2)     # fused avg_kconv
        w_pwconv_avg_bn, b_pwconv_avg_bn = self._fuse_seq_pwconv_kconv(w_pwconv, b_pwconv, w_kconv, b_kconv)

        # 5. asym-conv
        w_asymconv_1xk_bn, b_asymconv_1xk_bn = self._fuse_conv_bn(self.aysmconv_1xk.conv.weight, self.aysmconv_1xk.bn)   # fused 1xk asym-conv
        w_asymconv_1xk_bn = self._pwconv_to_kconv_padding(w_asymconv_1xk_bn)
        
        w_asymconv_kx1_bn, b_asymconv_kx1_bn = self._fuse_conv_bn(self.aysmconv_kx1.conv.weight, self.aysmconv_kx1.bn)   # fused kx1 asym-conv
        w_asymconv_kx1_bn = self._pwconv_to_kconv_padding(w_asymconv_kx1_bn)

        # # 6. (kconv + bn) + (pwconv + bn)
        # w_kconv__, b_kconv__ = self._fuse_conv_bn(self.kconv_pwconv.kconv.weight, self.kconv_pwconv.bn1)     # fused kconv
        # w_pwconv__, b_pwconv__ = self._fuse_conv_bn(self.kconv_pwconv.pwconv.weight, self.kconv_pwconv.bn2)   # fused pwconv
        # w_kconv_pwconv_bn, b_kconv_pwconv_bn = self._fuse_seq_pwconv_kconv(w_kconv__, b_kconv__, w_pwconv__, b_pwconv__)


        # fuse all branch kconv -> add directly
        return (sum((w_pwconv_bn, w_kconv_bn, w_pwconv_kconv_bn, w_pwconv_avg_bn, w_asymconv_1xk_bn, w_asymconv_kx1_bn)), 
                sum((b_pwconv_bn, b_kconv_bn, b_pwconv_kconv_bn, b_pwconv_avg_bn, b_asymconv_1xk_bn, b_asymconv_kx1_bn)))


    def _autopad(self, k, p=None):  # kernel, padding
        # Pad to 'same'
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        return p


    def _fuse_conv_bn(self, conv_weight=None, bn=None):
        # fuse conv and bn
        if conv_weight is None or bn is None:
            return 0, 0
        std = (bn.running_var + bn.eps).sqrt()
        return conv_weight * (bn.weight / std).reshape((-1, 1, 1, 1)), bn.bias - bn.running_mean * bn.weight / std 


    def _pwconv_to_kconv_padding(self, w_pwconv, k=None):
        # pad pwconv or asym-conv to kconv 
        if w_pwconv is None:
            return 0
        else:
            if k is None:
                k = w_pwconv.size()[-2:]    # get kernel_size(h, w)
            p_ = self._autopad(k)
            pad_ = [p_] * 4 if isinstance(p_, int) else [p_[0], p_[0], p_[1], p_[1]]
            return nn.functional.pad(w_pwconv, pad_)   # pad to kxk shape


    def _fuse_seq_pwconv_kconv(self, w_pwconv, b_pwconv, w_kconv, b_kconv):
        # fuse pwconv + kconv
        # x -> pwconv -> y1 -> kconv -> y2
        # pwconv: y1 = w1 * x + b1  |  kconv: y2 = w2 * y1 + b2 
        # y2 = (w2*w1)*x + (w2*b1+b2) 
        w_pwconv_kconv = F.conv2d(w_kconv, w_pwconv.permute(1, 0, 2, 3))    # weights fuse
        b_pwconv_kconv = (w_kconv * b_pwconv.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b_kconv   # bias fuse
        return w_pwconv_kconv, b_pwconv_kconv


    # TODO:
    # def _fuse_seq_kconv_pwconv(self, w_pwconv, b_pwconv, w_kconv, b_kconv):
    # def _fuse_seq_kconv_kconv(self, w_pwconv, b_pwconv, w_kconv, b_kconv):



    def _avg_to_kconv(self, kconv, avg):
        # turn avg to kconv, k=avg.kernel_size
        input_dim = kconv.out_channels // kconv.groups      # transfrom avg_pool to kconv
        avg_kconv = torch.zeros((kconv.out_channels, input_dim, avg.kernel_size, avg.kernel_size))  # fused kconv
        avg_kconv[np.arange(kconv.out_channels), np.tile(np.arange(input_dim), kconv.groups), :, :] = 1.0 / avg.kernel_size ** 2
        return avg_kconv     


    def _kconv_concat(w_list, b_list):
        # concat(kconv, kconv)
        return torch.cat(w_list, dim=0), torch.cat(b_list)


    def verify(self, x, verbose=True):
        '''
        Uasge:
            x = torch.rand(1, 3, 640, 640).to(device)
            repconvs = RepConvs(3, 64, 3, 2).to(device)
            # repconvs.fuse()
            repconvs.verify(x, verbose=True)
        '''

        for module in self.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                nn.init.uniform_(module.running_mean, 0, 0.1)
                nn.init.uniform_(module.running_var, 0, 0.1)
                nn.init.uniform_(module.weight, 0, 0.1)
                nn.init.uniform_(module.bias, 0, 0.1)
        self.eval()

        if verbose:
            print(f"\n{'-' * 20}\n un-fused \n{'-'*20}\n")
            print(self)
        y = self.forward(x)

        self.fuse()
        if verbose:
            print(f"\n{'-' * 20}\n fused \n{'-'*20}\n")
            print(self)
        y_fused = self.forward(x)

        print(f'>> Difference betweeen y and y_fused: {((y_fused - y) ** 2).sum()}')



# ------------------------------------------------
#   ConvNext
# ------------------------------------------------
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
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


class PatchDown(nn.Module):
    # LayerNorm + nn.Conv2d
    def __init__(self, c1, c2, k=2):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, k)  #  k = s 
        self.norm = LayerNorm(c1, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        return self.conv(self.norm(x))


class ConvNextBlock(nn.Module):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, c1, c2, dp_rate=0.4, e=4):
        super().__init__()
        c_ = c1 * e
        # self.dwconv = nn.Conv2d(c1, c1, kernel_size=7, padding=3, groups=c1) # depthwise conv
        self.dwconv = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1) # depthwise conv
        self.norm = LayerNorm(c1, eps=1e-6)
        self.pwconv1 = nn.Linear(c1, c_) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(c_, c2)

        # self.layer_scale_init_value = 1.0
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((c)), requires_grad=True) if layer_scale_init_value > 0 else None

        # 4: [3, 3, 9, 3] : P2(128) P3(256) P4(512) P5(1024)
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(3, 3, 9, 3))]    # TODO   stage ratio = (3, 3, 9, 3)
        self.drop_path = DropPath(dp_rate) if dp_rate > 0.0 else nn.Identity()


    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
            # x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)

        return x


class ConvNext(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, dp_rate=0.7, e=4
                    # shortcut=True, g=1, e=0.5
                ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.m = nn.Sequential(*(ConvNextBlock(c1, c2, dp_rate, e) for _ in range(n)))

    def forward(self, x):
        return self.m(x)


# ------------------------------------------------
#   ConvNext
# ------------------------------------------------

class Focus(nn.Module):
    # Focus wh information into c-space
    # Params      GFLOPs  GPU_mem (GB)  forward (ms) backward (ms)                 input                  output
    # 7040       1.468         0.132         1.736         2.868        (1, 3, 640, 640)       (1, 64, 320, 320)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class PatchConv(nn.Module):
    def __init__(self, c1, c2, k=2, s=2, p=0, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, act)  #  k = s Conv
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())  # set nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Patchify(nn.Module):
    def __init__(self, c1, c2, k=2, p=0, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1, c2, k, k, p, g, act)  #  k = s Conv

    def forward(self, x):  
        return self.conv(x)


class SPD(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self):
        super().__init__()

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class SA(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, groups=64):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        # flatten
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

class CrossConvSA(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, attention=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2
        self.sa = SA(c2)
        self.attention = attention

    def forward(self, x):
        if self.attention:
            return x + self.sa(self.cv2(self.cv1(x))) if self.add else self.sa(self.cv2(self.cv1(x)))
        else:
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3xSA(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConvSA(c_, c_, 3, 1, g, 1.0, shortcut, True) for _ in range(n)))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)



class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([
            nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))



# ------------------------------------------------
#   yolov7
# ------------------------------------------------
class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)
    

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))  
# ------------------------------------------------
#   yolov7
# ------------------------------------------------



class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        super().__init__()
        if verbose:
            LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else torch.zeros(1, device=self.model.device)  # for device, type
        autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), list(imgs)) if isinstance(imgs, (list, tuple)) else (1, [imgs])  # number, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, self.stride) if self.pt else size for x in np.array(shape1).max(0)]  # inf shape
        x = [letterbox(im, shape1, auto=False)[0] for im in imgs]  # pad
        x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(autocast):
            # Inference
            y = self.model(x, augment, profile)  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y if self.dmb else y[0],
                                    self.conf,
                                    self.iou,
                                    self.classes,
                                    self.agnostic,
                                    self.multi_label,
                                    max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label if labels else '', color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self, labels=True):
        self.display(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        self.display(render=True, labels=labels)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n  # override len(results)

    def __str__(self):
        self.print()  # override print(results)
        return ''


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)




