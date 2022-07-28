"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg xxx.yaml
"""

import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
import rich


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.general import check_version, check_yaml, make_divisible, print_args, LOGGER, CONSOLE
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Model(nn.Module):
    # YOLO model
    def __init__(self, cfg=None, ch=3, nc=None, nk=None): 
        super().__init__()

        # read model.yaml
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels

        # num of classes
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"{colorstr(f'Overriding model.yaml') } nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value


        # num of keypoints
        if nk and nk != self.yaml.get('nk', 0):
            LOGGER.info(f"{colorstr(f'Overriding model.yaml')} nk={self.yaml.get('nk', 0)} with nk={nk}")
            self.yaml.update({'nk': nk})  # override yaml value

        # parse model
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default det names
        self.inplace = self.yaml.get('inplace', True)

        # TODO
        # self.task = 'kpts' if nk > 0 else 'det'

        # Build strides, anchors
        m = self.model[-1]  # Head 
        if isinstance(m, (DetectX)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            self._initialize_biases()  # only run once


        # Init weights, biases
        initialize_weights(self)
        self.info()    # will infer once
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, (DetectX))  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    # def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
    #     # https://arxiv.org/abs/1708.02002 section 3.3
    #     # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
    #     m = self.model[-1]  # Detect() module
    #     for mi, s in zip(m.m, m.stride):  # from
    #         b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
    #         b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
    #         b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
    #         mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
            # https://arxiv.org/abs/1708.02002 section 3.3
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
            m = self.model[-1]  # Detect() module
            for mi, s in zip(m.m, m.stride):  # from
                
                # decoupled head
                if type(mi) is Decouple:
                    # obj
                    b = mi.b2.bias.view(m.na, -1)   # conv.bias(3*1) to (3,1)
                    b.data[:] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                    mi.b2.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
                    # box
                    # cls
                    b = mi.c.bias.data
                    b += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
                    mi.c.bias = torch.nn.Parameter(b, requires_grad=True)
                    
                # decoupled head
                elif type(mi) is HydraHead:
                    # obj
                    b = mi.conv_obj.conv2d.bias.view(m.na, -1)   # conv.bias(3*1) to (3,1)
                    b.data[:] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                    mi.conv_obj.conv2d.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
                    # box
                    # cls
                    b = mi.conv_cls.conv2d.bias.data
                    b += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
                    mi.conv_cls.conv2d.bias = torch.nn.Parameter(b, requires_grad=True)

               
                # coupled head
                else:  # default
                    b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                    b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                    b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
                    mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info(f"{colorstr('bright_cyan', 'Fusing layers...')}")

        for m in self.model.modules():   # TODO: move to Conv()
            # Conv, DWConv 
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
            # RepConv
            if isinstance(m, RepConv):
                m.fuse_repconv()
            # AsymConv
            if isinstance(m, AsymConv):
                m.fuse_asymconv()


        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        # if isinstance(m, Detect):
        #     m.stride = fn(m.stride)
        #     m.grid = list(map(fn, m.grid))
        #     if isinstance(m.anchor_grid, list):
        #         m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict(.yaml), input_channels(3)
    # CONSOLE.log(log_locals=True)      # local variables

    # rich table
    model_table = rich.table.Table(highlight=False, box=rich.box.ROUNDED)
    model_attrs = {
        "IDX": "right",
        "FROM": "right",
        "N": "left",
        "PARAMS": "right",
        "MODULE": "left",
        "ARGUMENTS": "left",
    }
    for k, v in model_attrs.items():
        model_table.add_column(f"{k}", justify=v, style="", no_wrap=True)

    # params
    na, nc, nk, gd, gw = 1, d['nc'], d['nk'], d['depth_multiple'], d['width_multiple']
    no = na * (nc + 5 + nk * 2)   # number of outputs = anchors * (classes + 5 + 2 * keypoints)
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # model structure
    if d.get('neck', None) is None:
        struct = d['backbone'] + d['head']
    elif d.get('neck', None) is not None:
        struct = d['backbone'] + d['neck'] + d['head']
    
    # parse
    for i, (f, n, m, args) in enumerate(struct):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in (Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x,
                 RepConv, C3xSA, CrossConvSA, SPPCSPC, C3xESE, AsymConv   # update
                 ):
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost, C3x, C3xSA, SPPCSPC, C3xESE]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)

        # Detect for yolox
        elif m is DetectX:
            args.append([ch[x] for x in f])

        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params

        # save model structure to table
        model_table.add_row(str(i), str(f), str(n_), str(np), str(t), str(args))  # TODO args too long with anchors

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    CONSOLE.print(model_table)    # show model structure
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size h,w')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    # parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    parser.add_argument('--detail', action='store_true', help='print model')
    parser.add_argument('--fuse', action='store_true', help='fuse model')
    parser.add_argument('--check', action='store_true', help='check ops')
    parser.add_argument('--output', action='store_true', help='show output shape')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    # print_args(vars(opt))

    device = select_device(opt.device)   # device
    im = torch.rand(opt.batch_size, 3, opt.imgsz, opt.imgsz).to(device)     # dummpy input


    # Options
    if opt.cfg:

        # build model
        model = Model(opt.cfg).to(device)

        # fuse
        if opt.fuse:
            model.fuse()

        # print model
        if opt.detail:
            LOGGER.info(model)
        
        # profile layer by layer
        if opt.line_profile:  
            _ = model(im, profile=True)

        # profile forward-backward
        if opt.profile:  
            results = profile(input=im, ops=[model], n=50)

        # output shape
        if opt.output:
            y = model(im)
            for y_ in y:
                LOGGER.info(f"Output Shape: {y_.shape}")
    else:
        LOGGER.info(f"{colorstr('No cfg...')}")



    # playground
    if opt.check:

        # # fused RepConv
        # repconv = RepConv(3, 128, 3, 2)
        # repconv.fuse_repconv()

        # # fused Conv
        # conv = Conv(3, 128, 3, 2)
        # conv.conv = fuse_conv_and_bn(conv.conv, conv.bn)  # update conv
        # delattr(conv, 'bn')  # remove batchnorm
        # conv.forward = conv.forward_fuse

        # rep = RepVGGBlock(3, 128, 3, 2)
        # c3x = C3x(3, 128)
        # c3xese = C3xESE(3, 128, ese=False)
        # ese = ESE(3)

        # c3tr = C3TR(3, 32)


        # x = torch.rand(opt.batch_size, 3, 32, 32).to(device)
        # # _ = profile(input=x, ops=[c3tr], n=1, device=device)
        # c_ = 32
        # a = nn.Sequential(DWConv(c_, c_, 3), Conv(c_, c_, 1))
        # print(type(a))
        # print(a)

        # b = Conv(3, 32, 3)    # fused conv 

        # print(type(b))

        x = torch.rand(1, 64, 32, 32).to(device)
        h = DecoupleH(64, nc=80, na=1, nk=17)
        d = Decouple(64, nc=80, na=1)
        print(h)
        # print(d)
        _ = profile(input=x, ops=[h, d], n=80, device=device)



    # # test all models
    # elif opt.test:  
    #     for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
    #         try:
    #             _ = Model(cfg)
    #         except Exception as e:
    #             print(f'Error in {cfg}: {e}')





