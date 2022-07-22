

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import torch
import yaml
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.yolo import DetectX
from models.common import *
from models.activations import *
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_version, colorstr,
                           file_size, print_args, url2file, increment_path)
from utils.torch_utils import select_device


def export_formats():
    # YOLOv5 export formats
    x = [
            ['PyTorch', '-', '.pt', True],
            ['ONNX', 'onnx', '.onnx', True],
            ['RKNN', 'rknn', '.rknn']
            # ['TorchScript', 'torchscript', '.torchscript', True],
            # ['TensorRT', 'engine', '.engine', True],
            # ['CoreML', 'coreml', '.mlmodel', False],
        ]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'GPU'])



def export_onnx(model, im, file, opset, train, dynamic, simplify, save_dir, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    try:
        check_requirements(('onnx',))
        import onnx

        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        # f = file.with_suffix('.onnx')
        f = save_dir / (file.with_suffix('.onnx').name)

        torch.onnx.export(
            model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
            im.cpu() if dynamic else im,
            f,
            verbose=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=not train,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'},  # shape(1,3,640,640)
                'output': {
                    0: 'batch',
                    1: 'anchors'}  # shape(1,25200,85)
            } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Metadata
        d = {'stride': int(max(model.stride)), 'names': model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(model_onnx, f)

        # Simplify
        if simplify:
            try:
                check_requirements(('onnx-simplifier',))
                import onnxsim

                LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     dynamic_input_shape=dynamic,
                                                     input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')



# export rknn & onnx
def export_rknn(file,
                calibration,   # images path.txt  or dir
                keep_onnx=True,
                save_dir='',
                prefix=colorstr('RKNN:')
                ):

    try:
        from rknn.api import RKNN
        # LOGGER.info(f'\n{prefix} starting export with rknn {rknn.__version__}...')

        # check onnx file is exist
        assert Path(file).suffix == '.onnx' and Path(file).exists(), "[RKNN CONVERT ERROR] to convert rknn model must using ONNX file."

        # rknn qnt calibration
        qnt_cali_file = None   # qnt cali file(.txt)
        if calibration and Path(calibration).exists():
            if Path(calibration).is_file():  # for txt file
                qnt_cali_file = calibration
                LOGGER.info(f"{colorstr('RKNN QNT calibration with file:')} {Path(calibration).resolve()}")
            elif Path(calibration).is_dir():  # for image directory
                qnt_cali_file = '.cali.txt'   # create invisiable temp file

                # delete cali txt if already has one
                if Path(qnt_cali_file).exists() and Path(qnt_cali_file).is_file():
                    Path(qnt_cali_file).unlink()

                # save images path to qnt_cali_file(.cali.txt)
                image_list = [str(x.resolve()) for x in Path(calibration).iterdir() if x.suffix.lower() in ('.png', '.jpg', '.jpeg')]
                for p in tqdm(image_list, ncols=100, desc=colorstr('Generating RKNN QNT calibration file')):
                    with open(qnt_cali_file, 'a') as f_temp:
                        f_temp.write(p + '\n')
                LOGGER.info(f"{colorstr('RKNN QNT calibration with directory:')} {Path(calibration).resolve()}")

        else:
            LOGGER.info(f"{colorstr('b', 'magenta', 'NO RKNN QNT calibration file!')}")


        # TODO: cali txt check, make sure all images can be used when do qnt(e.g. chinese path name. wrong suffix, decreapte format, ...)
        # cali_file_check()


        # load rknn & config
        rknn = RKNN(verbose=True)
        rknn.config(channel_mean_value='0 0 0 255',   # 123.675 116.28 103.53 58.395 # 0 0 0 255 # 
                    reorder_channel='0 1 2',          # '0 1 2' '2 1 0'
                    batch_size=1,
                    epochs=-1,
                    quantized_dtype='asymmetric_quantized-u8', 
                    optimization_level=3,
                    # target_platform=['rk3399pro'],    # 不写则默认是['rk1808'],生成的可以在 RK1806、RK1808 和 RK3399Pro 平台上运行
                    )

        # load onnx
        ret = rknn.load_onnx(file)
        if ret != 0:
            LOGGER.info(f'\nLoad yolo ONNX failed! Ret = {ret}.')
            exit(ret)

        # rknn build
        ret = rknn.build(do_quantization=True if qnt_cali_file else False,
                         dataset=qnt_cali_file,
                         pre_compile=False)
        if ret != 0:
            LOGGER.info(f'\nRKNN build failed! Ret = {ret}.')
            exit(ret)
        
        # export rknn model
        f = file.with_suffix('.rknn')
        ret = rknn.export_rknn(f)
        if ret != 0:
            LOGGER.info(f'\nExport rknn model failed! Ret = {ret}.')

        # delete invisiable qnt temp file
        if qnt_cali_file and Path(qnt_cali_file).name[0] == '.':
            Path(qnt_cali_file).unlink()


        # delete onnx file
        if not keep_onnx:
            file.unlink()

        return f

    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')
    



@torch.no_grad()
def run(
        weights='',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=(''),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        train=False,  # model.train() mode
        dynamic=False,  # ONNX/TF: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=11,  # ONNX: opset version
        project=ROOT / 'runs/export',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        cali='',
):


    # save dirs
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()['Argument'][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'


    # add new type here
    onnx, rknn = flags  # export booleans
    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    if half:
        # assert device.type != 'cpu' or coreml or xml, '--half only compatible with GPU export, i.e. use --device 0'
        assert device.type != 'cpu', '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model
    nc, names = model.nc, model.names  # number of classes, class names

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    assert nc == len(names), f'Model class count {nc} != len(names) {len(names)}'

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction

    # manipulate modules
    for k, m in model.named_modules():

        # SPPF optimizer for rknn export
        if rknn:    
            # rknn 1.6 not support SiLU operator
            if isinstance(m, Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.SiLU):
                    m.act = SiLU() 

            # optimized [5,9,13] 5x5 maxpool in SPPF() ==> two 3x3 size
            if isinstance(m, SPPF):
                m.m = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1) for i in range(2)])

        # Detect layer
        if isinstance(m, DetectX):
            m.inplace = inplace
            if rknn:    # rknn mode
                m.export_raw = True
            else:   # normal mode
                m.export = True


    # dry runs
    for _ in range(2):
        y = model(im)  

    # if half and not coreml:
    if half:
        im, model = im.half(), model.half()  # to FP16
    shape = tuple(y[0].shape)  # model output shape
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # Exports
    f = [''] * len(list(export_formats().Suffix))  # exported filenames
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    
    # if onnx or xml:  # OpenVINO requires ONNX
    if onnx:  # OpenVINO requires ONNX
        f[0] = export_onnx(model, im, file, opset, train, dynamic, simplify, save_dir)
    if rknn:
        f[1] = export_rknn(file=export_onnx(model, im, file, opset, train, dynamic, simplify, save_dir),    # onnx
                            calibration=cali, 
                            keep_onnx=True,
                            save_dir='',
                            prefix=colorstr('RKNN:')
                            )

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        LOGGER.info(f'\nExport complete ({time.time() - t:.2f}s)'
                    f"\nResults saved to {colorstr('bold', save_dir.resolve())}"
                    f"\nDetect:          python detect.py --weights {f} (not support RKNN for now)"
                    # f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')"
                    f"\nValidate:        python val.py --weights {f} (not support RKNN for now)"
                    f"\nVisualize:       https://netron.app")
    return f  # return list of exported files/dirs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default=ROOT / 'runs/export', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
    parser.add_argument('--include', nargs='+', default=[''], help='onnx, rknn')
    parser.add_argument('--cali', type=str, default='', help='do rknn qnt calibration')  

    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
