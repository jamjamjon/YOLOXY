# YOLOXY: Common 2D computer vision task with YOLO.

- [x] Object Detection
- [x] Keypoint Detection
- [ ] Instance Segmentation 


## Pretained Models 
### Nano
|Model |size|mAP<sup>val<br>0.5:0.95 |params<br><sup>(M) |FLOPs<br><sup>@640 (B) 
|---|---|---|---|---
|YOLOv5n-SiLU(v6.1, 1:2:3:1)      		|640 |28.0 |1.9 |4.5 | -
|**YOLOXY-N-baseline(1:1:3:1)**      	|640 |**32.6**|**2.69**|**5.7** 

### Small
|Model |size|mAP<sup>val<br>0.5:0.95 |params<br><sup>(M) |FLOPs<br><sup>@640 (B) 
|---|---|---|---|---
|YOLOv5s-SiLU(v6.1, 1:2:3:1) 			|640 |37.4 |**7.23** |**16.53**  
|YOLOX-s 								|640 |**40.5** |**9.0** |**26.8** 
|YOLOXY-S-Conv 							|640 |39.3 |7.6  |17.9|-
|**[YOLOXY-S-RepConv-AsymConv](https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/s.pt)** |640 |**40.4**     | 8.67 |19.7	
|**YOLOXY-S-baseline(1:1:3:1)**      	|640 |TODO |-|-


## Installation
**Python>=3.7.0**, **PyTorch>=1.8.1**

```bash
git clone https://github.com/jamjamjon/YOLOXY.git  # clone
cd YOLOXY
pip install -r requirements.txt  # install
```

## Inference / Demo 
__Object Detection__ & __Keypoints Detection__
```bash
python detect.py --weights s.pt	--source 0  # webcam
				  img.jpg  # image
				  vid.mp4  # video
				  ./data/images/  # image directory
				  'path/*.jpg'  # glob
				  'rtsp://admin:admin12345@192.168.0.188/h264'  # RTSP, RTMP, HTTP stream
```
<img src="./data/docs/demo/face-5-demo.jpg" height="280"> <img src="./data/docs/demo/FADID-FACE-demo.bmp" height="280">

## Test mAP
```
python val.py --weights N.pt --data data/projects/coco.yaml
```
```
=> Summary(s) 296 layers, 2692111 parameters, 5.7 GFLOPs, 32.6% mAP

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.326
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.507
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.346
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.160
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.364
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.448
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.289
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.481
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.526
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.327
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.583
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.686

```

## Train For Object Detection

__Train From Stratch__
```bash
python train.py --data data/projects/coco.yaml --cfg models/cfg/s.yaml --batch-size -1
```
__Transfer Learning__
```bash
python train.py --data data/projects/coco.yaml --weights s.pt --batch-size -1
```

## Train For Keypoints Detection

Remember to modify **`nk`** and **`kpt_lr_flip_idx`** when training for keypoints detection in [kpts_dataset.yaml](./data/projects/FADID-FACE-19.yaml). 
```
...

# Classes
nc: 1  # number of classes
nk: 19   # number of keypoints (optional, 0 => bbox detection; > 0 => keypoints)
kpt_lr_flip_idx: [3, 2, 1, 0, 10, 9, 8, 11, 6, 5, 4, 7, 12, 14, 13, 17, 16, 15, 18]   # left-right flip for kpts, required when using ==> hyp['fliplr']

```

__Train From Stratch__
```bash
python train.py --data data/projects/FADID-FACE-19.yaml --cfg models/cfg/s.yaml --batch-size -1
```
__Transfer Learning__
```bash
python train.py --data data/projects/FADID-FACE-19.yaml --weights s.pt --batch-size -1
```


## Export To Deploy
__General ONNX__
```bash
python export.py  --weights s.pt --img 640 --simplify  --include onnx
```
__ONNX For RKNN__
```bash
python export.py  --weights s.pt --img 640 --simplify  --include rknn --cali data/images/  # image dirdirectory
								    	     calibration.txt 	# text file of images path 
```

## Acknowledgements
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)


## TODO List

<details><summary> <b>Expand</b> </summary>

- [x] sa block -> increse 0.8% map in xs model =====> to test(speed)
- [x] siou
- [x] close mosaic in the last 5% epochs
- [x] hyps config
- [x] byte_tracker 
- [x] remove yolov5 parts
- [x] rknn export parts
- [x] rknn QNT calibration file: support dir(recommend), not only cali.txt
- [x] RKNN C++ deploy code ref
- [x] pose-estimation(keypoints detection) -> dataloader and model
- [x] pose-estimation(keypoints detection) -> head and loss
- [x] SimOTA bug fix(many2one): re-assign anchors for GTs whose anchors assigned before just have been removed
- [x] AsymConv() used in stem part or some other parts, can not replace all Conv()! It will cause hard traning!
- [x] Tasked alignment assignment in compute_loss()	
- [x] add KPTs cost to cost matrix in order to align all task
- [x] different branch has different branch head(TOOD, TAL)
- [x] DBB, DBBConv() => Diverse Branch Block: Building a Convolution as an Inception-like Unit
- [x] vari-focal loss in cls and obj; 
- [ ] nano model: baseline, yolov5 backbone;
- [ ] Reparametizing backbone, remove some activations.(Testing on yolov5 now!) 

- [ ] torchscript, tensorRT, coreML support   
- [ ] more backbones experiments(PP-YOLOE, Transformer-based, ConvNext, ...)    
- [ ] OC_tracker with kpt
- [ ] Instance segmentation

- [ ] End2End => NMS Free
- [ ] ObjectBox

</details>
