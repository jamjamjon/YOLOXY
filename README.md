# YOLOXY: Common 2D computer vision task with YOLO.

- [x] Object Detection
- [x] Keypoint Detection
- [ ] Instance Segmentation 


## Pretained Models(To Be Continued...)
|Model |size|mAP<sup>val<br>0.5:0.95 |params<br><sup>(M) |FLOPs<br><sup>@640 (B) | Speed<br><sup>GTX1080Ti b1(ms)
|---|---|---|---|---|---
|YOLOv5n-SiLU(v6.1)      		|640 |28.0 |1.9 |4.5 | -
|**YOLOXY-Nano-half-head**    		|640 |**30.4**|**1.83**|**4.4** | -
|**YOLOXY-Nano**      			|640 |**31.4**|**3.05**|**7.7** | -
|**YOLOXY-n**      			|640 |**(31.5)still training...**|**2.69**|**5.7** |-
|YOLOv5s-SiLU(v6.1) 			|640 |37.4 |**7.23** |**16.53** |- 
|YOLOX-s 				|640 |**40.5** |**9.0** |**26.8** | - 
|**YOLOXY-s-Conv** 			|640 |39.3 |7.6  |17.9|-
|**[YOLOXY-s-RepConv-AsymConv](https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/s.pt)** |640 |**40.4**     | 8.67 |19.7	| -


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
python val.py --weights s.pt --data data/projects/coco.yaml
```
```
Summary(s) 315 layers, 8673663 parameters, 19.7 GFLOPs, 40.4% mAP

Average Precision (AP) @[ IoU=0.50:0.95   | area= all | maxDets = 100 ] = 0.404
Average Precision (AP) @[ IoU=0.50        | area= all | maxDets = 100 ] = 0.601
Average Precision (AP) @[ IoU=0.75        | area= all | maxDets = 100 ] = 0.434
Average Precision (AP) @[ IoU=0.50:0.95   | area=small| maxDets = 100 ] = 0.231
Average Precision (AP) @[ IoU=0.50:0.95   | area= medium| maxDets=100 ] = 0.450
Average Precision (AP) @[ IoU=0.50:0.95   | area= large | maxDets=100 ] = 0.533
Average Recall    (AR) @[ IoU=0.50:0.95   | area= all | maxDets =   1 ] = 0.330
Average Recall    (AR) @[ IoU=0.50:0.95   | area= all | maxDets =  10 ] = 0.542
Average Recall    (AR) @[ IoU=0.50:0.95   | area= all | maxDets = 100 ] = 0.586
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets = 100 ] = 0.415
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets = 100 ] = 0.640
Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets = 100 ] = 0.729
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

- [ ] more backbones experiments(PP-YOLOE, Transformer-based, ConvNext, ...)    
- [ ] DBB, DBBConv() => Diverse Branch Block: Building a Convolution as an Inception-like Unit
- [ ] X_focal loss => vari-focal loss in cls and obj; d-focal loss in box 
- [ ] End2End => NMS Free
- [ ] OC_tracker with kpt
- [ ] ObjectBox
- [ ] Instance segmentation
</details>
