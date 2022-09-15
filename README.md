# YOLOXY: Common 2D computer vision task with Anchor Free YOLO.


## TODO
- [x] Object Detection
- [x] Keypoint Detection
- [x] Multi Objects Tracking(ByteTrack)
- [ ] Instance Segmentation(SparseInst, YOLACT)
- [ ] Model Prune


## Updates
==> 2022.08.29: **Nano** size model achieve **32.6% mAP** (coco2017 val) with **2.69M params** and **5.7GFLOPs**\
==> 2022.09.08: **Nano** size model achieve **33.8% mAP** (coco2017 val) with **2.35M params** and **5.2GFLOPs**


## Pretained Models 
### Nano size
|Model |size|mAP<sup>val<br>0.5:0.95 |params<br><sup>(M) |FLOPs<br><sup>@640 (B) 
|---|---|---|---|---
|YOLOv5n(v6.2)      		|640 |28.0 |**1.9** |**4.5** 
|YOLOX-n      			|416 |25.8 |**0.91** |**1.08** 
|YOLOX-tiny      		|416 |32.8 |5.06 |6.45 
|**[YOLOXY-N(ours)](https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/N.pt)**      	|640 |**33.8**|2.35|5.2 

### Small size
|Model |size|mAP<sup>val<br>0.5:0.95 |params<br><sup>(M) |FLOPs<br><sup>@640 (B) 
|---|---|---|---|---
|YOLOv5s(v6.2) 				|640 |37.4 |**7.23** |**16.53**  
|YOLOX-s 				|640 |**40.5** |9.0 |26.8
|**YOLOXY-S-baseline(ours)** 		|640 |**39.3** |7.6  |17.9
|**[YOLOXY-S-RepConv-AsymConv(ours)](https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/s.pt)** |640 |**40.4**     | 8.67 |19.7	
|**YOLOXY-S(ours)**      		|640 |Training |7.37|18.3


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
python detect.py --weights N.pt	--source 0  # webcam
			   N-face-5.pt	 img.jpg  # image
					 vid.mp4  # video
					 ./data/images/  # image directory
					 'path/*.jpg'  # glob
					 'rtsp://admin:admin12345@192.168.0.188/h264'  # RTSP, RTMP, HTTP stream
```
<img src="https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/bus-N.jpg" height="500"> <img src="https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/FADID-FACE.bmp" height="400">\
<img src="https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/nba.png" height="600"> 

## Multi Objects Tracking
```bash
python detect.py --weights N.pt	--source rtsp://admin:admin12345@192.168.0.188/h264 --tracking
```


## Test mAP
```
python val.py --weights N.pt --data data/datasets/coco.yaml
```
```
Summary(N) 333 layers, 2357167 parameters, 5.2 GFLOPs, 33.8% mAP

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.338
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.515
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.356
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.159
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.368
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.471
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.297
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.491
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.535
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.324
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.586
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.702

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

