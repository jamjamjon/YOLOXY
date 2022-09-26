# YOLOXY: Common 2D computer vision task with Anchor Free YOLO.


## Roadmap
- [x] Object Detection
- [x] Keypoint Detection
- [x] Multi Objects Tracking(ByteTrack)
- [ ] Instance Segmentation(SparseInst, YOLACT)
- [ ] Model Prune


## Updates
==> 2022.08.29: **Nano** size model achieve **32.6% mAP** (coco2017 val) with **2.69M params** and **5.7GFLOPs**\
==> 2022.09.08: **Nano** size model achieve **33.8% mAP** (coco2017 val) with **2.35M params** and **5.2GFLOPs**\
==> 2022.09.17: **Small** size model achieve **42.0% mAP** (coco2017 val) with **7.37M params** and **18.3GFLOPs**

## Pretained Models 
### Nano size
|Model |size|mAP<sup>val<br>0.5:0.95 |params<br><sup>(M) |FLOPs<br><sup>(G) 
|---|---|---|---|---
|YOLOv5n(v6.2)      		|640 |28.0 |**1.9** |**4.5** 
|YOLOX-n      			|416 |25.8 |**0.91** |**1.08** 
|YOLOX-tiny      		|416 |32.8 |5.06 |6.45 
|YOLOv6-N     			|640 |36.3 |4.3 |**11.1** 
|RTMDet-tiny     		|640 |**40.9** |4.8 |**8.1** 
|**[YOLOXY-N(ours)](https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/N.pt)**      	|640 |**33.8**|2.35|5.2 

### Small size
|Model |size|mAP<sup>val<br>0.5:0.95 |params<br><sup>(M) |FLOPs<br><sup>(G) 
|---|---|---|---|---
|YOLOv5s(v6.2) 				|640 |37.4 |7.23 |16.53  
|YOLOX-s 				|640 |40.5 |9.0  |26.8
|PP-YOLOE+_s     			|640 |43.7 |7.93 |17.36
|YOLOv6-S     				|640 |43.8 |17.2 |44.2
|YOLOv7-tiny-SiLU     			|640 |38.7 |**6.2** |**13.8**
|RTMDet-s     				|640 |**44.5** |8.89 |**14.8** 
|**[YOLOXY-S(ours)](https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/S.pt)**      		|640 |**42.0**|7.37|18.3



## Installation
**Python>=3.7.0**, **PyTorch>=1.8.1**

```bash
git clone https://github.com/jamjamjon/YOLOXY.git  # clone
cd YOLOXY
pip install -r requirements.txt  # install
```

## Inference / Demo 
```bash
python tools/detect.py --weights N.pt	--source 0  # webcam
			 N-face-5.pt	  img.jpg  # image
					  video.mp4  # video
					  ./data/images/  # image directory
					  'rtsp://admin:admin12345@192.168.0.188/h264'  # RTSP
```

Object Detection | Human Pose Estimation | Object Detection with Keypoints
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/bus-N.jpg" width="300"> |<img src="https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/kunball.png" width="400"> | <img src="https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/FADID-FACE.bmp" width="300"> |



## Multi Objects Tracking
```bash
python tools/detect.py --weights N.pt	--source rtsp://admin:admin12345@192.168.0.188/h264 --tracking
```

## Test mAP

```
python tools/val.py --weights N.pt --data data/datasets/coco.yaml
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

__Directories Structure__
```
parent
├── YOLOXY
|    └── data
|   	  └── datasets
|	  	└──your-dataset.yaml  ← step 1. prepare custom dataset.yaml 
└── datasets
|    └── your-datasets  ←  step 2. custom dataset (images and labels)
|   	  └── images
|   	  └── labels
```
	
<details close>
<summary>Train Custom Dataset</summary>	
	
**same as YOLOv5, [check this](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)**


__Train From Stratch__
```bash
python tools/train.py --data data/projects/your-custom-dataset.yaml --cfg models/cfg/N.yaml --batch-size -1
```
__Transfer Learning__
```bash
python tools/train.py --data data/projects/your-custom-dataset.yaml --weights N.pt --batch-size -1
```
</details>
	
	
	
## Train For Keypoints Detection
	
<details close>
<summary>Train Custom Dataset</summary>	

	
**Make sure your keypoints label has following format**

	
```bash
# class_id x_center y_center width height kpt1_x kpt1_y kpt2_x kpt2_y ... kptn_x kptn_y (normalized, 0-1)
0  0.03369140625 0.4786450662739323 0.0205078125 0.03829160530191458 0.0317119140625 0.4736480117820324 0.04082421875 0.4736480117820324 0.03767578125 0.48089396170839466 0.033203125 0.48838880706921944 0.0403271484375 0.48813843888070696   	# one object
1  0.09765625 0.47201767304860087 0.029296875 0.05154639175257732 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0	# another object
```

	
__step 1. Prepare Your Dataset.yaml__
```
parent
├── YOLOXY
|    └── data
|   	  └── datasets
|	  	└──your-dataset.yaml  ← put here

```
	
**Modify **`nk`** and **`kpt_lr_flip_idx`** in your-dataset.yaml**

```bash
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco-kpts  # dataset root dir
train: images/train2017  # train images 
val: images/val2017 # val images 

# Classes & Keypoints
nc: 1  # number of classes
nk: 17   # number of keypoints (optional, 0 => bbox detection; > 0 => keypoints)
kpt_lr_flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # left-right flip for kpts, required when using ==> hyp['fliplr']

names: ['person']
```

	
__step 2. Prepare Your Dataset__	
```
parent
├── YOLOXY
|    └── data
|   	  └── datasets
|	  	└──your-dataset.yaml  
└── datasets
|    └── your-datasets  ←  pute here
|   	  └── images
|   	  └── labels
```
	
</details>	


__Train From Stratch__
```bash
python tools/train.py --data data/projects/FADID-FACE-19.yaml --cfg models/cfg/s.yaml --batch-size -1
```
__Transfer Learning__
```bash
python tools/train.py --data data/projects/FADID-FACE-19.yaml --weights s.pt --batch-size -1
```
	


## Export To Deploy
__General ONNX__
```bash
python tools/export.py  --weights s.pt --img 640 --simplify  --include onnx
```
__ONNX For RKNN__
```bash
python tools/export.py  --weights s.pt --img 640 --simplify  --include rknn --cali data/images/  # image dirdirectory
								    	     calibration.txt 	# text file of images path 
```

## References
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

