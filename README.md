# YOLOXY: Common 2D computer vision task with Anchor Free YOLO.


## Roadmap
- [x] Object Detection
- [x] Keypoint Detection
- [x] Multi Objects Tracking(ByteTrack)
- [ ] Model Prune


Object Detection | Object Detection With Keypoints | 
:-------------------------:| :-------------------------:|
<img src="https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/bus-N.jpg" width="440"> | <img src="https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/ikun.jpeg" width="560"> |


<details close>
<summary>Other Demo</summary>	

Face Detection With Keypoints  | Face Detection With Keypoints |
:-------------------------:|:-------------------------:
<img src="https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/FADID-FACE.bmp" width="500"> | <img src="https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/FACE.jpg" width="500"> |

</details>


<details open>
<summary>Benchmark</summary>	

**Nano Size**
|Model |size|mAP<sup>val<br>0.5:0.95 |Params(M) |FLOPs(G) | Speed(ms)<br>b32 fp32<br>RTX2080Ti
|---|---|---|---|---|---
|YOLOv5n(v6.2)      		|640 |28.0 |1.9 |4.5 |**0.8**
|YOLOv6-N     			|640 |36.3 |4.3 |11.1| 1.38
|YOLOX-Nano      		|416 |25.8 |**0.91** |**1.08** | -
|**[YOLOXY-N](https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/yoloxy-n.pt)**      	|640|**33.9**|**1.97**|**4.3**|**0.8**

**Small Size** 
|Model |size|mAP<sup>val<br>0.5:0.95 |Params(M) |FLOPs(G) | Speed(ms)<br>b32 fp32<br>RTX2080Ti
|---|---|---|---|---|---
|YOLOv5s(v6.2) 				|640 |37.4 |7.23 |16.53 |**1.4**
|YOLOv6-S     				|640 |43.8 |17.2 |44.2| 3.49
|YOLOX-s 				|640 |40.5 |9.0  |26.8| -
|PP-YOLOE+_s     			|640 |43.7 |7.93 |17.36| -
|RTMDet-s     				|640 |**44.5** |8.89 |14.8 | -
|**[YOLOXY-S](https://github.com/jamjamjon/YOLOXY/releases/download/v1.0/yoloxy-s.pt)**     |640 |**42.0**|7.71|16.7|**1.4**

<!-- 
### Tiny Size 
|Model |size|mAP<sup>val<br>0.5:0.95 |Params(M) |FLOPs(G) 
|---|---|---|---|---
|YOLOv6-T     			|640 |41.1 |15.0 |36.7
|YOLOv7-tiny-SiLU     		|640 |38.7 |6.2 |13.8
|YOLOX-Tiny      		|416 |32.8 |5.06 |6.45 
|RTMDet-Tiny     		|640 |40.9 |4.8 |8.1 
|**YOLOXY-Tiny**      	|640 |training|4.3|9.5
 -->

</details>



## Quick Start


<details close>
<summary>Installation</summary>	

**Python>=3.7.0**, **PyTorch>=1.8.1**

```bash
git clone https://github.com/jamjamjon/YOLOXY.git  # clone
cd YOLOXY
pip install -r requirements.txt  # install
```

</details>


<details close>
<summary>Inference Detection</summary>	

```bash
python tools/detect.py --weights yoloxy-n.pt --conf 0.5 --source 0  		# webcam
								 img.jpg  	# image
								 video.mp4  	# video
								 ./data/images/  # image directory
								 rtsp://admin:admin12345@192.168.0.177/h264  # RTSP stream
```

</details>

<details close>
<summary>Inference Detection With Keypoints</summary>	

**`--is-coco`** means to draw skeleton for COCO human keypoints(17). It is optional.

```bash
python tools/detect.py --weights yoloxy-s-coco-kpts.pt --conf 0.4 --kpts-conf 0.5 --is-coco --source 0  		# webcam
											   	     img.jpg  	# image
											             video.mp4  	# video
											  	     ./data/images/  # image directory
											  	     rtsp://admin:admin12345@192.168.0.177/h264  # RTSP stream
```

</details>



<details close>
<summary>Inference With Multi Objects Tracking(MOT)</summary>

```bash
python tools/detect.py --weights yoloxy-n.pt --tracking --source rtsp://admin:admin12345@192.168.0.188/h264 
```

</details>


<details close>
<summary>Test mAP</summary>

```
python tools/val.py --weights yoloxy-n.pt --data data/datasets/coco.yaml --img 640 --conf 0.001 --iou 0.65
```

</details>


<details close>
<summary>Test Speed</summary>

```
python tools/val.py --weights yoloxy-n.pt --task speed --data data/datasets/coco.yaml --img 640 --batch 32
```

</details>


<details close>
<summary>Export ONNX</summary>

```bash
python tools/export.py  --weights yoloxy-s.pt --img 640 --simplify  --include onnx
```


</details>


<details close>
<summary>Training</summary>


__Directories Structure__
```
parent
├── YOLOXY
     └── data
    	  └── datasets
	  	└──your-dataset.yaml  ← step 1. prepare custom dataset.yaml 
└── datasets
     └── your-datasets  ←  step 2. custom dataset (images and labels)
   	  └── images
   	  └── labels
```
		
**same as YOLOv5, [check this](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)**


__Train From Stratch__
```bash
# Single GPU
python tools/train.py --data data/datasets/your-custom-dataset.yaml --cfg models/cfg/yoloxy-n.yaml --batch-size 32

# Multi-GPU
python -m torch.distributed.run -nproc_per_node 4 tools/train.py --data data/datasets/your-custom-dataset.yaml --cfg models/cfg/yoloxy-n.yaml --batch 128 --device 0,1,2,3
```
__Transfer Learning__
```bash
# Single GPU
python tools/train.py --data data/datasets/your-custom-dataset.yaml --weights yoloxy-n.pt --batch-size 32

# Multi-GPU
python -m torch.distributed.run -nproc_per_node 4 tools/train.py --data data/datasets/your-custom-dataset.yaml --weights yoloxy-n.pt --batch 128 --device 0,1,2,3
```
</details>
	
	
<details close>
<summary>Training With Keypoints</summary>

**Keypoints Label Format 1: **`kpt_x, kpt_y, kpt_visibility(conf)`****

```bash
# class_id x_center y_center width height kpt1_x kpt1_y kpt1_visibility kpt2_x kpt2_y kpt2_visibility ... kptn_x kptn_y kptn_visibility (normalized, 0-1)
0  0.03369140625 0.4786450662739323 0.0205078125 0.03829160530191458 0.0 0.0 0.0 0.04082421875 0.4736480117820324 1.0 0.03767578125 0.48089396170839466 2.0 0.033203125 0.48838880706921944 2.0 0.0403271484375 0.48813843888070696 2.0 	# one object
```

****Keypoints Label Format 2: `kpt_x, kpt_y`****

```bash
# class_id x_center y_center width height kpt1_x kpt1_y kpt2_x kpt2_y ... kptn_x kptn_y (normalized, 0-1)
0  0.03369140625 0.4786450662739323 0.0205078125 0.03829160530191458 0.0317119140625 0.4736480117820324 0.04082421875 0.4736480117820324 0.03767578125 0.48089396170839466 0.033203125 0.48838880706921944 0.0403271484375 0.48813843888070696   	# one object
1  0.09765625 0.47201767304860087 0.029296875 0.05154639175257732 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0	# another object
```

	
**step 1. Prepare Dataset.yaml**
```
parent
├── YOLOXY
     └── data
   	  └── datasets
	  	└──your-dataset.yaml  ← put here

```
	
**And modify configs**

```bash
# datasets path
path: ../datasets/coco-kpts  # dataset root dir
train: images/train2017  # train images 
val: images/val2017 # val images 

# Classes & Keypoints
nc: 1  # number of classes
names: ['person']
nk: 17   # number of keypoints
kpt_lr_flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # left-right flip for kpts, required when using ==> hyp['fliplr']
```
	
**step 2. Prepare Dataset**
```
parent
├── YOLOXY
     └── data
    	  └── datasets
	  	└──your-dataset.yaml  
└── datasets
      └── your-datasets  ←  put here
   	  └── images
   	  └── labels
```	

**step 3. Training**

```bash
# Single GPU
python tools/train.py --data data/datasets/your-custom-dataset.yaml --weights yoloxy-s-coco-kpts.pt --batch 32

# Multi-GPU
python -m torch.distributed.run -nproc_per_node 4 tools/train.py --data data/datasets/your-custom-dataset.yaml --weights yoloxy-s-coco-kpts.pt --batch 128 --device 0,1,2,3
```

</details>



## References
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

