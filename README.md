# YOLOX-BETA: support YOLOv5 & YOLOX both
To be continued...


### Baseline pretained model (check baseline branch)
|Model |size<br><sup>(pixels)|Epochs |mAP<sup>val<br>0.5:0.95 |P<br><sup>(300 epochs) |R<br><sup>(300 epochs)|params<br><sup>(M) |FLOPs<br><sup>@640 (B) | Speed<br><sup>GTX1080Ti b1(ms)
|---                    |---  |---    |---    |---    |---    |--- |--- |---
|nano-rep      		|640 |300, 338  |**30.8**, **31.4**   | 59.8   |46    |3.05    |7.7 | 6.1
|nano-rep-half-head     |640 |300, 316  |**30.2**, **30.4**   | 58.7   |44.4  |1.83    |4.4 | 6.0
|yolov5n      		|640 |300 	|28.0   	      |57.4    |43.2  |1.9     |4.5 | 5.2

**It seems like half-size-decoupled head won't hurt model bad.**


## v1.0 Ablation study
|Model |size|mAP<sup>val<br>0.5:0.95 |params<br><sup>(M) |FLOPs<br><sup>@640 (B) | Speed<br><sup>GTX1080Ti b1(ms)
|---|---|---|---|---|---
|yolov5s-silu(v6.0) 	|640 |37.4 |**7.23** |**16.53** |**7.7** 
|yolov5s-relu(v6.0) 	|640 | -   |**7.23** |**16.53** |**6.7**
|x-s-half-head-silu 	|640 |-    |7.8  |17.6 |8.7
|x-s-half-head-relu 	|640 |-    |7.8  |17.6 |8.0
|x-s-silu 				|640 |-    |9.0  |26.4 |9.8
|x-s-relu 				|640 |-    |9.0  |26.4 |9.2
|x-s-silu-v6-style 		|640 |-    |9.0  |26.4 |11.0
|x-s-relu-v6-style 		|640 |-    |9.7  |28.6 |9.9
|x-s-cross-not-half-head 	|640 |ing    |7.9  |17.0	|9.6 
|x-s-cross-half-head 		|640 |todo    |7.4  |14.8	|8.9 


**yolov6 style: in one word, based on yolov5n, then doubled num of bottleneck block in backbone, they compare this model which has much bigger Params and GFLOPS to yolov5n, then comes the higher mAP results. As for inference speed, replacing all SiLU() with ReLU(). That's funny.**



### Exp1: Conv, AsymConv, RepConv
|Model |size|mAP<sup>val<br>0.5:0.95 |params<br><sup>(M) |FLOPs<br><sup>@640 (B) | Speed<br><sup>GTX1080Ti b1(ms)
|---|---|---|---|---|---
|yolov5s-silu(v6.0) 				|640 |37.4 |**7.23** |**16.53** |**7.7**  
|x-s-cross-not-half-head-RepConv 	|640 |ing    |7.9  |17.0	|9.6
|x-s-cross-not-half-head-AsymConv 	|640 |TODO(hard train)    |7.9  |17.0	|9.6
|x-s-cross-not-half-head-Conv 		|640 |ing    |7.9  |17.0	|9.6
|Combinations ... 


### Exp2: Conv in head: (Conv, AsymConv, RepConv)
|Model |size|mAP<sup>val<br>0.5:0.95 |params<br><sup>(M) |FLOPs<br><sup>@640 (B) | Speed<br><sup>GTX1080Ti b1(ms)
|---|---|---|---|---|---
...

### Exp3: Conv in other place: (Conv, AsymConv, RepConv)
|Model |size|mAP<sup>val<br>0.5:0.95 |params<br><sup>(M) |FLOPs<br><sup>@640 (B) | Speed<br><sup>GTX1080Ti b1(ms)
|---|---|---|---|---|---
...

### TODO List
	
	[x] using SiLU() anywhere
	[x] C3xESE block: C3x + ese attention
	[x] fused decoupled head: half head ??? wait to see experiments 
	[x] sa block -> increse 0.8% map in xs model =====> to test(speed)
	[x] siou
	[x] close mosaic in the last 5% epochs
	[x] hyps config

	[ing] RepConv() , Conv(), AsymConv()
	
	[] Mac calculations in model_info()
	[] export rknn
	
	[] ObjectBox
	[] ATSS 
	[] Task-Align-Learning, TOOD
	[] end to end => NMS
	
	[] OOB
	[] tracking
	[] pose-estimation
	[] segmentation

