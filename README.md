# YOLOXY: Common 2D(XY) computer vision task with YOLO.
To be continued...


## Pretained model
|Model |size|mAP<sup>val<br>0.5:0.95 |params<br><sup>(M) |FLOPs<br><sup>@640 (B) | Speed<br><sup>GTX1080Ti b1(ms)
|---|---|---|---|---|---
|yolov5n-silu(v6.1)      		|640 |28.0 |1.9 |4.5 | **5.2**
|**nano-rep-half (ours)**    		|640 |**30.2(300)**, **30.4(316)**   |**1.83**    |**4.4** | 6.0
|**nano-rep (ours)**      		|640 |**30.8(300)**, **31.4(338)**   |**3.05**    |**7.7** | 6.1
|yolov5s-silu(v6.1) 			|640 |37.4 |**7.23** |**16.53** |**7.7** 
|yolox-s 				|640 |**40.5** |**9.0** |**26.8** | - 
|**s-Conv-ciou (ours)** 		|640 |38.6(300) 39.3(370)    |7.6  |17.9|
|**s-RepConv-AsymConv (ours)**		|640 |**40.4**     | 8.6 |19.7	|




## onnx export
	python export.py  --weights weights/best.pt --img 640 --simplify  --include onnx


#### RKNN export
	python export.py  --weights weights/best.pt --img 640 --simplify  --include rknn --cali data/images/



## BUGs



## TODO List
	[x] sa block -> increse 0.8% map in xs model =====> to test(speed)
	[x] siou
	[x] close mosaic in the last 5% epochs
	[x] hyps config
	[x] byte_tracker 
	[x] remove yolov5 parts
	[x] rknn export parts
	[x] rknn QNT calibration file: support dir(recommend), not only cali.txt
	[x] RKNN C++ deploy code ref
	[x] pose-estimation(keypoints detection) -> dataloader and model
	[x] pose-estimation(keypoints detection) -> head and loss
	[x] many2one: 对没有分配anchor的gt进行分配cost最小的为分配anchor(SimOTA bug)
	
	[] backbone experiments
	[] yoloe backbone
	[-] AsymConv() used in stem part or some other parts, can not replace all Conv()! It will cause hard traning!
	[-] DBBConv(), Inception_like_conv(), Xception_like_conv() 
	[-] DBB => Diverse Branch Block: Building a Convolution as an Inception-like Unit
            

	[] X_focal loss => vari-focal loss in cls and obj; d-focal loss in box 
    [] End2End => NMS Free

	
	[x] Tasked alignment assignment in compute_loss()	
	[x] add KPTs cost to cost matrix in order to align all task
	[x] different branch has different branch head(TOOD, TAL)

	[] metrics of kpts for saving model 
 
	[] OC_tracker with kpt
	[] ObjectBox
	[] segmentation

