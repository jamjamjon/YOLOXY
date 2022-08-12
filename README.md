# YOLOXY: Common 2D(XY) computer vision task with YOLO.
To be continued...


### Pretained model
|Model |size<br><sup>(pixels)|Epochs |mAP<sup>val<br>0.5:0.95 |P<br><sup>(300 epochs) |R<br><sup>(300 epochs)|params<br><sup>(M) |FLOPs<br><sup>@640 (B) | Speed<br><sup>GTX1080Ti b1(ms)
|---                    |---  |---    |---    |---    |---    |--- |--- |---
|yolov5n      			|640 |300 	|28.0   |57.4    |43.2  |1.9     |4.5 | 5.2
|nano-rep(baseline branch)      			|640 |300, 338  |**30.8**, **31.4**   | 59.8   |46    |3.05    |7.7 | 6.1
|nano-rep-half-head(baseline branch)     |640 |300, 316  |**30.2**, **30.4**   | 58.7   |44.4  |1.83    |4.4 | 6.0
|yolov5s-silu(v6.1) 	|640 | 300  |37.4 | | |**7.23** |**16.53** |**7.7** 
|s-crossconv-head-RepConv(v5x branch) 	|640 | 300| **39.2** | |    |7.9  |17.0	|9.6
|xs-Conv-ciou 								|640 |38.6(300) 39.3(370)    |7.6  |17.9	|


## First priority: get baseline mAP
|Model |size|mAP<sup>val<br>0.5:0.95 |params<br><sup>(M) |FLOPs<br><sup>@640 (B) | Speed<br><sup>GTX1080Ti b1(ms)
|---|---|---|---|---|---
|yolov5s-silu(v6.1) 						|640 |37.4 |**7.23** |**16.53** |**7.7** 
|yoloxs 									|640 |40.5 |**9.0** |**26.8** | - 
|xs-Conv-ciou 								|640 |38.6(300) 39.3(370)    |7.6  |17.9	|
|xs-Conv-siou 								|640 |ING(no need?)    | 7.6  |17.9	|
|xs-repConv-AsymConv-siou(3693) 			|640 |ing     | 7.7 |17.9	|
|xs-repConv-AsymConv-siou(612186) 			|640 |TODO    | 8.4 |20.1|
|xs-repConv-AsymConv-ese-siou 				|640 |TODO    | - |	- |- 



### onnx export
python export.py  --weights weights/best.pt --img 640 --simplify  --include onnx


### RKNN export
python export.py  --weights weights/best.pt --img 640 --simplify  --include rknn --cali data/images/


### RKNN CPP Deploy 
	-> ./deploy/RKNN


### BUG



### TODO List
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
	[x] many2one: 对没有分配anchor的gt进行分配cost最小的为分配anchor(SimOTA bug)
	
	[ing] AsymConv() used in stem part or some other parts, can not replace all Conv()! It will cause hard traning!
	[ing] DBBConv(), Inception_like_conv(), Xception_like_conv() 
	[ing] DBB => Diverse Branch Block: Building a Convolution as an Inception-like Unit
        

       [] AssertionError: >>> Matching matrix still has conflicts!!!!!!!

        
    	[] End2End => NMS Free

	[x] different branch has different branch head(TOOD, TAL)
	[] X_focal loss
	[] Tasked alignment assignment in compute_loss()
	[] assigner => compute_loss()
	[] metrics of kpts for saving model 
 
	[] OC_tracker with kpt
	[] pose-estimation(keypoints detection) -> head and loss
	[] ObjectBox
	[] segmentation

