YOLOXT based on YOLOv5 code.
To be continued.


# TODO

	1. to train(anchor free, not anchor-based): 

	!!! 
	[ing @192.168.0.18] xt-n-ciou-csp, no repConv, decouled head: c_ = min(c1 // 2, 256)    |  map: 27.3 (105epochs)
	[ing @192.168.0.35] xt-n-rep.yaml, siou, repConv  decouled head: c_ = min(c1 // 2, 256) | map: 26.8, 86epchs
	[ing @192.168.0.26] xt-n-rep.yaml, siou, repConv  decouled head: c_ = min(c1, 256) | map: 26.4, 74epochs


	[] xt-n-rep-sa.yaml (sa will increase 0.8% map in yolov7-s-sa(37.2 -> 38))

	

	2. structure
	[x] crossconv 
	[] bottleneck switch 1x1 conv and 3x3 conv
	[] csp + spp, csp + pan , csp + head
	[x] YOLOR module in Detect() head


	3. features
	[] compute loss code keep refine, cpu parts 
	[] Mac calculations
	[] export rknn
	[] end to end  => NMS
	[] ATSS 
	[] Task-Align-Learning, TOOD
	[] yolov7 head and backbone
	[] cut off yolov5 part

	[] byte-track trackers
	[] pose-estimation
	[] segmentation


	4. bugs
	



# Done
	[x] support yolov5 and yolox both
	[x] s-iou
	[x] val. tag, not include in weights(.pt)
	[x] remove hyp evolve part
	[x] cfg and weights
	[x] REPVGG-block  => to test(map & speed)
	[x] sa block -> increse 0.8% map =====> to test(speed)
	
	
	



