YOLOX-BETA based on YOLOv5 code. To be continued...


# TODO

	0. test yolov6 backbone speed.   & test OOM change!


	1. to train(anchor free, not anchor-based): 

	!!! 
	[ing @192.168.0.18] x-n-ciou-csp, no repConv, decouled head: c_ = min(c1 // 2, 256)    |  map: 27.3 (105epochs)
	[ing @192.168.0.35] x-n-rep.yaml, siou, repConv  decouled head: c_ = min(c1 // 2, 256) | map: 26.8, 86epchs
	[ing @192.168.0.26] x-n-rep.yaml, siou, repConv  decouled head: c_ = min(c1, 256) | map: 26.4, 74epochs
	[] x-n-rep-sa.yaml (sa will increase 0.8% map in yolov7-s-sa(37.2 -> 38))


	[] crossConv decoupled head experiments: c_ = min(c1 // 2, 256) ?  c_ = min(c1, 256) ?  up to above experiments


	2. structure

	[] decoupled head: 
		- 1. not fuse  V.S.  fuse(brach_b, brach_c)  
			=> params(1884799 -> 1690815) GFLOPs(4.5 -> 4.5) when c_ = min(c1 // 2, 256)
			=> params(3109407 -> 2334367) GFLOPs(7.9 -> 7.9) when c_ = min(c1, 256); 
		- 2. use Conv   V.S.   CrossConv
			=> params(1690815 -> 1626751) GFLOPs(4.5 -> 4.2) when c_ = min(c1 // 2, 256)
			=> params(2334367 -> 2077215) GFLOPs(7.9 -> 6.5) when c_ = min(c1, 256); 
		- TODO: AP test


	[] R-bottleneck: switch 1x1 conv and 3x3 conv
	[] csp + spp, csp + pan
	[] yolov7 head and backbone(E-ELAN)


	3. features
	[] compute loss code keep refine, cpu parts 
	[] Mac calculations
	[] hyps config
	[] export rknn
	[] end to end => NMS
	[] ATSS 
	[] Task-Align-Learning, TOOD
	[] cut off yolov5 part ????
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
	[x] YOLOR module in Detect() head
	[x] crossconv 


	
	
	



