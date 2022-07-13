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
	[] cancel mosiac in last 20 epochs for small model.
	[] compute loss code keep refine
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
	[x] cfg and weights
	[x] REPVGG-block  => to test(map & speed)
	[x] sa block -> increse 0.8% map =====> to test(speed)
	[x] YOLOR module in Detect() head
	[x] crossconv 

# Test Corsely 

### num of c3x / bottleneck (yolov6)

	- 1. cfg = x-n-rep.yaml | backbone c3x num = [3,6,9,3]  
	fused:  249 layers, 1974847 parameters, 1397727 gradients, 4.5 GFLOPs

      Params      GFLOPs  GPU_mem (GB)  forward (ms) backward (ms)                   input                  output
     2045695        4.67         0.426         9.969         14.84        (1, 3, 640, 640)                    list

	- 2. cfg = x-n-rep.yaml | backbone c3x num = [6,12,18,6]  
	
	fused : 298 layers, 2161503 parameters, 1584383 gradients, 5.0 GFLOPs
	Params      GFLOPs  GPU_mem (GB)  forward (ms) backward (ms)                   input                  output
     2233151        5.23         0.487         11.43         16.96        (1, 3, 640, 640)                    list


### yolov7 backbone speed

	- yolov7-tiny-silu.yaml
  	Params      GFLOPs  GPU_mem (GB)  forward (ms) backward (ms)                   input                  output
 	7759199       17.74         0.774         10.01         23.02        (1, 3, 640, 640)                    list

 	- yolov7.yaml
      Params      GFLOPs  GPU_mem (GB)  forward (ms) backward (ms)                   input                  output
    40301087       118.7         2.982         38.18          89.1        (1, 3, 640, 640)                    list



