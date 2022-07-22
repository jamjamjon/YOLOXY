# YOLOX style CPP code => Detection + Tracking

### Usage
1. 安装RKNN/data下的eigen.zip,
2. 编译： 
	mkdir build
	cd build
	cmake ..
	make 
3. 运行: 
	<rknn model> <image/video> conf_thresh nms_thresh[optional]
	./main ../data/weights/test.rknn ../data/images/bus.jpg .4 .4


## 注意
data/weights下的.rknn模型是测试模型，对应yolov5的s模型，该模型还没完全运行完毕，仅仅作测试使用，后续会更新该rknn模型

### 后续完善readme...