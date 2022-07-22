#pragma once
#include <sys/time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "rknn_api.h"
#include <chrono>
#include <array>
#include <vector>
#include <stdint.h>
#include <array>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>				
#include <string>
#include "baseStruct.h"
#include "BYTETracker.h"



// different task
// enum TASK { GeneralDetection = 0, FaceWithLandMark, HumanPose2D};

// rknn class
class RKNN {
public:	
	RKNN() { }
	RKNN(std::string f, uint32_t rknn_init_flag) { 
		this->Init(f, rknn_init_flag);
	}
	~RKNN() {	
		// this->Release();
	}
	
	// load rknn & rknn init
	rknn_context ctx = 0;				// rknn ctx
	rknn_input_output_num io_num;		// get rknn model num_input. num_output
	
	// rknn inputs & outputs
	// rknn_input* rknn_inputs;			// rknn_input inputs[io_num.n_input]; 
	// rknn_output* rknn_outputs;		// rknn_output outputs[io_num.n_output]; 
	rknn_input rknn_inputs[1];
	rknn_output rknn_outputs[4]; 

	// rknn model key params
	int nc;   			// number of classes, compute automaticlly!
	int height;			// model input height size, compute automaticlly!    @jamjamjon (448, 800) better and faster than (640, 640); 
	int width;			// model input width size, compute automaticlly! 
	int ch;   			// number of channels, compute automaticlly! 

	// functions
	void Init(std::string f, uint32_t rknn_init_flag);  // load model and get model params
	// virtual void IOConfig();	
	void Infer();	
	virtual void Release();		// release rknn ctx & free mem
};



// yolov5 model.
class YOLO: public RKNN {

public:

	YOLO(std::string f, uint32_t rknn_init_flag) {
		RKNN::Init(f, rknn_init_flag);		// rknn init
	}
	~YOLO() {
		this->Release();
	};


	std::vector<BBOX> detections;  	// bboxes for saving detection results
	
	/* 	About Tracking   */
	std::vector<STrack> stracks;  		// tracked tracker
	std::vector<STrack> stracks_all;  	// lost_tracks + tracked tracks
	std::unordered_map<int, std::pair<std::vector<cv::Point>, bool>> trajectories;  // trajectory  ==>  {tracker_id: <[(cx, cy), ...], state>}
	// std::vector<TRAJECTORY> trajectory;

	void Detect(cv::Mat& frame, float conf_thresh, float nms_thresh);  // Detect
	void Track(BYTETracker& tracker, cv::Mat& frame, bool enable_trajectory=true);	// Track
	void Draw(cv::Mat& img);	// draw results
	void Release();		// release rknn model
	

	
private:
	// vars for input pre-process
	std::array<int, 4> padding = {0, 0, 0, 0};  //  top, left, bottom, right
	float scale4lb = 1.0f;						// scale for letterbox
	std::vector<std::array<int, 6>> anchors;	// anchors

	void _PreProcess(cv::Mat& image_in, cv::Mat& image_out);		// pre-process
	void _Decode(float* input, int stride, float threshold, std::vector<BBOX>& bboxes);	// de-qnt & get bboxes from rknn output
	void _NonMaxSuppression(float threshold, std::vector<BBOX>& bboxes);	// NMS
	void _PostProcess(float conf_thresh, float nms_thresh); // post-process 
	void _xyScaleCoords(float& x, float& y) {		// de-leterbox & de-scaled
		x -= this->padding[1];
		y -= this->padding[0];
		x = std::min(std::max(0.f, x), (float)this->width) / this->scale4lb;		// or: clamp(x1 / c_scale4lb, 0, image_width)
		y = std::min(std::max(0.f, y), (float)this->height) / this->scale4lb;	
		
	}
	void _whScaleCoords(float& w, float& h) {		// de-scaled
		w = std::min(std::max(0.f, w), (float)this->width) / this->scale4lb;		// or: clamp(x1 / c_scale4lb, 0, image_width)
		h = std::min(std::max(0.f, h), (float)this->height) / this->scale4lb;	
	}

	float _bboxes_iou(BBOX& a, BBOX& b) {
		// xywh
		float w = std::max(0.f, std::min(a.rect.x + a.rect.width, b.rect.x  + b.rect.width) - std::max(a.rect.x, b.rect.x) + 1.0f);
		float h = std::max(0.f, std::min(a.rect.y + a.rect.height, b.rect.y + b.rect.height) - std::max(a.rect.y, b.rect.y) + 1.0f);
		float intersection = w * h;	// intersection
		float union_ = (a.rect.width + 1.0f) * (a.rect.height + 1.0f) + (b.rect.width + 1.0f) * (b.rect.height + 1.0f) - intersection;	// union
		return union_ <= 0.f ? 0.f : (intersection / union_);		
	}
	cv::Scalar _get_color(int idx);		// get color 
	float _sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }	// sigmoid 

};




