#pragma once


// keypoint with conf
typedef struct {
	cv::Point_<float> kpt;
	float conf;
} KEYPOINT;


// bbox with kpts
typedef struct {
	cv::Rect_<float> rect;  // xywh
	int id;			// class_id
	float score; // conf_obj * conf_cls 
	std::vector<KEYPOINT> kpts;	
} BBOX;	


// TRAJECTORY
typedef struct {
	int track_id;
	int state;
	std::vector<cv::Point> center;
} TRAJECTORY;







