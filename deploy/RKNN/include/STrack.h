#pragma once

#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"

using namespace cv;
using namespace std;

enum TrackState { New = 0, Tracked, Lost, Removed };

// new 
enum Zone { X = 0, A, B};

class STrack
{
public:

	STrack(vector<float> tlwh_, float score);
	STrack(vector<float> tlwh_, float score, int class_id);
	~STrack();

	vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
	void static multi_predict(vector<STrack*> &stracks, byte_kalman::KalmanFilter &kalman_filter);
	void static_tlwh();
	void static_tlbr();
	vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
	vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();
	
	void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
	void re_activate(STrack &new_track, int frame_id, bool new_id = false);
	void update(STrack &new_track, int frame_id);

public:
	bool is_activated;
	int track_id;
	int state;
	int class_id;	// new add
	float score;
	
	vector<float> _tlwh;
	vector<float> tlwh;
	vector<float> tlbr;
	int frame_id;
	int tracklet_len;
	int start_frame;

	KAL_MEAN mean;
	KAL_COVA covariance;
	

private:
	byte_kalman::KalmanFilter kalman_filter;
};

// new add: out side, using list to save in & out 
	// int zone_cur = Zone::X;
	// int zone_last = Zone::X;
	// bool zone_changed = false;
	
	/*	
	// TODO: {num_people, num_people_crossed_line}
	// line
	cv::Point p1(0, frame.rows / 2);	
	cv::Point p2(frame.cols, frame.rows / 2);
	cv::line(frame, p1, p2, (255, 255, 0), 4);
	
	
	// 2 zone
	cv::Rect_<float> rect_A;
	rect_A.x = 0;
	rect_A.y = 0;
	rect_A.width = frame.cols;
	rect_A.height = frame.rows / 2;		
	
	cv::Rect_<float> rect_B;
	rect_B.x = 0;
	rect_B.y = frame.rows / 2;
	rect_B.width = frame.cols;
	rect_B.height = frame.rows / 2;		
	*/
