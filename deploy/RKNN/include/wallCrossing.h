#ifndef WALLCROSSING_H
#define WALLCROSSING_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <iostream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"



// calculate two vector_2d
int64_t cross_2d(cv::Vec<int, 2>& a, cv::Vec<int, 2>& b);

// check if two segments intersect
bool is_segments_intersected(cv::Point& ax, cv::Point& ay, cv::Point& bx, cv::Point& by);

// check if two segments intersect
bool is_bbox_intersected_with_segment(cv::Mat& frame, cv::Rect& bbox, cv::Point& A, cv::Point& B, float scale_w=0.4, float scale_h=0.3, int min_bbox_w=15, int min_bbox_h=15);


#endif

