#include "yolo.h"



/* --------------------
 	RKNN parts
-----------------------*/ 

// init
void RKNN::Init(std::string f, uint32_t rknn_init_flag) {

	// load rknn model
	FILE* fp;
	int ret;

	fp = fopen(f.c_str(), "rb");
	if (NULL == fp) 	
		printf("Open file %s failed.\n", f.c_str());
	fseek(fp, 0, SEEK_END);

	int	rknn_model_size = ftell(fp);	// cancel rknn_model_size in model class
	ret = fseek(fp, 0, SEEK_SET);
	if (ret != 0)	
		printf("SEEK_SET failure.\n");
	
	unsigned char* rknn_model_data = (unsigned char*)malloc(rknn_model_size);	// allocate mem for rknn model	
	if (rknn_model_data == NULL)	
		printf("rknn model malloc failure.\n");
	
	ret = fread(rknn_model_data, 1, rknn_model_size, fp);	// read model
	fclose(fp);
	std::cout << "> RKNN Model Load.\n"; 	// info

	// rknn init
    ret = rknn_init(&this->ctx, rknn_model_data, rknn_model_size, rknn_init_flag);			// RKNN_FLAG_ASYNC_MASK 
    if (ret < 0)	
    	printf("rknn_init error ret=%d\n", ret);
	printf("> RKNN init succeed.\n");

	// free memory(rknn model data) on the fly
	if (rknn_model_data)	
		free(rknn_model_data);

	// Rknn input num & output num ==>  rknn_input_output_num 
	ret = rknn_query(this->ctx, RKNN_QUERY_IN_OUT_NUM, &this->io_num, sizeof(this->io_num));
	if (ret < 0)	
		printf("\nrknn_init error ret=%d\n", ret);
	printf("> Num_inputs: %d  |  Num_outputs: %d\n", this->io_num.n_input, this->io_num.n_output);

	// input attr ==> rknn_tensor_attr 
	rknn_tensor_attr input_attrs[this->io_num.n_input];
	for (int i = 0; i < this->io_num.n_input; i++) {
	    input_attrs[i].index = i;
	    ret = rknn_query(this->ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
	    if (ret < 0)	
	    	printf("\nrknn_query_rknn_tensor_attr input_attrs error ret=%d\n", ret);
		printf("[Input]: index=%d dims=[%d, %d, %d, %d] size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
			   	input_attrs[i].index, input_attrs[i].dims[3], 
				input_attrs[i].dims[2], input_attrs[i].dims[1], input_attrs[i].dims[0],  
			  	input_attrs[i].size, input_attrs[i].fmt, input_attrs[i].type, input_attrs[i].qnt_type, 
			   	input_attrs[i].fl, input_attrs[i].zp, input_attrs[i].scale); 	
	}
		
	// output attr
	rknn_tensor_attr output_attrs[this->io_num.n_output];	
	for (int i = 0; i < this->io_num.n_output; i++) {
	    output_attrs[i].index = i;
	    ret = rknn_query(this->ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
	    if (ret < 0)
	    	printf("\nrknn_query_rknn_tensor_attr output_attrs error ret=%d\n", ret);
		printf("[Output]: index=%d dims=[%d, %d, %d, %d] size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
			   	output_attrs[i].index, output_attrs[i].dims[3], 
				output_attrs[i].dims[2], output_attrs[i].dims[1], output_attrs[i].dims[0], 
			  	output_attrs[i].size, output_attrs[i].fmt, output_attrs[i].type, output_attrs[i].qnt_type, 
			   	output_attrs[i].fl, output_attrs[i].zp, output_attrs[i].scale);   
		
	}	

	// get model num_class
	this->nc = output_attrs[0].dims[2] - 5;		
	std::cout << "> Num_Classes: " << this->nc << "\n";	

	// rknn shape format 
	if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
	    this->width = input_attrs[0].dims[0];
		this->height = input_attrs[0].dims[1];
	    this->ch = input_attrs[0].dims[2];
	} else {
	    this->width = input_attrs[0].dims[1];
	    this->height = input_attrs[0].dims[2];
	    this->ch = input_attrs[0].dims[0];
	}
	

	// Model input data setting
	// this->rknn_inputs = new rknn_input[this->io_num.n_input];		
	for (int i = 0; i < this->io_num.n_input; i++) {
		this->rknn_inputs[i].index = i;
		this->rknn_inputs[i].type = RKNN_TENSOR_UINT8;		
		this->rknn_inputs[i].size = input_attrs[i].size;    // this->width * this->height * this->num_channel; 
		this->rknn_inputs[i].fmt = RKNN_TENSOR_NHWC;
		this->rknn_inputs[i].pass_through = false;  
	}

	// output sets 
	// TODO
	// this->rknn_outputs = new rknn_output[this->io_num.n_output];	
	memset(this->rknn_outputs, 0, sizeof(this->rknn_outputs)); 		// remove will cause problems 
	for (int i = 0; i < this->io_num.n_output; i++) {
		this->rknn_outputs[i].want_float = true;	//  use uint8_t type or float type
		// rknn_outputs[i].is_prealloc = false; 	// = 1, user decide where to allocate buffer and release Mem by himself; = 0, rknn auto mode. 
		// this->rknn_outputs[i].index = i;			
		// this->rknn_outputs[i].size = output_attrs[i].size;
	}

}



// rknn run
void RKNN::Infer() { 
	int ret = rknn_run(this->ctx, NULL);		// rknn inference	 
	if (ret < 0) 
		std::cout << "rknn_run() error = " << ret << "\n";

	ret = rknn_outputs_get(this->ctx, this->io_num.n_output, this->rknn_outputs, NULL); 		// get model outputs
	if (ret < 0) 
		std::cout << "rknn_outputs_get() error = " << ret << "\n";
}


// rknn ctx destory
void RKNN::Release() {
	if (this->ctx > 0)	rknn_destroy(this->ctx);	// ctx
	// if (rknn_inputs)	
	// 	delete[] rknn_inputs;		// inputs
	// if (rknn_outputs)	
	// 	delete[] rknn_outputs;		// outputs

}


/* --------------------
 	YOLO Parts
-----------------------*/
void YOLO::_PreProcess(cv::Mat& image_in, cv::Mat& image_out) {

	// resize 
    this->scale4lb = std::min((float)this->width / image_in.cols, (float)this->height / image_in.rows); 

    // TODO: if this->scale4lb > 0 { }
	cv::resize(image_in, image_out, cv::Size(), this->scale4lb, this->scale4lb, cv::INTER_AREA);

	// padding
	this->padding[0] = floor((this->height - image_out.size().height) / 2.0);		// top
	this->padding[1] = floor((this->width - image_out.size().width) / 2.0);			// left
	this->padding[2] = ceil((this->height - image_out.size().height) / 2.0);		// bottom
	this->padding[3] = ceil((this->width - image_out.size().width) / 2.0);			// right

	// Set to left-top
	// this->padding[0] = 0;		// top
	// this->padding[1] = 0;		// left
	// this->padding[2] = this->height - image_out.size().height;	// bottom
	// this->padding[3] = this->width - image_out.size().width;		// right

	// make boarder
	cv::copyMakeBorder(image_out, image_out, this->padding[0], this->padding[2], this->padding[1], this->padding[3], cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));	
	cv::cvtColor(image_out, image_out, CV_BGR2RGB); // BGR -> RGB

	// [rknn] inputs set, send images to rknn inputs;
	for (int i = 0; i < this->io_num.n_input; i++) {
		this->rknn_inputs[0].buf = image_out.data;			
	}
	rknn_inputs_set(this->ctx, this->io_num.n_input, this->rknn_inputs);

}


// for un-permute type => shape like [(1,255,80,80), (1,255,40,40), (1,255,20,20)]
void YOLO::_Decode(float* input, int stride, float threshold, std::vector<BBOX>& bboxes) {

	int grid_h = this->height / stride; 
	int grid_w = this->width / stride;
	int grid_len = grid_h * grid_w;
	
	// loop 
    for (int b = 0; b < 1; b++) {   		// each grid predict 1 box(yolov5 is 3). 
        for (int i = 0; i < grid_h; i++) {			
            for (int j = 0; j < grid_w; j++) {		

            	// offset
                int offset = ((this->nc + 5) * b) * grid_len + i * grid_w + j;
                float *pos = input + offset;	// current pos
                float conf_obj = this->_sigmoid(pos[4 * grid_len]);		// conf
				
				// filter
                for (int class_idx = 0; class_idx < this->nc; class_idx++) {

                	// conf
                    float conf_cls = this->_sigmoid(pos[(5 + class_idx) * grid_len]);		// class prob

					// filter
                    float score = conf_obj * conf_cls;
                    if (score >= threshold ) {

						// x1x2y1y2
				        float cx = (*pos + j) * stride;		// (cx + grid_i) * stride
				        float cy = (pos[grid_len] + i) * stride;	// (cy + grid_y) * stride              
				        float w = std::exp(pos[2 * grid_len]) * stride;		// exp(w) * stride
				        float h = std::exp(pos[3 * grid_len]) * stride;		// exp(h) * stride
						float x1 = cx - w / 2.0;			// bbox.left
						float y1 = cy - h / 2.0;			// bbox.top
						this->_xyScaleCoords(x1, y1);
						this->_whScaleCoords(w, h);

						// temp bbox to save
			            BBOX bbox_temp;
			            bbox_temp.rect = cv::Rect_<float>(x1, y1, w, h);
			            bbox_temp.id = class_idx;
			            bbox_temp.score = score;
						
						// save results
			            bboxes.emplace_back(bbox_temp);
                    }
                }
            }
        }
    }
}


// post-process
void YOLO::_PostProcess(float conf_thresh, float nms_thresh) {

	// clear before push
	this->detections.clear();
	
	// compute all valid detections & save all bboxes
	for (int i = 0; i < this->io_num.n_output; i++) {
		_Decode((float*)this->rknn_outputs[i].buf, std::pow(2, 3 + i), conf_thresh, this->detections);   // stride = std::pow(2, 3 + i): 8, 16, 32, (64)
	}

	// clear current frame results
	int ret = rknn_outputs_release(this->ctx, this->io_num.n_output, this->rknn_outputs);	
	if (ret < 0)	
		std::cout << "rknn_outputs_release() error!\n";

	// NMS
	_NonMaxSuppression(nms_thresh, this->detections);

}

//TODO: By class_id
void YOLO::_NonMaxSuppression(float threshold, std::vector<BBOX>& bboxes) {
  	std::sort(bboxes.begin(), bboxes.end(), [](BBOX a, BBOX b) { return a.score > b.score; } );  // sort(>=) by score
    for (int i = 0; i < int(bboxes.size()); i++) {
        for (int j = i + 1; j < int(bboxes.size()); ) {
        	if (this->_bboxes_iou(bboxes[i], bboxes[j]) >= threshold) {
                bboxes.erase(bboxes.begin() + j);
            } else { j++; }
        }
    }
}



// pre-process + inference + post-process
void YOLO::Detect(cv::Mat& frame, float conf_thresh, float nms_thresh) {
	cv::Mat image;		// do letterbox, for inference
	this->_PreProcess(frame, image);      // pre-process
	RKNN::Infer(); // inference
	this->_PostProcess(conf_thresh, nms_thresh);   	 //	post-process 
}


// track 
void YOLO::Track(BYTETracker& tracker, cv::Mat& frame, bool enable_trajectory) {

	// tracking 
	this->stracks = tracker.update(this->detections, this->stracks_all);
	
	// calculate trajectory ==>  {tracker_id: <[(cx, cy), ...], state>} 
	if (enable_trajectory) {
	
		// saving all tracks
		for (int i = 0; i < this->stracks_all.size(); i++) {
			// std::cout << "idx: " << this->stracks_all[i].track_id << " frame_id: " << this->stracks_all[i].frame_id << " state: " << this->stracks_all[i].state << " length: " << this->stracks_all[i].tracklet_len << std::endl;
			
			// calculate center point when tracked
			std::vector<cv::Point> center;
			float cx, cy;
			if (stracks_all[i].state == 1) {	// tracked: 1
				cx = stracks_all[i].tlwh[0] + stracks_all[i].tlwh[2] / 2;
				cy = stracks_all[i].tlwh[1] + stracks_all[i].tlwh[3] / 2;			
				center.emplace_back(cv::Point(cx, cy));	
			}
			
			// update state
			trajectories[stracks_all[i].track_id].second = stracks_all[i].state;	
			
			// generate trajectories
			if (trajectories.find(stracks_all[i].track_id) == trajectories.end()) { 	// not find track_id  ==>  create new one
				trajectories[stracks_all[i].track_id].first = center;
			} else {	// find track_id 
				if (stracks_all[i].state == 1) {	// lost & removed  ==>  update;  lost & removed  ==>  do nothing
					trajectories[stracks_all[i].track_id].first.emplace_back(cv::Point(cx, cy));	// update state
				} 
			}
		}
	}
}

// draw
void YOLO::Draw(cv::Mat& frame) {

	// 1.for tracking
	if (stracks.size() > 0) {
	
		// tracking labels * bboxes
		for (int i = 0; i < stracks.size(); i++) {
			vector<float> tlwh = stracks[i].tlwh;
			cv::putText(frame, 
						format("%d [%d] [%d]: %.2f", stracks[i].track_id, stracks[i].class_id, stracks[i].tracklet_len, stracks[i].score), 
						Point(tlwh[0], tlwh[1] - 5), 
		            	cv::FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 
		            	1, LINE_AA);
		    cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), 
		    		      this->_get_color(stracks[i].track_id), 2);
		}
		
		// draw trajecories
		if (trajectories.size() > 0) {
			for (auto trajectory: trajectories) {
				if (trajectory.second.second == 1) {	// tracked, not lost or removed
					for (auto it: trajectory.second.first) {	// trajectory.second  => <center_points,state>
						cv::circle(frame, it, 4, this->_get_color(trajectory.first), -1);
					}
				} 
			}
		}
	} else {	// 2.for detection
		for (std::vector<BBOX>::iterator it = this->detections.begin(); it != this->detections.end(); ++it) {
			// if ((it->rect.width <= 40) && (it->rect.height <= 40)) continue;	

			// detection 
			cv::rectangle(frame, it->rect.tl(), it->rect.br(), this->_get_color(it->id), 2);
			cv::putText(frame, 
						format("%d: %.2f", it->id, it->score), 						
						cv::Point(it->rect.tl().x, it->rect.tl().y - 5), 
					    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 
					    2, LINE_AA);	
		}

	}

}


// generate color
cv::Scalar YOLO::_get_color(int idx) {
	idx += 3;
	return cv::Scalar(137 * idx % 255, 77 * idx % 255, 129 * idx % 255);
}


// release
void YOLO::Release() {
	RKNN::Release();	// release rknn
	// ..., others mem 

}










