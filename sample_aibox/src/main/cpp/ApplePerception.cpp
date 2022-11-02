#include <android/log.h>

#include "./include/ApplePerception.h"

//notice the order of these above two headers, it will influence the "_DOUBLE_SPHERES_DISTORTION_"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

//#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

//#include "ninebot_log.h"
#include <iostream>
using namespace cv;
using std::vector;
using namespace ninebot_algo;


#define _USE_TF_FLOAT_MODEL

#define OVERLAP_FRONT_MASK

//#define _USE_ARGMAX_FOR_SEGMENTATION

#define CALCULATE_COST_TIME_

#define SP_LOG_TAG "sidewalk_perception_so_log"
#define SP_LOG(...) __android_log_print(ANDROID_LOG_DEBUG, SP_LOG_TAG, __VA_ARGS__)

namespace ninebot_algo{ namespace cnn_ninebot {
	ApplePerception::ApplePerception(const segmentor_config &cfg)
	{
		cfg_ = cfg;
		model_ = tflite::FlatBufferModel::BuildFromFile(cfg_.frozen_net_path.c_str());
		tflite::ops::builtin::BuiltinOpResolver resolver;
		tflite::InterpreterBuilder builder(*model_.get(), resolver);
		builder(&interpreter_);
        if(cfg_.enable_multi_thread)
            interpreter_->SetNumThreads(2);
		interpreter_->AllocateTensors();

        if(cfg_.enable_gpu_inference){
            const TfLiteGpuDelegateOptionsV2 options = {
                .is_precision_loss_allowed = 1,
                .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
            };
            delegate_ = TfLiteGpuDelegateV2Create(&options);

            if (interpreter_->ModifyGraphWithDelegate(delegate_) != kTfLiteOk) {
              return;
            }
        }

#if defined _FOV_DISTORTION_
        cam_paras_ = new float[5];
		cam_paras_[0] = cfg_.fx;
		cam_paras_[1] = cfg_.fy;
		cam_paras_[2] = cfg_.cx;
		cam_paras_[3] = cfg_.cy;
		cam_paras_[4] = cfg_.distortion;
#elif defined _DOUBLE_SPHERES_DISTORTION_
        cam_paras_ = new float[6];
		cam_paras_[0] = cfg_.fx;
		cam_paras_[1] = cfg_.fy;
		cam_paras_[2] = cfg_.cx;
		cam_paras_[3] = cfg_.cy;
		cam_paras_[4] = cfg_.ds_distortion_alpha;
		cam_paras_[5] = cfg_.ds_distortion_xi;
#endif
	}

    ApplePerception::~ApplePerception(){
        delete cam_paras_;
        if(cfg_.enable_gpu_inference){
            //Clean up.
            TfLiteGpuDelegateV2Delete(delegate_);
        }
    }

    // pedestrian detection
    void ApplePerception::operator() (const cv::Mat &src_img, std::vector<bbox> &pedestrian_boxes,
        int handLoc){

        std::lock_guard<std::mutex> guard(operator_mutex_);
        return run_quantization(src_img, pedestrian_boxes, handLoc);
    }

    float sigmoid(float x){
        return (1 / (1 + exp(-x)));
    }

    bool comp(const bbox &a, const bbox &b){
        return a.score > b.score; 
    }

    std::vector<bbox> nms(std::vector<bbox> boxes, int classes, float thresh)
    {
        std::vector<bbox> res;
        for(int i = 0; i < classes; i++)
        {
            vector<bbox> res_one_cls;
            for(int j = 0; j < boxes.size(); j++)
            {
                if(boxes[j].classId == i)
                    res_one_cls.push_back(boxes[j]);
            }
            sort(res_one_cls.begin(),res_one_cls.end(),comp);
            for(int j = 0; j < res_one_cls.size(); j++)
            {
                float area = (res_one_cls[j].x2 - res_one_cls[j].x1 + 0.0001) 
                                * (res_one_cls[j].y2 - res_one_cls[j].y1 + 0.0001);
                for(int k = res_one_cls.size()-1; k > j; k--)
                {
                    float xx1 = max(res_one_cls[j].x1, res_one_cls[k].x1);
                    float xx2 = min(res_one_cls[j].x2, res_one_cls[k].x2);
                    float yy1 = max(res_one_cls[j].y1, res_one_cls[k].y1);
                    float yy2 = min(res_one_cls[j].y2, res_one_cls[k].y2);
                    float w = xx2 - xx1 + 0.0001;
                    float h = yy2 - yy1 + 0.0001;
                    if (w > 0 && h > 0)
                    {
                        float o = w * h / area;
                        if (o > thresh)
                        {
                            res_one_cls.erase(res_one_cls.begin() + k);
                        }
                    }
                }
            }
            res.insert(res.end(), res_one_cls.begin(), res_one_cls.end());
        }
        return res;
    }

    void ApplePerception::pedestrian_yolo_parse(const std::vector<TfLiteTensor*> outs, std::vector<bbox> &pedestrian_boxes){
        std::vector<bbox> boxes;
        for (int s = 0; s < outs.size(); ++s){
            float* resdata = (float*)outs[s]->data.uint8;
            int scale = pow(2, s); 
            int outputwidth = cfg_.grid_w * scale;
            int outputheight = cfg_.grid_h * scale;
            int map_pixel_size = outputheight * outputwidth;
            for (int i = 0; i < map_pixel_size; i++){
                int j = i * (cfg_.num_object * (cfg_.classes + 5));
                for (int k = 0; k < cfg_.num_object; k++){
                    int obj = k * (cfg_.classes + 5);
                    // conf
                    float conf = sigmoid(resdata[j + obj + 4]);
                    if(conf < cfg_.conf_thresh)
                        continue;
                    // class
                    float cls_score;
                    int max_class = 0;
                    float class_conf = -10000.;
                    for (int cls = 0; cls < cfg_.classes; cls ++){
                        cls_score = sigmoid(resdata[j + obj + cls + 5]);
                        if (cls_score > class_conf){
                            class_conf = cls_score;
                            max_class = cls;
                        }
                    }
                    // bbox
                    float bw = exp(resdata[j + obj + 2]);
                    float bh = exp(resdata[j + obj + 3]);
                    float width = bw * (cfg_.anchors[s * cfg_.num_object * 2 + k * 2] / cfg_.input_w); // anchor.x / input_w
                    float height = bh * (cfg_.anchors[s * cfg_.num_object * 2 + k * 2 + 1] / cfg_.input_h); // anchor.y / input_h
                    float offset_x = sigmoid(resdata[j + obj]);
                    float offset_y = sigmoid(resdata[j + obj + 1]);
                    bbox box;
                    int y = i / outputwidth;
                    int x = i % outputwidth;
                    box.x1 = (x + offset_x) / outputwidth - width / 2;
                    box.y1 = (y + offset_y) / outputheight - height / 2;
                    box.x2 = (x + offset_x) / outputwidth + width / 2;
                    box.y2 = (y + offset_y) / outputheight + height / 2;
                    box.x1 = box.x1 > 0 ? box.x1 : 0;
                    box.y1 = box.y1 > 0 ? box.y1 : 0;
                    box.x2 = box.x2 < 1 ? box.x2 : 1;
                    box.y2 = box.y2 < 1 ? box.y2 : 1;

                    box.score = class_conf * conf;
                    box.classId = max_class;
                    boxes.push_back(box);
                }
            }

        }
        boxes = nms(boxes, cfg_.classes, cfg_.nms_thresh);
        pedestrian_boxes.assign(boxes.begin(), boxes.end());
    }


    void ApplePerception::run_quantization(const cv::Mat &src_img, std::vector<bbox> &pedestrian_boxes,
            int handLoc){
#ifdef CALCULATE_COST_TIME_
        auto start = std::chrono::high_resolution_clock::now();
#endif
        auto input_width = cfg_.input_width;
        auto input_height = cfg_.input_height;
        cv::Mat cropped_img, rotated_img, resized_img, normalized_img;
        cropped_img = cv::Mat::zeros(src_img.rows, src_img.cols - handLoc, CV_8UC3);
        src_img(cv::Rect(handLoc, 0.0, src_img.cols - handLoc, src_img.rows)).copyTo(cropped_img);
        cv::resize(src_img, resized_img, cv::Size(input_width, input_height), 0, 0, cv::INTER_NEAREST);
        resized_img.convertTo(normalized_img, CV_32F, 1.0 / 255, 0);

#ifdef _USE_TF_FLOAT_MODEL
        auto input = interpreter_->typed_input_tensor<float>(0);

        float *input_data = (float*)normalized_img.data;
        int input_idx = 0;
        while(input_idx < input_height * input_width * 3){
            input[input_idx] = *input_data;
            input_idx ++;
            input_data ++;
        }
#else
        auto input_node_index = interpreter_->inputs()[0];
        auto input = interpreter_->typed_tensor<std::uint8_t>(input_node_index);//float
        memcpy(input, resized_img.data, resized_img.rows*resized_img.cols*3);

#endif

#ifdef CALCULATE_COST_TIME_
        auto end0 = std::chrono::high_resolution_clock::now();
#endif

        if(interpreter_->Invoke() != kTfLiteOk){
            return;
        }


#ifdef CALCULATE_COST_TIME_
        auto end1 = std::chrono::high_resolution_clock::now();
#endif
        auto pedestrian_yolo_node_index_0 = interpreter_->outputs()[0];
        TfLiteTensor* p_tflts_output0 = interpreter_->tensor(pedestrian_yolo_node_index_0);
        auto pedestrian_yolo_node_index_1 = interpreter_->outputs()[1];
        TfLiteTensor* p_tflts_output1 = interpreter_->tensor(pedestrian_yolo_node_index_1);
        auto pedestrian_yolo_node_index_2 = interpreter_->outputs()[2];
        TfLiteTensor* p_tflts_output2 = interpreter_->tensor(pedestrian_yolo_node_index_2);
        std::vector<TfLiteTensor*> tensors = {p_tflts_output0, p_tflts_output1, p_tflts_output2};
        pedestrian_yolo_parse(tensors, pedestrian_boxes);


#ifdef CALCULATE_COST_TIME_
        auto end2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed0 = end0 - start;
        std::chrono::duration<double, std::milli> elapsed1 = end1 - end0;
        std::chrono::duration<double, std::milli> elapsed2 = end2 - end1;
#endif
    }
   
    segmentor_config ApplePerception::get_segmentor_config(){
        return cfg_;
    }

}}