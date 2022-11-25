//
// Created by chenpeng on 2019/5/8.
//
#include "AlgoApplePerception.h"

#include <chrono>
#include <cmath>
#include <time.h>
#include <fstream>
#include <assert.h>
#include <algorithm>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#define SHOW_IMG_RES
//#define USE_TOF_IN_PERC
#define TAG "AlgoApplePerception"

using namespace cv;
using namespace std;
using namespace ninebot_algo;
using namespace cnn_ninebot;
// using namespace segway_scooter;

AlgoApplePerception::AlgoApplePerception(){
    _sidewalk_config.enable_multi_thread = false;
    _sidewalk_config.input_width = 512;
    _sidewalk_config.input_height = 512;
    _sidewalk_config.input_depth = 3;
    _sidewalk_config.grid_h = 16;
    _sidewalk_config.grid_w = 16;
    _sidewalk_config.num_object = 3;
    _sidewalk_config.classes = 1;
    _sidewalk_config.conf_thresh = 0.45;
    _sidewalk_config.class_thresh = 0.5;
    _sidewalk_config.nms_thresh = 0.3;
    _sidewalk_config.frozen_net_path = "/sdcard/apple_model.tflite";
    _sidewalk_config.num_classes = 3;
    _sidewalk_config.softmax_CE = true;
    _sidewalk_config.robot_base_type=3000;

    _uq_pedestrian_perception = std::make_shared<ninebot_algo::cnn_ninebot::ApplePerception>(_sidewalk_config);
}

std::vector<int> AlgoApplePerception::getFrontMask(){
    segmentor_config sidewalk_config = _uq_pedestrian_perception->get_segmentor_config();
    int basetype = sidewalk_config.robot_base_type;
    bool is_large_fov_ = true;
    std::vector<int> front_mask;
    if(is_large_fov_){
        if(basetype <= 1004)
            front_mask.assign(sidewalk_config.front_mask_border_pts_large_fov.begin(),
                                sidewalk_config.front_mask_border_pts_large_fov.end());
        else if(basetype <= 1006)
            front_mask.assign(sidewalk_config.front_mask_border_pts_large_fov_1005.begin(),
                                sidewalk_config.front_mask_border_pts_large_fov_1005.end());
        else if(basetype <= 1008)
            front_mask.assign(sidewalk_config.front_mask_border_pts_large_fov_1007.begin(),
                                sidewalk_config.front_mask_border_pts_large_fov_1007.end());
        else
            front_mask.assign(sidewalk_config.front_mask_border_pts_large_fov_2000.begin(),
                                sidewalk_config.front_mask_border_pts_large_fov_2000.end());
    }
    else
        front_mask.assign(sidewalk_config.front_mask_border_pts.begin(),
                            sidewalk_config.front_mask_border_pts.end());
    return front_mask;
}

void AlgoApplePerception::interset(int &x1, int &y1, int &x2, int &y2, int w_input, int h_input){
    // 防止预测的行人边框超过网络输入图边界
    x1 = max(_crop_config.crop_x, x1);
    y1 = max(_crop_config.crop_y, y1);
    x2 = min(_crop_config.crop_x + w_input - 1, x2);
    y2 = min(_crop_config.crop_y + h_input - 1, y2);
}


void AlgoApplePerception::setCropParameter(crop_config &_crop_config, int x, int y, int w, int h){
    _crop_config.crop_x = x;
    _crop_config.crop_y = y;
    _crop_config.crop_w = w;
    _crop_config.crop_h = h;
}

vector<bbox> AlgoApplePerception::PerceptionProcess(const cv::Mat &frame){

    auto coreAlgoStart = std::chrono::high_resolution_clock::now();

    // Init
    int original_width = frame.cols;
    int original_height = frame.rows;
    front_mask = getFrontMask();
    int handLoc = front_mask[3] % _sidewalk_config.input_width *
            original_width / _sidewalk_config.input_width;
    //ALOGTAGD(TAG,"VisionLog pedestrainPerception Time");

    std::vector<bbox> pedestrian_res;
    if((1920 == frame.cols) && (1080 == frame.rows)){
        setCropParameter(_crop_config, 240, 0, 1440, 1080);
        cv::Mat cropped_frame;
        cv::Rect crop_rect = cv::Rect(_crop_config.crop_x, _crop_config.crop_y, _crop_config.crop_w, _crop_config.crop_h);
        cropped_frame = frame(crop_rect);
        (*_uq_pedestrian_perception)(cropped_frame, pedestrian_res, handLoc);
    }
    else if((1280 == frame.cols) && (720 == frame.rows)){
        setCropParameter(_crop_config, 160, 0, 1120, 720);
        (*_uq_pedestrian_perception)(frame, pedestrian_res, handLoc);
    }
    else if((960 == frame.cols) && (540 == frame.rows)){
        setCropParameter(_crop_config, 0, 0, 640, 480);
        (*_uq_pedestrian_perception)(frame, pedestrian_res, handLoc);
    }
    else if((640 == frame.cols) && (480 == frame.rows)){
        setCropParameter(_crop_config, 0, 0, 640, 480);
        (*_uq_pedestrian_perception)(frame, pedestrian_res, handLoc);
    }
    else{
        setCropParameter(_crop_config, 0, 0, 640, 480);
        (*_uq_pedestrian_perception)(frame, pedestrian_res, handLoc);
    }

    // Pedestrian PostProcess
    std::vector<cv::Point> pedestrian_coord, other_coord;
    std::vector<cv::Rect> pedestrian_bboxes, other_bboxes;
    std::vector<int> box_offset;

    cv::Mat mask;
    cv::Mat seg_choose;

    for (int bid = 0; bid < pedestrian_res.size(); bid++){
        int ptx1_resized = _crop_config.crop_x + (int)(pedestrian_res[bid].x1 * _crop_config.crop_w);
        int ptx2_resized = _crop_config.crop_x + (int)(pedestrian_res[bid].x2 * _crop_config.crop_w);
        int pty1_resized = _crop_config.crop_y + (int)(pedestrian_res[bid].y1 * _crop_config.crop_h);
        int pty2_resized = _crop_config.crop_y + (int)(pedestrian_res[bid].y2 * _crop_config.crop_h);

        interset(ptx1_resized, pty1_resized, ptx2_resized, pty2_resized,
                _crop_config.crop_w, _crop_config.crop_h);

        pedestrian_res[bid].x1 = ptx1_resized;
        pedestrian_res[bid].x2 = ptx2_resized;
        pedestrian_res[bid].y1 = pty1_resized;
        pedestrian_res[bid].y2 = pty2_resized;
    }
    return pedestrian_res;

}


