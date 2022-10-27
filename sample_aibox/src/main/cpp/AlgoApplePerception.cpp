//
// Created by chenpeng on 2019/5/8.
//
#include "AlgoApplePerception.h"
// #include "../util/app_callback.h"
// #include "../core/Scooter.h"
#include <chrono>
#include <cmath>
#include <time.h>
#include <fstream>
#include <assert.h>
#include <algorithm>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

// #include "ImuProcess.h"
// #include "RawDataUtil.h"


#define SHOW_IMG_RES
//#define USE_TOF_IN_PERC
#define TAG "AlgoApplePerception"
#define PEDEST  "pedestrian"
#define SIDEW   "sidewalk"
#define PARKS  "parkingSpace"

using namespace cv;
using namespace std;
using namespace ninebot_algo;
using namespace cnn_ninebot;
// using namespace segway_scooter;

// AlgoPedestrianPerception::AlgoPedestrianPerception(RawData* rawInterface,int run_sleep_ms,
//                                                std::shared_ptr<cnn_ninebot::sidewalk_perception> ptrCnnPedestrian,
//                                                Scooter * context, bool is_render)
//         :AlgoBase(rawInterface,run_sleep_ms)
// {
//     mThreadName="pedestrainPerception";

//     _ptime=200;

//     _canvas = cv::Mat::zeros(cv::Size(640, 360), CV_8UC3);
//     _main_rawdata = mRawDataInterface;
//     this->_is_render = is_render;

//     _uq_pedestrian_perception=ptrCnnPedestrian;

//     _uq_SDP = std::make_unique<ScooterDataProcessor>(_main_rawdata);
//     _uq_SDP->init();

//     _mContext=context;
//     segmentor_config sidewalk_config = _uq_pedestrian_perception->get_segmentor_config();
//     front_mask = getFrontMask();


//     ALOGTAGD(TAG,"VisionLog pedestrainPerception init finished.");
// }

// AlgoPedestrianPerception::~AlgoPedestrianPerception() {

//     ALOGTAGD(TAG,"VisionLog pedestrainPerception destroy.");

//     _uq_pedestrian_perception.reset();

// }




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



void AlgoApplePerception::PerceptionProcess(const cv::Mat &frame){
    // frameCnt++;
    // auto start = std::chrono::high_resolution_clock::now();
    // _main_rawdata->retrieveFisheyeYUV(_raw_fisheye_yuv);

    // if (_raw_fisheye_yuv.timestampSys == 0)
    // {
    //     ALOGTAGE(TAG,"VisionLog _raw_color.timestampSys !");
    //     return false;
    // }
    // struct timeval tv;
    // gettimeofday(&tv,NULL);
    // long tmpfisheyetime=tv.tv_sec * 1000 + tv.tv_usec / 1000-_raw_fisheye_yuv.timestampSys/1000;
    // /*
    // if(tmpfisheyetime>200){
    //     ALOGTAGE(TAG,"VisionLog finsheye delay >200ms !");
    //     return false;
    // }
    // */
    // if(_raw_fisheye_yuv.image.channels() != 1)
    // {
    //     ALOGTAGD(TAG,"VisionLog _raw_color.image.channels() != 1 !");
    //     return false;
    // }

    // ALOGTAGD(TAG,"VisionLog AlgoApplePerception Begin!");
    std::shared_ptr<cnn_ninebot::sidewalk_perception> _uq_pedestrian_perception;


    // cv::Mat tcolor, frame;
    // cv::cvtColor(_raw_fisheye_yuv.image, tcolor, CV_YUV2BGR_NV12);
    // frame = tcolor.clone();

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
    // int64_t pathTimeStamp;
    // bool sidewalk_flag;

    // getLock();
    // seg_choose = _mContext->p_res->getSegChoose(pathTimeStamp);
    // sidewalk_flag = _mContext->p_res->getSideWalkFlag(pathTimeStamp);
    // mask = _mContext->p_res->getMask(pathTimeStamp);
    // releaseLock();

    for (int bid = 0; bid < pedestrian_res.size(); bid++){
        int ptx1_resized = _crop_config.crop_x + (int)(pedestrian_res[bid].x1 * _crop_config.crop_w);
        int ptx2_resized = _crop_config.crop_x + (int)(pedestrian_res[bid].x2 * _crop_config.crop_w);
        int pty1_resized = _crop_config.crop_y + (int)(pedestrian_res[bid].y1 * _crop_config.crop_h);
        int pty2_resized = _crop_config.crop_y + (int)(pedestrian_res[bid].y2 * _crop_config.crop_h);

        interset(ptx1_resized, pty1_resized, ptx2_resized, pty2_resized,
                _crop_config.crop_w, _crop_config.crop_h);

        rectangle(frame, cv::Point(ptx1_resized, pty1_resized),
                        cv::Point(ptx2_resized, pty2_resized), cv::Scalar(0, 255, 0), 2);

        // cv::Mat seg_choose_resize;
        // cv::resize(seg_choose, seg_choose_resize, cv::Size(original_width, original_height));


        // pedestrian_coord.push_back(cv::Point(
        //         (int)((ptx1_resized + ptx2_resized) / 2),
        //         pty2_resized));
        // pedestrian_bboxes.push_back(cv::Rect(
        //         ptx1_resized, pty1_resized,
        //         ptx2_resized,
        //         pty2_resized));
    }

    // auto coreAlgoPedFil = std::chrono::high_resolution_clock::now();

    // // Pedestrian Distance
    // std::vector<float> pts_x, pts_y;
    // for(int pt_id = 0; pt_id < pedestrian_coord.size(); ++pt_id){
    //     pts_x.push_back(pedestrian_coord[pt_id].x);
    //     pts_y.push_back(pedestrian_coord[pt_id].y);
    // }

    // if(pedestrainFilVec.size() >= 5){
    //     pedestrainFilVec.erase(pedestrainFilVec.begin(), pedestrainFilVec.begin() + 1);
    //     pedestrainFilVec.push_back(pts_x.size());
    // }
    // else{
    //     pedestrainFilVec.push_back(pts_x.size());
    // }

    // auto maxNum = max_element(pedestrainFilVec.begin(), pedestrainFilVec.end());
    // int maxPedestrainNum = *maxNum;
    // ALOGTAGD(TAG,"VisionLog pedestrainPerception personCnt %d persons", maxPedestrainNum);

    // auto coreAlgoEnd = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsedCore = coreAlgoEnd - coreAlgoStart;
    // ALOGTAGD(TAG,"VisionLog pedestrainPerception Time, total core algo cost %f ms", elapsedCore.count());
    // ALOGTAGD(TAG,"VisionLog pedestrainPerception res, personCnt, total detected %d persons", pts_x.size());


    // // Result
    // return maxPedestrainNum;
}



//int main(int argc, char** argv)
//{
//    string image_path = argv[1];
//    cout << "image_path:" << image_path << endl;
//
//    Mat image = imread(image_path);
//    if (image.empty())
//    {
//        cerr << "Read image " << image_path << " failed!";
//        exit(1);
//    }
//
//    AlgoApplePerception algoApplePerception;
//    algoApplePerception.PerceptionProcess(image);
//
//    imwrite("./res.jpg", image);
//}
