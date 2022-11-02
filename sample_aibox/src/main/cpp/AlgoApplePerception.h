#ifndef ALGOAPPLEPERCEPTION_H
#define ALGOAPPLEPERCEPTION_H

// #include "AlgoBase.h"
#include "./include/ApplePerception.h"
#include <map>

namespace ninebot_algo {
    namespace cnn_ninebot {
        class AlgoApplePerception{
        public:
            AlgoApplePerception();

            std::vector<bbox> PerceptionProcess(const cv::Mat &frame);

            bool step();    // run algorithm once
            // RawData *_main_rawdata;

            float runTime();    // return the runtime of algorithm in ms, stat this number in step()
            bool showScreen(
                    void *pixels);    // output the content inside a char array *data, the memory is allocated outside


            void getLock();

            void releaseLock();

        private:

            //llx20220424  for 1280*720
            struct crop_config {
                int crop_x = 240;
                int crop_y = 0;
                int crop_w = 1440;
                int crop_h = 1080;
            } _crop_config;



            bool _canDetection = false;

            std::shared_ptr<ApplePerception> _uq_pedestrian_perception;

            // std::unique_ptr<ScooterDataProcessor> _uq_SDP;
            segmentor_config _sidewalk_config;

            bool _is_render;

            std::mutex _mutex_timer;
            std::mutex _mutex_display;
            std::mutex _mutex_getRes;
            std::mutex _mutex_stepdelay;

            float _ptime;
            float _step_time_ave=-1;
            float _step_time=-1;
            float _core_time=-1;

            // StampedMat _raw_fisheye_yuv;

            cv::Mat _canvas;

            // config folder
            std::string m_config_folder;

            int frameCnt = 0;

            // yolov3
            std::vector<int> front_mask;
            std::vector<int> pedestrainFilVec;

            // segway_scooter::Scooter * _mContext;

            // StampedOrientation _orientation_val;

        #if defined _USE_H_FROM_TF
            cv::Mat _fisheye_to_birdview_transform = (cv::Mat_<float>(3,3) << 0.351310,0.000000,27.539680,-0.235978,0.303552,122.242065,-0.001180,0.000000,1.000000);
        #else
            cv::Mat _fisheye_to_birdview_transform = (cv::Mat_<float>(3,3) << 5.76677149e-01, -6.34594425e-03, -1.51908185e+02,
                1.25732169e-02,  3.03323585e-01, -8.06119408e+01,
                -9.70264761e-04, -3.88964426e-05,  1.00000000e+00);
        #endif
        //            float _cam_paras[5] = {275.543825,273.041509,482.351164,270.168540,0.994796};
            //float _cam_paras[5] = {270.444126, 266.361114, 475.813021, 269.730255, 0.991961};
            float _cam_paras[5] = {568.714800,566.166463,712.200828,537.018834,0.993844};

            void undistort_fisheye_points(std::vector<float>& pts_x, std::vector<float>& pts_y, float* fisheyeParameter);
            void transform_pts_to_birdview(std::vector<float>& pts_x, std::vector<float>& pts_y);
            void transform_pts_to_distance(std::vector<float>& pts_x_bird, std::vector<float>& pts_y_bird,
                    std::vector<float>& pts_x, std::vector<float>& pts_y,
                    std::vector<float>& distance, std::vector<cv::Rect>& bboxes, const int valid_dis);
            void roi_compare(const cv::Mat area0, const cv::Mat area1, bool &sidewalk_flag, bool &valid, const float valid_thresh);
            std::vector<int> getFrontMask();
            void interset(int &x1, int &y1, int &x2, int &y2, int w_input, int h_input);
            float frame_coverage(cv::Mat seg_roi, int x1, int y1, int x2, int y2);
            void setCropParameter(crop_config &_crop_config, int x, int y, int w, int h);
        };
    }
}
#endif
