#ifndef APPLEPERCEPTION_H
#define APPLEPERCEPTION_H

#include <thread>
#include <mutex>
#include <tensorflow/lite/context.h>
#include <opencv2/opencv.hpp>
#include <unordered_map>

#define _FOV_DISTORTION_

namespace tflite {
	class Interpreter;
	class FlatBufferModel;
}

namespace ninebot_algo { namespace cnn_ninebot {
	struct bbox
	{
		int classId;
		float x1;
		float y1;
		float x2;
		float y2;
		float score;
	};
	struct segmentor_config {
		int input_width = 512;
		int input_height = 512;
        int input_depth = 3;
		int num_classes = 3;
		std::string frozen_net_path;
        bool enable_multi_thread = false;

#if defined _FOV_DISTORTION_
		float fx = 467.3418184973979 * 512 / 1280; //184.0;
		float fy = 466.9720339209149 * 512 / 720; //327.0;
		float cx = 651.5441842425998 * 512 / 1280; //256.0;
		float cy = 371.1062915055246 * 512 / 720; //256.0;
		float distortion = 0.995537338859336; //1.0;
#elif defined _DOUBLE_SPHERES_DISTORTION_
        float fx = 496.5533223034946 * 512 / 1280;
        float fy = 497.1418830952384 * 512 / 720;
		float cx = 664.8323519339328 * 512 / 1280;
        float cy = 370.04013896202156 * 512 / 720;
        float ds_distortion_alpha = 0.7304240489005107;
        float ds_distortion_xi = -0.04279171468833978;
#endif

		std::vector<int> front_mask_border_pts{85, 223, 77535, 77563, 184571, 184543, 261855, 261717};//small fov
		//std::vector<int> front_mask_border_pts_large_fov{114, 254, 80126, 80142, 182030, 182014, 261886, 261746};//large fov
		std::vector<int> front_mask_border_pts_large_fov{113, 233, 85225, 85241, 176889, 176873, 261865, 261745};//large fov 2020-4-30
		//std::vector<int> front_mask_border_pts{78,223,77535,77561,184569,184543,261855,261710};
		std::vector<int> front_mask_border_pts_post_process{82, 226, 76002, 76030, 186110, 186082, 261858, 261714};//small fov
		//std::vector<int> front_mask_border_pts_post_process_large_fov{111, 257, 78593, 78609, 183569, 183553, 261889, 261743};//large fov
		std::vector<int> front_mask_border_pts_post_process_large_fov{110, 236, 83692, 83708, 178428, 178412, 261868, 261742};//large fov 2020-4-30

		std::vector<int> front_mask_border_pts_large_fov_1005{147, 261, 76037, 76075, 186155, 186117, 261893, 261779};
		std::vector<int> front_mask_border_pts_post_process_large_fov_1005{144, 264, 74504, 74542, 187694, 187656, 261896, 261776};//large fov 2020-7-17
		
		std::vector<int> front_mask_border_pts_large_fov_1007{126, 238, 77038, 77093, 185125, 185070, 261870, 261758};
		std::vector<int> front_mask_border_pts_post_process_large_fov_1007{123, 241, 75505, 75560, 186664, 186609, 261873, 261755};//large fov 2020-12-15

		std::vector<int> front_mask_border_pts_large_fov_2000{0, 233, 94953, 95001, 167193, 167145, 261865, 261632};
		std::vector<int> front_mask_border_pts_post_process_large_fov_2000{0, 236, 93420, 93468, 168732, 168684, 261868, 261632};//large fov 2021-03-28
		
		bool enable_gpu_inference = true;
		float mean_b = 127.9489;
		float mean_g = 125.3112;
		float mean_r = 125.6642;
		float standard_deviation_reciprocal = 0.0154;

		float small_h_fov = 152;//degree
		float small_v_fov = 79;
		float large_h_fov = 210;
		float large_v_fov = 100;

		int robot_base_type = 2000;

		uchar background_label = 0;
		uchar road_label = 128;
		uchar sidewalk_label = 255;

		bool softmax_CE = true;

		// yoloV3
		std::vector<float> anchors{169.54, 131.54, 228.0, 289.38, 545.15, 476.46, 43.85, 89.15, 90.62, 65.77, 86.23, 173.92, 14.62, 19.0, 23.38, 43.85, 48.23, 33.62};
		
		int grid_h = 16;
        int grid_w = 16;
        int input_h = 512;
        int input_w = 512;
        int num_object = 3;
        int classes = 1;
        float conf_thresh = 0.45;
        float class_thresh = 0.5;
        float nms_thresh = 0.3;

	};
	class ApplePerception {
	public:
		ApplePerception(const segmentor_config &cfg);
		~ApplePerception();
		void operator() (const cv::Mat &src_img, std::vector<bbox> &pedestrian_boxes, int handLoc);

        segmentor_config get_segmentor_config();
	private:
		std::unique_ptr<tflite::Interpreter> interpreter_;
		std::unique_ptr<tflite::FlatBufferModel> model_;
		TfLiteDelegate* delegate_;
		segmentor_config cfg_;
		float* cam_paras_;
		mutable std::mutex operator_mutex_;

		bool is_large_fov_;
		void pedestrian_yolo_parse(const std::vector<TfLiteTensor*> outs, std::vector<bbox> &pedestrian_boxes);
		void run_quantization(const cv::Mat &src_img, std::vector<bbox> &pedestrian_boxes, int handLoc);
    	bool large_fov_decider();
	};
} }
#endif