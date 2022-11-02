#include "VisionNative.h"
#include <jni.h>
#include <android/log.h>
#include "AlgoApplePerception.h"

using namespace ninebot_algo;
using namespace cnn_ninebot;

#define ROBOT_LOG_TAG        "aibox_native"

#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, ROBOT_LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, ROBOT_LOG_TAG, __VA_ARGS__)

#define RGBA8888  1
#define YUV420  5

JavaVM *javaVM;

typedef struct {
    jclass clazz;
    jmethodID DetectedResult_id;
} com_segway_robot_sample_aibox_DetectedResult;

jclass mVisionNativeClazz;
com_segway_robot_sample_aibox_DetectedResult mDetectedResult;

static JNINativeMethod methodTable[] = {
        {"nativeDetect", "(Ljava/nio/ByteBuffer;III)[Lcom/segway/robot/sample/aibox/DetectedResult;", (void *) jni_detect},
};

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    JNIEnv *env;
    javaVM = vm;
    if ((vm)->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR; // JNI version not supported.
    }

    mVisionNativeClazz = (jclass) env->NewGlobalRef(
            env->FindClass("com/segway/robot/sample/aibox/VisionNative"));


    mDetectedResult.clazz = (jclass) env->NewGlobalRef(
            env->FindClass("com/segway/robot/sample/aibox/DetectedResult"));
    mDetectedResult.DetectedResult_id = env->GetMethodID(mDetectedResult.clazz, "<init>",
                                                         "(IFFFFF)V");

    env->RegisterNatives(mVisionNativeClazz, methodTable,
                         sizeof(methodTable) / sizeof(methodTable[0]));
    return JNI_VERSION_1_6;
}

void yuv2bgr(cv::Mat &frame, char *data, jint width, jint height) {
    cv::Mat srcFrame(cv::Size(width, height + height / 2), CV_8UC1, data, cv::Mat::AUTO_STEP);
    cv::cvtColor(srcFrame, frame, CV_YUV2BGR_NV12);
}

void rgba2bgr(cv::Mat &frame, char *data, jint width, jint height) {
    //cv::Mat srcFrame(cv::Size(width, height), CV_8UC4, data, cv::Mat::AUTO_STEP);
    //cv::cvtColor(srcFrame, frame, CV_RGBA2BGR);
    cv::Mat srcFrame(cv::Size(width, height), CV_8UC4, data, cv::Mat::AUTO_STEP);
    cv::Mat resizeFrame;
    cv::cvtColor(srcFrame, resizeFrame, CV_RGBA2BGR);
    cv::resize(resizeFrame, frame, cv::Size(1280, 720));
    cv::imwrite("/sdcard/apple1.jpeg", frame);
    LOGD("detect, frame.cols %d, frame.rows %d", frame.cols, frame.rows);
}

AlgoApplePerception *algoApplePerception = nullptr;

JNIEXPORT jobjectArray JNICALL
jni_detect(JNIEnv *env, jclass obj, jobject data, jint format, jint width, jint height) {
    LOGD("width: %d, height: %d, format %d", width, height, format);
    char *imageData = (char *) env->GetDirectBufferAddress(data);
    cv::Mat frame;

    switch (format) {
        case RGBA8888:
            rgba2bgr(frame, imageData, width, height);
            LOGD("detect image");
            break;
        case YUV420:
            yuv2bgr(frame, imageData, width, height);
            LOGD("detect video");
            break;
        default:
            return nullptr;
    }

    //调用算法
    if (algoApplePerception == nullptr) {
        algoApplePerception = new AlgoApplePerception();
    }
    std::vector<bbox> appleDetectResult = algoApplePerception->PerceptionProcess(frame);
    LOGD("appleDetectResult size is %d", appleDetectResult.size());

    jobjectArray objArray = env->NewObjectArray(appleDetectResult.size(), mDetectedResult.clazz, nullptr);

    for(int i=0; i<appleDetectResult.size(); i++) {
        bbox box = appleDetectResult[i];
        jobject detectedResultObj = env->NewObject(mDetectedResult.clazz,
                                                       mDetectedResult.DetectedResult_id, box.classId, box.x1, box.y1,
                                                       box.x2, box.y2, box.score);
        env->SetObjectArrayElement(objArray, i, detectedResultObj);

        LOGD("appleDetectResult box.classId %d, x1 %f, x2 %f, y1 %f, y2 %f,score %f", box.classId, box.x1, box.y1, box.x2, box.y2, box.score);
    }
    return objArray;

}