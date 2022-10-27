#include "VisionNative.h"
#include <jni.h>
#include <android/log.h>
#include "AlgoApplePerception.h"


#define ROBOT_LOG_TAG        "aibox_native"

#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, ROBOT_LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, ROBOT_LOG_TAG, __VA_ARGS__)

#define RGBA8888  1
#define YUV420  7

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

void rgba2bgr(cv::Mat &frame, char *data, jint width, jint height) {
    cv::Mat srcFrame(cv::Size(width, height + height / 2), CV_8UC1, data, cv::Mat::AUTO_STEP);
    cv::cvtColor(srcFrame, frame, CV_YUV2BGR_NV12);
}

void yuv2bgr(cv::Mat &frame, char *data, jint width, jint height) {
    cv::Mat srcFrame(cv::Size(width, height), CV_8UC4, data, cv::Mat::AUTO_STEP);
    cv::cvtColor(srcFrame, frame, CV_RGBA2BGR);
}

JNIEXPORT jobjectArray JNICALL
jni_detect(JNIEnv *env, jclass obj, jobject data, jint format, jint width, jint height) {
    LOGD("width: %d, height: %d", width, height);
    char *imageData = (char *) env->GetDirectBufferAddress(data);
    cv::Mat frame;
    switch (format) {
        case RGBA8888:
            rgba2bgr(frame, imageData, width, height);
            break;
        case YUV420:
            yuv2bgr(frame, imageData, width, height);
            break;
        default:
            return nullptr;
    }

    //TODO::调用算法

    jobjectArray objArray = env->NewObjectArray(1, mDetectedResult.clazz, nullptr);
    jobject detectedResultObj = env->NewObject(mDetectedResult.clazz,
                                               mDetectedResult.DetectedResult_id, 10, 10.0, 100.0f,
                                               100.0f, 200.0f, 1.0f);
    env->SetObjectArrayElement(objArray, 0, detectedResultObj);
    return objArray;

}