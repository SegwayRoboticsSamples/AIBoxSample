//
// Created by zxh on 10/17/22.
//

#ifndef VISIONSERVICE_VISIONNATIVE_H
#define VISIONSERVICE_VISIONNATIVE_H

#include <jni.h>
#include "./include/ApplePerception.h"
#include <list>

void rgba2bgr(cv::Mat &frame, char *data, jint width, jint height);
void yuv2bgr(cv::Mat &frame, char *data, jint width, jint height);
JNIEXPORT jobjectArray JNICALL jni_detect(JNIEnv *env, jclass obj, jobject data, jint format, jint width, jint height);

#endif //VISIONSERVICE_VISIONNATIVE_H
