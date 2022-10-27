cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_C_FLAGS "-O3 -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS "-std=c++11 -fexceptions -frtti -Wattributes -Werror=return-type -fvisibility=hidden")

#####################################
#opencv dir  && ref opencv libs
#####################################
set(OpenCV_DIR "${PROJECT_SOURCE_DIR}/dependency/3rdparty_android/opencv-3.4.0_shared_arm/share/OpenCV/")
message("the CPU ABI is ${ANDROID_ABI}, and OpenCV is at ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)

###########################################
# for package external libs into java apk #
###########################################
macro(package_lib lib_path)
    set(PACK_LIB_LIST ${PACK_LIB_LIST} ${lib_path})
endmacro()

##3rd party lib DIR...
set(ALGO_3RD_DIR "${PROJECT_SOURCE_DIR}/dependency/3rdparty_android")
message(" 3rdparty path is ${ALGO_3RD_DIR}")

#####################################
#using tensorflow lite gpu delegate
#####################################
package_lib(${ALGO_3RD_DIR}/tensorflow_lite_gpu_delegate/lib/${ANDROID_ABI}/libtensorflowlite.so)
package_lib(${ALGO_3RD_DIR}/tensorflow_lite_gpu_delegate/lib/${ANDROID_ABI}/libtensorflowlite_gpu_delegate.so)

include_directories(src/main/cpp
        src/main/cpp/include
        ${ALGO_3RD_DIR}/tensorflow_lite_gpu_delegate/include
        ${OpenCV_INCLUDE_DIRS}
        ${ANDROID_SYSROOT}/usr/include/aarch64-linux-android)

add_library(SidewalkPerception SHARED IMPORTED)
set_target_properties( # Specifies the target library.
        SidewalkPerception
        # Specifies the parameter you want to define.
        PROPERTIES IMPORTED_LOCATION

        # Provides the path to the library you want to import.
        ${PROJECT_SOURCE_DIR}/lib/${ANDROID_ABI}/libSidewalkPerception.so)

add_library(vision_aibox SHARED
        VisionNative.cpp
        AlgoApplePerception.cpp
        )

target_link_libraries(vision_aibox
        android
        log
        ${OpenCV_LIBS}
        ${PACK_LIB_LIST}
        SidewalkPerception
        )