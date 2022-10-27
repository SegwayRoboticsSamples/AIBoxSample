package com.segway.robot.sample.aibox;

import java.nio.ByteBuffer;

public class VisionNative {
    public static native DetectedResult[] nativeDetect(ByteBuffer data, int format, int width, int height);
}
