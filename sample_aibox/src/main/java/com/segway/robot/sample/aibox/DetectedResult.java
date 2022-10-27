package com.segway.robot.sample.aibox;

public class DetectedResult {
    public DetectedResult(int id, float x1, float y1, float x2, float y2, float score) {
        this.id = id;
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.score = score;
    }

    public int id;
    public float x1;
    public float y1;
    public float x2;
    public float y2;
    public float score;
}
