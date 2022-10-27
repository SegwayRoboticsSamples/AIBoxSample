package com.segway.robot.sample.aibox;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.widget.ImageView;

import java.util.ArrayList;
import java.util.List;

public class VisionImageView extends ImageView {

    private static final int LINE_SIZE = 20;
    private List<RectF> mRectList = new ArrayList<>();
    private Paint mPaint;

    public VisionImageView(Context context) {
        super(context);
    }

    public VisionImageView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public VisionImageView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public VisionImageView(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public void mark(List<RectF> list) {
        mRectList.clear();
        mRectList.addAll(list);
        invalidate();
    }

    @Override
    public void draw(Canvas canvas) {
        super.draw(canvas);
        canvas.save();
        if(mPaint == null) {
            mPaint = new Paint();
            mPaint.setAntiAlias(true);
            mPaint.setColor(Color.GREEN);
            mPaint.setTextSize(LINE_SIZE);
            mPaint.setStyle(Paint.Style.STROKE);
        }
        for(RectF rectF : mRectList) {
            canvas.drawRect(rectF, mPaint);
        }
        canvas.restore();
    }
}
