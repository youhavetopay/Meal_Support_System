package com.example.androidopencv;
import org.opencv.core.Rect;
public class Spoon {

    private static Spoon spoon;
    private int targetId = 0;
    private String targetName = null;
    private double score = 0.0;
    private Rect box;

    private Spoon(){};

    public static Spoon getInstance(){
        if (spoon == null){
            spoon = new Spoon();

        }
        return spoon;

    }

    public int getTargetId() {
        return targetId;
    }

    public void setTargetId(int targetId) {
        this.targetId = targetId;
    }

    public String getTargetName() {
        return targetName;
    }

    public void setTargetName(String targetName) {
        this.targetName = targetName;
    }

    public Rect getBox() {
        return box;
    }

    public void setBox(Rect box) {
        this.box = box;
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }
}
