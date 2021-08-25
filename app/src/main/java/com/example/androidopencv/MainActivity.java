package com.example.androidopencv;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.speech.tts.TextToSpeech;
import static android.speech.tts.TextToSpeech.ERROR;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.dnn.Dnn;
import org.opencv.utils.Converters;
import org.opencv.core.Rect;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * https://github.com/ivangrov/Android-Deep-Learning-with-OpenCV
 *  여기 유튜브 참고함
 * 
 * 
 * **/
public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {


    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;

    private TextToSpeech tts;

    boolean startYOLO = false;
    boolean firstTimeYolo = false;

    Net tinyYolo;

    // assets 폴더에서 해당 파일 가져오기
    // https://m.blog.naver.com/bdg9412/221804437201
    public String getPath(String file, Context context){
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        try{
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();

            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();

            return outFile.getAbsolutePath();

        } catch (IOException ex){
            ex.printStackTrace();
        }
        return "";
    }

    public void YOLO(View Button){
        if (!startYOLO){
            startYOLO = true;

            if (!firstTimeYolo){

                firstTimeYolo = true;
                // cfg이랑 weights 파일 경로 넣기   예제에서는 휴대폰 폴더에 넣고 경로 가져옴 Enviroment ..
                String tinyYoloCfg = getPath("best.cfg", this);
                String tinyYoloWeights = getPath("best_best.weights", this);
                tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);
            }

        } else {
            startYOLO = false;
        }
    }
    
    // 두 객체의 IOU 계산
    // https://gaussian37.github.io/math-algorithm-iou/
    public double getIoU(Rect rect1, Rect rect2){

        double ret = 0.0;

        int rect1Area = rect1.width * rect1.height;
        int rect2Area = rect2.width * rect2.height;

        int intersection_x_length = Math.min(rect1.getMaxX(), rect2.getMaxX()) - Math.max(rect1.x, rect2.x);
        int intersection_y_length = Math.min(rect1.getMaxY(), rect2.getMaxY()) - Math.max(rect1.y, rect2.y);

        if(intersection_x_length > 0 && intersection_y_length > 0){
            int intersectionArea = intersection_x_length * intersection_y_length;
            int union_area = rect1Area + rect2Area - intersectionArea;
            
            ret = (double)intersectionArea / (double)union_area;
        }

        return ret;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        cameraBridgeViewBase = (JavaCameraView) findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        Button yoloBtn = findViewById(R.id.yoloBtn);

        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != ERROR){
                    tts.setLanguage(Locale.KOREAN);
                }
            }
        });

        yoloBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                YOLO(v);
            }
        });

        // System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                if (status == BaseLoaderCallback.SUCCESS) {
                    cameraBridgeViewBase.enableView();
                } else {
                    super.onManagerConnected(status);
                }
            }
        };

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat frame = inputFrame.rgba();
        if (startYOLO){

            Spoon spoon = Spoon.getInstance();

            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

            Mat imageBlob = Dnn.blobFromImage(frame, 0.00392, new Size(416, 416), new Scalar(0,0,0), false, false);

            tinyYolo.setInput(imageBlob);

            java.util.List<Mat> result = new java.util.ArrayList<Mat>(2);

            List<String> outBlobNames = new java.util.ArrayList<>();
            outBlobNames.add(0, "yolo_16");
            outBlobNames.add(1, "yolo_23");

            tinyYolo.forward(result, outBlobNames);

            float confiThreshold = 0.3f;
            List<Integer> clsIds = new ArrayList<>();
            List<Float> confs = new ArrayList<>();
            List<Rect> rects = new ArrayList<>();

            List<String> cocoNames = Arrays.asList("rice", "spoon");

            for(int i=0; i < result.size(); i++){

                Mat level = result.get(i);
                for(int j =0; j <level.rows(); j++){
                    Mat row = level.row(j);

                    Mat scores = row.colRange(5, level.cols());
                    Core.MinMaxLocResult mm = Core.minMaxLoc(scores);

                    float confidence = (float)mm.maxVal;

                    Point classesIdPoint = mm.maxLoc;

                    if(confidence > confiThreshold){
                        int centerX = (int)(row.get(0, 0)[0]*frame.cols());
                        int centerY = (int)(row.get(0, 1)[0]*frame.rows());
                        int width = (int)(row.get(0, 2)[0]*frame.cols());
                        int height = (int)(row.get(0,3)[0]*frame.rows());

                        int left = centerX - width/2;
                        int top = centerY - height/2;


                        clsIds.add((int)classesIdPoint.x);
                        confs.add((float)confidence);
                        rects.add(new Rect(left, top, width, height));

                    }
                }
            }

            int ArrayLength = confs.size();

            if (ArrayLength >= 1){
                float nmsThread = 0.3f;

                MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));

                Rect[] boxesArray = rects.toArray(new Rect[0]);

                MatOfRect boxes = new MatOfRect(boxesArray);
                MatOfInt indices = new MatOfInt();

                Dnn.NMSBoxes(boxes, confidences, confiThreshold, nmsThread, indices);

                int[] ind = indices.toArray();
                for(int i=0; i<ind.length; i++){
                    int idx = ind[i];
                    Rect box = boxesArray[idx];

                    int idGuy = clsIds.get(idx);

                    //float conf = confs.get(idx);



                    //int intConf = (int)conf * 100;

                    if (idGuy == 1){
                        spoon.setBox(box);
                    } else if (spoon.getBox() != null){
                        double score = getIoU(box, spoon.getBox());

                        if(score >= 0.1){

                            // 현재 IoU가 더 높으면 현재껄로 설정하기
                           if(score > spoon.getScore()){
                               spoon.setTargetId(idGuy);
                                spoon.setTargetName(cocoNames.get(idGuy));
                                spoon.setScore(score);
                                Log.d("New Target Name", spoon.getTargetName());

                            }
                        }
                    }

                    Imgproc.putText(frame,cocoNames.get(idGuy),box.tl(),Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255,255,0),2);
                    Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 0, 0), 2);
                }

                if (spoon.getScore() >= 0.1){
                    //tts기능
//                    tts.speak(cocoNames.get(spoon.getTargetId()), TextToSpeech.QUEUE_ADD, null);
                    /**
                     *  그냥 여기서 TTS나 Toast메시지 띄우면 터짐
                     *  이유는 쓰레드 안에서 쓰레드 만들어서??
                     *   그래서 핸들러 만들어서 해야 함
                     * **/

                    Handler mHandler = new Handler(Looper.getMainLooper());
                    mHandler.postDelayed(new Runnable() {
                        @Override
                        public void run() {
                            Log.d("TTS", "start TTS");
                            Toast.makeText(MainActivity.this, ""+spoon.getTargetName()+" 을 감지함", Toast.LENGTH_SHORT).show();
                            spoon.setScore(0.0);
                            spoon.setTargetName(null);
                        }
                    }, 0);

                }

            }
        }

        return frame;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

//        if(startYOLO){
//            // cfg이랑 weights 파일 경로 넣기   예제에서는 휴대폰 폴더에 넣고 경로 가져옴 Environment ..
//            String tinyYoloCfg = getPath("best.cfg", this);
//            String tinyYoloWeights = getPath("best.weights", this);
//            tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);
//        }

    }

    @Override
    public void onCameraViewStopped() {

    }



    @Override
    protected void onResume() {
        super.onResume();
        if(!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(), "There`s a problem", Toast.LENGTH_SHORT).show();
        } else{
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }
    }
    @Override
    protected void onPause() {
        super.onPause();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }
}