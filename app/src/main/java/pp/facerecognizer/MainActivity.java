/*
* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package pp.facerecognizer;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.ClipData;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.media.ImageReader.OnImageAvailableListener;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.Toast;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import androidx.appcompat.app.AlertDialog;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.DialogFragment;

import app.akexorcist.bluetotohspp.library.BluetoothState;
import pp.facerecognizer.env.BorderedText;
import pp.facerecognizer.env.FileUtils;
import pp.facerecognizer.env.ImageUtils;
import pp.facerecognizer.env.Logger;
import pp.facerecognizer.ml.BlazeFace;
import pp.facerecognizer.tracking.MultiBoxTracker;

/**
* An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
* objects.
*/
public class MainActivity extends CameraActivity implements OnImageAvailableListener, SingleChoiceDialogFragment.SingleChoiceListener {
    private static final Logger LOGGER = new Logger();

    private static final int CROP_HEIGHT = BlazeFace.INPUT_SIZE_HEIGHT;
    private static final int CROP_WIDTH = BlazeFace.INPUT_SIZE_WIDTH;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    private Integer sensorOrientation;

    String face_info = "";
    String distanceMode = "";

    private Recognizer recognizer;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private byte[] luminanceCopy;

    private BorderedText borderedText;

    private Snackbar initSnackbar;
    private Snackbar trainSnackbar;
    private FloatingActionButton button;
    private Button compositionModeSelect;

    private boolean initialized = false;
    private boolean training = false;

    // Device Screen Size info
    private int Device_height;
    private int Device_width;

    private int fin_X;
    private int faces_numb;
    int[] fh = new int[5];

    private int compositionMode = 0;

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        DisplayMetrics dm = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(dm);

        Device_width = dm.widthPixels;
        Device_height = dm.heightPixels;

        compositionModeSelect = findViewById(R.id.modeSelect);

        FrameLayout container = findViewById(R.id.container);
        initSnackbar = Snackbar.make(
                container, getString(R.string.initializing), Snackbar.LENGTH_INDEFINITE);
        trainSnackbar = Snackbar.make(
                container, getString(R.string.training), Snackbar.LENGTH_INDEFINITE);

        View dialogView = getLayoutInflater().inflate(R.layout.dialog_edittext, null);
        EditText editText = dialogView.findViewById(R.id.edit_text);
        AlertDialog editDialog = new AlertDialog.Builder(MainActivity.this)
                .setTitle(R.string.enter_name)
                .setView(dialogView)
                .setPositiveButton(getString(R.string.ok), (dialogInterface, i) -> {
                    int idx = recognizer.addPerson(editText.getText().toString());
                    performFileSearch(idx - 1);
                })
                .create();

        button = findViewById(R.id.add_button);
        button.setOnClickListener(view ->
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle(getString(R.string.select_name))
                        .setItems(recognizer.getClassNames(), (dialogInterface, i) -> {
                            if (i == 0) {
                                editDialog.show();
                            } else {
                                performFileSearch(i - 1);
                            }
                        })
                        .show());

        /* 어플 모드 선택 (일반 모드, 중앙 구도 모드, 3분할 왼쪽 모드, 3분할 오른쪽 모드) */
        compositionModeSelect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                DialogFragment singleChoiceDialog = new SingleChoiceDialogFragment();
                singleChoiceDialog.setCancelable(false);
                singleChoiceDialog.show(getSupportFragmentManager(), "Single Choice Dialog");
            }
        });

        Drawable on = ContextCompat.getDrawable(this, R.drawable.button);
        Drawable off = ContextCompat.getDrawable(this, R.drawable.list_item_background);
        Drawable activated = ContextCompat.getDrawable(this, R.drawable.list_item_background_square);
        Drawable transParent = ContextCompat.getDrawable(this, R.color.transparent);

        final Runnable runCompositionMode = new Runnable() {
            @SuppressLint("ResourceAsColor")
            @Override
            public void run() {
//                final View handsFreeCapture = findViewById(R.id.camera_btn_tf);
                if (faces_numb >= 1) {
                    if (compositionMode == 1) {
                        if (fin_X > 450 && fin_X < 550) {
                            compositionModeSelect.setBackground(on);
                            container.setBackground(activated);
//                            if (safeToTakePicture) {        //플래그 검사 및 카메라 캡쳐 기능
//                                handsFreeCapture.performClick();
//                                safeToTakePicture = false;
//                            }
                        } else {
                            compositionModeSelect.setBackground(off);
                            container.setBackground(transParent);
                        }
                    } else if (compositionMode == 2) {
                        if (fin_X > 283 && fin_X < 384) {
                            compositionModeSelect.setBackground(on);
                            container.setBackground(activated);
//                            if (safeToTakePicture) {        //플래그 검사 및 카메라 캡쳐 기능
//                                handsFreeCapture.performClick();
//                                safeToTakePicture = false;
//                            }
                        } else {
                            compositionModeSelect.setBackground(off);
                            container.setBackground(transParent);
                        }
                    } else if (compositionMode == 3) {
                        if (fin_X > 616 && fin_X < 717) {
                            compositionModeSelect.setBackground(on);
                            container.setBackground(activated);
//                            if (safeToTakePicture) {        //플래그 검사 및 카메라 캡쳐 기능
//                                handsFreeCapture.performClick();
//                                safeToTakePicture = false;
//                            }
                        } else {
                            compositionModeSelect.setBackground(off);
                            container.setBackground(transParent);
                        }
                    } else if (compositionMode == 0) {
                        compositionModeSelect.setBackground(off);
                        container.setBackground(transParent);
                    }
                }
            }
        };

        class NewRunnable implements Runnable {
            @Override
            public void run() {
                while (true) {

                    try {
//                        safeToTakePicture = true;
                        Thread.sleep(5000);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    // 메인 스레드에 runnable 전달.
                    runOnUiThread(runCompositionMode);
                }
            }
        }

        NewRunnable nr = new NewRunnable();
        Thread t = new Thread(nr);
        t.start();
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        if (!initialized)
            init();

        final float textSizePx =
        TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(CROP_WIDTH, CROP_HEIGHT, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        CROP_WIDTH, CROP_HEIGHT,
                        sensorOrientation, false);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);
        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> {
                    String mode = "";
                    tracker.draw(canvas, getCameraFacing());
                    if (isDebug()) {
                        tracker.drawDebug(canvas);
                    }

                    fin_X = (int) ((tracker.getFinX() / Device_width)*3100);
                    int fin_Y = (int) ((tracker.getFinY() / Device_height)*4000);
                    if(fin_X > 1000) fin_X = 1000;
                    if(fin_Y > 1000) fin_Y = 1000;
                    if(fin_X < 0) fin_X = 0;
                    if(fin_Y < 0) fin_Y = 0;

                    if(compositionMode == 0){
                        mode = "N";
                    }else if(compositionMode == 1){
                        mode = "C";
                    }else if(compositionMode == 2){
                        mode = "L";
                    }else if(compositionMode == 3){
                        mode = "R";
                    }

                    if (tracker.getFaces_numb() > 0) {
                        fh = tracker.get_fh();
                    }

                    faces_numb = tracker.getFaces_numb();
                    if (tracker.getFaces_numb() > 0){
                        face_info = mode + "!" + fin_X + "/" + fin_Y;
                        face_info += ":" + tracker.getDistanceMode();
                    }else{
                        face_info = "E";
                        // Empty
                    }
                    LOGGER.i("Face Info : " + face_info);

                    /** 20.04.20 Continuous Bluetooth send Code -> Success! */
                    if (bt.isServiceAvailable()) {
                        bt.send(face_info, true);
                    } else {
                        bt.setupService();
                        bt.startService(BluetoothState.DEVICE_OTHER);
                        bt.send(face_info, true);
                    }
                });


        addCallback(
                canvas -> {
                    if (!isDebug()) {
                        return;
                    }
                    final Bitmap copy = cropCopyBitmap;
                    if (copy == null) {
                        return;
                    }

                    final int backgroundColor = Color.argb(100, 0, 0, 0);
                    canvas.drawColor(backgroundColor);

                    final Matrix matrix = new Matrix();
                    final float scaleFactor = 2;
                    matrix.postScale(scaleFactor, scaleFactor);
                    matrix.postTranslate(
                            canvas.getWidth() - copy.getWidth() * scaleFactor,
                            canvas.getHeight() - copy.getHeight() * scaleFactor);
                    canvas.drawBitmap(copy, matrix, new Paint());

                    final Vector<String> lines = new Vector<>();
                    lines.add("Frame: " + previewWidth + "x" + previewHeight);
                    lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
                    lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
                    lines.add("Rotation: " + sensorOrientation);
                    lines.add("Inference time: " + lastProcessingTimeMs + "ms");

                    borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
                });
    }

    OverlayView trackingOverlay;

    void init() {
        runInBackground(() -> {
            runOnUiThread(()-> initSnackbar.show());
            File dir = new File(FileUtils.ROOT);

            if (!dir.isDirectory()) {
                if (dir.exists()) dir.delete();
                dir.mkdirs();

                AssetManager mgr = getAssets();
                FileUtils.copyAsset(mgr, FileUtils.DATA_FILE);
                FileUtils.copyAsset(mgr, FileUtils.MODEL_FILE);
                FileUtils.copyAsset(mgr, FileUtils.LABEL_FILE);
            }

            try {
                recognizer = Recognizer.getInstance(getAssets());
            } catch (Exception e) {
                LOGGER.e("Exception initializing classifier!", e);
                finish();
            }

            runOnUiThread(()-> initSnackbar.dismiss());
            initialized = true;
        });
    }

    //  최적 구도 Dialog 창에서 모드 선택시 실행되는 메소드
    @Override
    public void onPositiveButtonClicked(String[] list, int position) {
        if(position == 0){
            compositionModeSelect.setText("N");
            compositionMode = 0;
        }else if(position == 1){
            compositionModeSelect.setText("C");
            compositionMode = 1;
        }else if(position == 2){
            compositionModeSelect.setText("L");
            compositionMode = 2;
        }else{
            compositionModeSelect.setText("R");
            compositionMode = 3;
        }
    }

    @Override
    public void onNegativeButtonClicked() {

    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        byte[] originalLuminance = getLuminance();
        tracker.onFrame(
                previewWidth,
                previewHeight,
                getLuminanceStride(),
                sensorOrientation,
                originalLuminance,
                timestamp);
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection || !initialized || training) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        if (luminanceCopy == null) {
            luminanceCopy = new byte[originalLuminance.length];
        }
        System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                () -> {
                    LOGGER.i("Running detection on image " + currTimestamp);
                    final long startTime = SystemClock.uptimeMillis();

                    cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                    List<Recognizer.Recognition> mappedRecognitions =
                            recognizer.recognizeImage(croppedBitmap,cropToFrameTransform);

                    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                    tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                    trackingOverlay.postInvalidate();

                    requestRender();
                    computingDetection = false;
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (!initialized) {
            Snackbar.make(
                    getWindow().getDecorView().findViewById(R.id.container),
                    getString(R.string.try_it_later), Snackbar.LENGTH_SHORT)
                    .show();
            return;
        }

        if (resultCode == RESULT_OK) {
            trainSnackbar.show();
            button.setEnabled(false);
            training = true;

            ClipData clipData = data.getClipData();
            ArrayList<Uri> uris = new ArrayList<>();

            if (clipData == null) {
                uris.add(data.getData());
            } else {
                for (int i = 0; i < clipData.getItemCount(); i++)
                    uris.add(clipData.getItemAt(i).getUri());
            }

            new Thread(() -> {
                try {
                    recognizer.updateData(requestCode, getContentResolver(), uris);
                } catch (Exception e) {
                    LOGGER.e(e, "Exception!");
                } finally {
                    training = false;
                }
                runOnUiThread(() -> {
                    trainSnackbar.dismiss();
                    button.setEnabled(true);
                });
            }).start();

        }

        //////////////////////////////// Bluetooth Permission Result /////////////////////////////////
        if (requestCode == BluetoothState.REQUEST_CONNECT_DEVICE) {
            if (resultCode == Activity.RESULT_OK)
                bt.connect(data);
        } else if (requestCode == BluetoothState.REQUEST_ENABLE_BT) {
            if (resultCode == Activity.RESULT_OK) {
                Toast.makeText(getApplicationContext()
                        , "Bluetooth is enabled!!"
                        , Toast.LENGTH_SHORT).show();

//                bt.setupService();
//                bt.startService(BluetoothState.DEVICE_OTHER);
//                setup();
            } else {
                // Do something if user doesn't choose any device (Pressed back)
                Toast.makeText(getApplicationContext()
                        , "Bluetooth was not enabled."
                        , Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    public void performFileSearch(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        intent.setType("image/*");

        startActivityForResult(intent, requestCode);
    }
}
