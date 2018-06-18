package com.doxua.www.facer;

import android.content.Intent;
import android.graphics.Bitmap;
import org.bytedeco.javacpp.opencv_core.Point;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Rect;

import static org.bytedeco.javacpp.opencv_core.Mat;

import java.io.IOException;

import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.opencv.core.Core.LINE_8;




public class TrainFaces extends AppCompatActivity {

    private static final int PICK_IMAGE = 100;

    private ImageView imageView;
    private TextView textView;
    private opencv_objdetect.CascadeClassifier faceDetector;
    private int absoluteFaceSize = 0;


    Rect rectCrop = null;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_trainfaces);

        // Create the image view and text view.
        imageView = (ImageView) findViewById(R.id.imageView);
        textView = (TextView) findViewById(R.id.faces_value);

        // Pick an image and recognize
        Button pickImageButton = (Button) findViewById(R.id.btnGallery);
        pickImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openGallery();
            }
        });
    }

    private void openGallery() {
        Intent gallery =
                new Intent(Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        startActivityForResult(gallery, PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == PICK_IMAGE) {
            Uri imageUri = data.getData();

            // Convert to Bitmap.
            Bitmap bitmap = null;
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
            } catch (IOException e) {
                e.printStackTrace();
            }


            // Detect faces...
            // Display number of faces detected.
            // Draw a rectangle around the first face detected.
            detectAndDisplay(bitmap, textView);

        }
    }

    void detectAndDisplay(Bitmap bitmap, TextView facesValue) {


        opencv_core.Mat greyMat = new opencv_core.Mat();

        AndroidFrameConverter converterToBitmap = new AndroidFrameConverter();
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();

        // Convert to Bitmap.
        Frame frame = converterToBitmap.convert(bitmap);
        // Convert to Mat.
        opencv_core.Mat mat = converterToMat.convert(frame);

        // Load the CascadeClassifier class to detect objects.
        faceDetector = TrainHelper.loadClassifierCascade(TrainFaces.this, R.raw.frontalface);

        // Convert to Gray scale.
        cvtColor(mat, greyMat, CV_BGR2GRAY);
        // Vector of rectangles where each rectangle contains the detected object.
        opencv_core.RectVector faces = new opencv_core.RectVector();

        // Detect the face.
        faceDetector.detectMultiScale(greyMat, faces, 1.25f, 3, 1,
                new opencv_core.Size(absoluteFaceSize, absoluteFaceSize),
                new opencv_core.Size(4 * absoluteFaceSize, 4 * absoluteFaceSize));


        // Count number of faces and display in text view.
        int numFaces = (int) faces.size();
        facesValue.setText(Integer.toString(numFaces));

        // Display the detected face.
        int x = faces.get(0).x();
        int y = faces.get(0).y();
        int w = faces.get(0).width();
        int h = faces.get(0).height();

        // Crop the detected face.
        rectCrop = new Rect(x, y, w, h);

        // Convert the original image to dropped image.
        Mat croppedImage = new Mat(mat, rectCrop);

        // Convert processedMat back to a Frame
        frame = converterToMat.convert(croppedImage);

        // Copy the data to a Bitmap for display or something
        Bitmap bm = converterToBitmap.convert(frame);

        // Display the picked image.
        imageView.setImageBitmap(bm);

    }
}
