package com.doxua.www.facer;

import android.content.Intent;
import android.graphics.Bitmap;
import org.bytedeco.javacpp.opencv_core.Point;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
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

import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import org.bytedeco.javacpp.opencv_core.Rect;

import static org.bytedeco.javacpp.opencv_core.Mat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Random;

import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.opencv.core.Core.LINE_8;




public class TrainFaces extends AppCompatActivity {

    private static final int PICK_IMAGE = 100;

    private ImageView imageView;
    private TextView textView;
    private opencv_objdetect.CascadeClassifier faceDetector;
    private int absoluteFaceSize = 0;


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



    /**
     * Introducing JavaCV frame converters.
     * http://bytedeco.org/news/2015/04/04/javacv-frame-converters/
     * @param bitmap
     * @param facesValue
     */
    void detectAndDisplay(Bitmap bitmap, TextView facesValue) {

        // Initializing.
        opencv_core.Mat greyMat = new opencv_core.Mat();
        // JavaCV frame converters.
        AndroidFrameConverter converterToBitmap = new AndroidFrameConverter();
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();

        // -------------------------------------------------------------------
        //                  CONVERT BACK TO MAT FOR PROCESSING
        // -------------------------------------------------------------------
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

        // -------------------------------------------------------------------
        //                      STORE THE TRAINED PHOTOS
        // -------------------------------------------------------------------
        if ( numFaces > 0 ) {
            // Multiple face detection.
            for (int i = 0; i < numFaces; i++) {

                // Display the detected face.
                int x1 = faces.get(i).x();
                int y1 = faces.get(i).y();
                int w1 = faces.get(i).width();
                int h1 = faces.get(i).height();



                // Crop the detected face.
                Rect rectCrop = new Rect(x1, y1, w1, h1);

                // Convert the original image to dropped image.
                Mat croppedImage = new Mat(mat, rectCrop);

                // Important: Needed or the image will come out blurring.
                resize(croppedImage, croppedImage, new opencv_core.Size(TrainHelper.IMG_SIZE,TrainHelper.IMG_SIZE));



                // -------------------------------------------------------------------
                //              CONVERT BACK TO BITMAP FOR DISPLAYING
                // -------------------------------------------------------------------
                // Convert processedMat back to a Frame
                frame = converterToMat.convert(croppedImage);

                // Copy the data to a Bitmap for display or something
                Bitmap bm1 = converterToBitmap.convert(frame);

                // Store image.
                storeImage(bm1);
            }
        }



        // -------------------------------------------------------------------
        //                               DISPLAY
        // -------------------------------------------------------------------
        if ( numFaces > 0 ) {
            // Multiple face detection.
            for (int i = 0; i < numFaces; i++) {

                int x = faces.get(i).x();
                int y = faces.get(i).y();
                int w = faces.get(i).width();
                int h = faces.get(i).height();

                rectangle(mat, new Point(x, y), new Point(x + w, y + h), opencv_core.Scalar.GREEN, 2, LINE_8, 0);

                // -------------------------------------------------------------------
                //              CONVERT BACK TO BITMAP FOR DISPLAYING
                // -------------------------------------------------------------------
                // Convert processedMat back to a Frame
                frame = converterToMat.convert(mat);

                // Copy the data to a Bitmap for display or something
                Bitmap bm = converterToBitmap.convert(frame);

                // Display the picked image.
                imageView.setImageBitmap(bm);
            }
        } else {
            imageView.setImageBitmap(bitmap);
        }
    }


    void storeImage(Bitmap bitmap) {

        String root = Environment.getExternalStorageDirectory().toString();
        File myDir = new File(root + "/saved_images");
        myDir.mkdirs();
        Random generator = new Random();
        int n = 10000;
        n = generator.nextInt(n);
        String fname = "Image-"+ n +".jpg";
        File file = new File (myDir, fname);
        if (file.exists ()) file.delete ();
        try {
            FileOutputStream out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out);
            out.flush();
            out.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
