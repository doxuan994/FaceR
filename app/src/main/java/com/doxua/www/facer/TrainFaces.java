package com.doxua.www.facer;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.Frame;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.IntBuffer;
import java.util.Random;

import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.opencv.core.Core.LINE_8;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.RectVector;
import static org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_core.MatVector;
import static org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import static org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.EigenFaceRecognizer;



/**
 * Train our model of photos.
 */
public class TrainFaces extends AppCompatActivity {

    public static final String TAG = "TrainFaces";
    public static final String TRAIN_FOLDER = "train_folder";
    public static final int IMG_SIZE = 160;
    public static final String EIGEN_FACES_CLASSIFIER = "eigenFacesClassifier.yml";
    public static final int PHOTOS_TRAIN_QTY = 10;
    public static final String FILE_NAME_PATTERN = "person.%d.%d.jpg";

    // Select images from gallery variables.
    private static final int PICK_IMAGE = 100;

    // View variables.
    private ImageView imageView;
    private TextView textView;

    // Load the CascadeClassifier class to detect objects.
    private CascadeClassifier faceDetector;
    private int absoluteFaceSize = 0;

    // Keep track of number of images.
    private int photoNumber = 1;


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

        // Train button.
        Button trainBtn = (Button) findViewById(R.id.btnTrain);
        trainBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    trainExternalStorage();
                } catch (Exception e) {
                    e.printStackTrace();
                }
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

            // Detect faces... Sometimes, hands, boobs, dogs, barbie dolls, etc.
            // Display number of detected faces.
            // Draw a green rectangle around the first detected face.
            detectAndDisplay(bitmap, textView);

            // Store images with the correct format.
            if (photoNumber <= PHOTOS_TRAIN_QTY) {
                try {
                    storeFormat(bitmap, 1, photoNumber);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                photoNumber++;
            }
        }
    }

    /**
     * Introducing JavaCV frame converters.
     * http://bytedeco.org/news/2015/04/04/javacv-frame-converters/
     * @param bitmap
     * @param facesValue
     */
    void detectAndDisplay(Bitmap bitmap, TextView facesValue) {

        // Create a new gray Mat.
        Mat greyMat = new Mat();
        // JavaCV frame converters.
        AndroidFrameConverter converterToBitmap = new AndroidFrameConverter();
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();

        // -------------------------------------------------------------------
        //                  CONVERT TO MAT FOR PROCESSING
        // -------------------------------------------------------------------
        // Convert to Bitmap.
        Frame frame = converterToBitmap.convert(bitmap);
        // Convert to Mat.
        Mat colorMat = converterToMat.convert(frame);

        // Convert to Gray scale.
        cvtColor(colorMat, greyMat, CV_BGR2GRAY);
        // Vector of rectangles where each rectangle contains the detected object.
        RectVector faces = new RectVector();

        // Load the CascadeClassifier class to detect objects.
        faceDetector = loadClassifierCascade(this, R.raw.frontalface);

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
                Mat croppedImage = new Mat(colorMat, rectCrop);

                // Important: Needed or images will come out blurring.
                resize(croppedImage, croppedImage, new Size(IMG_SIZE, IMG_SIZE));

                // -------------------------------------------------------------------
                //              CONVERT BACK TO BITMAP FOR DISPLAYING
                // -------------------------------------------------------------------
                // Convert processed Mat back to a Frame
                frame = converterToMat.convert(croppedImage);
                // Copy the data to a Bitmap for display or something
                Bitmap bm1 = converterToBitmap.convert(frame);

                // Store image.
                String s = "/tom_cruise";
                storeImages(bm1, s);
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

                rectangle(colorMat, new Point(x, y), new Point(x + w, y + h), opencv_core.Scalar.GREEN, 2, LINE_8, 0);

                // -------------------------------------------------------------------
                //              CONVERT BACK TO BITMAP FOR DISPLAYING
                // -------------------------------------------------------------------
                // Convert processed Mat back to a Frame
                frame = converterToMat.convert(colorMat);
                // Copy the data to a Bitmap for display or something
                Bitmap bm = converterToBitmap.convert(frame);

                // Display the picked image.
                imageView.setImageBitmap(bm);
            }
        } else {
            imageView.setImageBitmap(bitmap);
        }

    }

    // --------------------------------------------------------------------------------------------------
    //                                          HELPER METHODS
    // --------------------------------------------------------------------------------------------------

    /**
     * Load the CascadeClassifier for Face Detection.
     * @param context
     * @param resId
     * @return
     */
    public static CascadeClassifier loadClassifierCascade(Context context, int resId) {
        FileOutputStream fos = null;
        InputStream inputStream;

        inputStream = context.getResources().openRawResource(resId);
        File xmlDir = context.getDir("xml", Context.MODE_PRIVATE);
        File cascadeFile = new File(xmlDir, "temp.xml");
        try {
            fos = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                fos.write(buffer, 0, bytesRead);
            }
        } catch (IOException e) {
            Log.d(TAG, "Can\'t load the cascade file");
            e.printStackTrace();
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        CascadeClassifier detector = new CascadeClassifier(cascadeFile.getAbsolutePath());
        if (detector.isNull()) {
            Log.e(TAG, "Failed to load cascade classifier");
            detector = null;
        } else {
            Log.i(TAG, "Loaded cascade classifier from " + cascadeFile.getAbsolutePath());
        }
        // Delete the temporary directory
        cascadeFile.delete();
        return detector;
    }

    /**
     * Store the detected faces captured to the external storage of the phone.
     * The folder of all the captured called "saved_images.
     * @param bitmap
     */
    void storeImages(Bitmap bitmap, String personName) {

        // Find the string root of the phone path.
        String root = Environment.getExternalStorageDirectory().toString();
        // Create a new folder call 'saved_images'.
        File myDir = new File(root + "/saved_images");
        if (!myDir.exists()) {
            myDir.mkdirs();
        }
        Random generator = new Random();
        int n = 10000;
        n = generator.nextInt(n);

        String fname = personName + "." + n + ".jpg";
        File file = new File (myDir, fname);

        if (file.exists ()) {
            file.delete ();
        }
        try {
            FileOutputStream out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out);
            out.flush();
            out.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Save the grayscale format.
     * @param bitmap The image.
     * @param personId
     * @param photoNumber
     */
    void storeFormat(Bitmap bitmap, int personId, int photoNumber) throws Exception {

        // Find the string root of the phone path.
        String root = Environment.getExternalStorageDirectory().toString();
        // Create a new folder call 'myTrainDir'.
        File myTrainDir = new File(root + "/myTrainDir");
        if (!myTrainDir.exists()) {
            myTrainDir.mkdirs();
        }

        // Keep track of saved images.
        int qtyPhotos = PHOTOS_TRAIN_QTY;

        // Create a new gray Mat.
        Mat greyMat = new Mat();
        // JavaCV frame converters.
        AndroidFrameConverter converterToBitmap = new AndroidFrameConverter();
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();

        // -------------------------------------------------------------------
        //                  CONVERT TO MAT FOR PROCESSING
        // -------------------------------------------------------------------
        // Convert to Bitmap.
        Frame frame = converterToBitmap.convert(bitmap);
        // Convert to Mat.
        Mat colorMat = converterToMat.convert(frame);

        // Convert to Gray scale.
        cvtColor(colorMat, greyMat, CV_BGR2GRAY);
        // Vector of rectangles where each rectangle contains the detected object.
        RectVector faces = new RectVector();

        // Load the CascadeClassifier class to detect objects.
        faceDetector = loadClassifierCascade(this, R.raw.frontalface);
        // Detect the face.
        faceDetector.detectMultiScale(greyMat, faces, 1.1, 1, 0, new Size(150,150), new Size(500,500));

        // Count number of faces and display in text view.
        int numFaces = (int) faces.size();

        // Save all the detected faces.
        for (int i = 0; i < numFaces; i++) {
            Rect rectFace = faces.get(i);
            rectangle(colorMat, rectFace, new Scalar(0,0,255, 0));
            Mat capturedFace = new Mat(greyMat, rectFace);
            resize(capturedFace, capturedFace, new Size(IMG_SIZE,IMG_SIZE));

            // Save an image of limit of images not reach.
            if (photoNumber <= qtyPhotos) {
                File f = new File(myTrainDir, String.format(FILE_NAME_PATTERN, personId, photoNumber));
                f.createNewFile();
                imwrite(f.getAbsolutePath(), capturedFace);
            }
        }
    }

    /**
     * Train our model.
     * Important!
     * @return
     * @throws Exception
     */
    boolean trainExternalStorage() throws Exception {
        // Find the string root of the phone path.
        String root = Environment.getExternalStorageDirectory().toString();
        String photosFolderPath = root + "/myTrainDir";
        File photosFolder = new File(photosFolderPath);

        FilenameFilter imageFilter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return  name.endsWith(".jpg") || name.endsWith(".gif") || name.endsWith(".png");
            }
        };

        // Create a list of photo paths.
        File[] files = photosFolder.listFiles(imageFilter);

        MatVector photos = new MatVector(files.length);
        Mat labels = new Mat(files.length, 1, CV_32SC1);
        IntBuffer rotulosBuffer = labels.createBuffer();

        int counter = 0;
        for (File image: files) {
            Mat photo = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int classe = Integer.parseInt(image.getName().split("\\.")[1]);
            resize(photo, photo, new Size(IMG_SIZE, IMG_SIZE));
            photos.put(counter, photo);
            rotulosBuffer.put(counter, classe);
            counter++;
        }

        // Save our model as YAML file to the folder created on the top of the method.
        if (photosFolder.length() > 0) {
            FaceRecognizer eigenfaces = EigenFaceRecognizer.create();
            eigenfaces.train(photos, labels);
            File f = new File(photosFolder, EIGEN_FACES_CLASSIFIER);
            f.createNewFile();
            eigenfaces.save(f.getAbsolutePath());
        }
        return true;
    }

}
