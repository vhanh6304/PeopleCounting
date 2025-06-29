package com.peoplecounting.vision;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Handles the detection of persons in a given video frame using a pre-trained TinyYOLO model.
 * This class encapsulates the deep learning logic.
 */
public class HeadDetector {

    // --- Model Configuration ---
    private ComputationGraph yoloModel;
    private static final int YOLO_INPUT_WIDTH = 416;
    private static final int YOLO_INPUT_HEIGHT = 416;
    private static final double DETECTION_THRESHOLD = 0.5; // Only accept detections with 50% or higher confidence.
    private static final List<String> COCO_CLASSES = {"person"}; // We only care about the 'person' class.

    public HeadDetector() {
        try {
            System.out.println("Loading pre-trained TinyYOLO model...");
            // DL4J's Model Zoo makes loading pre-trained models straightforward.
            // This will download the model weights the first time it is run.
            this.yoloModel = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
            System.out.println("Model loaded successfully.");
        } catch (IOException e) {
            // This is a critical error. The application cannot run without the model.
            throw new RuntimeException("Fatal error: Could not initialize or load the TinyYOLO model.", e);
        }
    }

    /**
     * Detects persons in a given video frame.
     * @param frame The raw video frame from the camera.
     * @return A list of BoundingBox objects for each detected person.
     */
    public List<BoundingBox> detectPersons(Frame frame) {
        // Get the original frame dimensions for scaling the results later.
        int frameWidth = frame.imageWidth;
        int frameHeight = frame.imageHeight;

        // Preprocess the frame to be suitable for the YOLO model.
        INDArray inputImage = preprocessFrame(frame);

        // Feed the preprocessed image into the model to get predictions.
        // This is the core inference step.
        INDArray results = yoloModel.outputSingle(inputImage);

        // The model's output is a complex tensor. We need a utility to parse it.
        // YoloUtils.getDetectedObjects will decode this tensor into a list of human-readable objects.
        List<DetectedObject> detectedObjects = YoloUtils.getDetectedObjects(
                results, DETECTION_THRESHOLD, COCO_CLASSES
        );

        // Convert the DL4J DetectedObject list into our project's BoundingBox list.
        return convertToBoundingBoxes(detectedObjects, frameWidth, frameHeight);
    }

    /**
     * Converts a raw frame into an INDArray suitable for the YOLO model.
     * This involves converting to a Mat, resizing, and normalizing pixel values.
     */
    private INDArray preprocessFrame(Frame frame) {
        // Convert the frame to a BufferedImage, then to an ND4J INDArray.
        // The Mat is the bridge between OpenCV-land and DL4J-land.
        Mat mat = Java2DFrameUtils.toMat(Java2DFrameUtils.toBufferedImage(frame));

        // The NativeImageLoader handles resizing and arranging color channels correctly.
        org.datavec.image.loader.NativeImageLoader loader = new org.datavec.image.loader.NativeImageLoader(
                YOLO_INPUT_HEIGHT,
                YOLO_INPUT_WIDTH,
                3 // 3 color channels (RGB)
        );

        INDArray rawImage;
        try {
            rawImage = loader.asMatrix(mat);
        } catch (IOException e) {
            throw new RuntimeException("Could not convert frame to INDArray", e);
        } finally {
            // It's crucial to release the Mat to prevent memory leaks.
            mat.release();
        }

        // Normalize the pixel values to be between 0 and 1 (from 0-255).
        // This is a standard practice for neural networks.
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(rawImage);

        return rawImage;
    }

    /**
     * Converts the model's output objects into our simple BoundingBox format.
     * The model's output coordinates are relative to the 416x416 input size,
     * so they must be scaled back to the original frame's dimensions.
     */
    private List<BoundingBox> convertToBoundingBoxes(List<DetectedObject> objects, int originalWidth, int originalHeight) {
        List<BoundingBox> boxes = new ArrayList<>();
        for (DetectedObject obj : objects) {
            // Extract the coordinates from the detected object.
            // These are relative to the 416x416 input.
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();

            // Calculate the absolute coordinates on the original frame.
            int x = (int) Math.round(xy1[0] * originalWidth / YOLO_INPUT_WIDTH);
            int y = (int) Math.round(xy1[1] * originalHeight / YOLO_INPUT_HEIGHT);
            int width = (int) Math.round((xy2[0] - xy1[0]) * originalWidth / YOLO_INPUT_WIDTH);
            int height = (int) Math.round((xy2[1] - xy1[1]) * originalHeight / YOLO_INPUT_HEIGHT);

            boxes.add(new BoundingBox(x, y, width, height, obj.getConfidence()));
        }
        return boxes;
    }
}

