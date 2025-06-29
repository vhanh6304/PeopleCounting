package com.peoplecounting;

import com.peoplecounting.database.DatabaseManager;
import com.peoplecounting.logic.EntryCounter;
import com.peoplecounting.vision.BoundingBox;
import com.peoplecounting.vision.CentroidTracker;
import com.peoplecounting.vision.HeadDetector;

import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import java.awt.event.KeyEvent;
import java.util.List;
import java.util.Map;

/**
 * The main class that orchestrates the entire people counting system.
 * It initializes all components and runs the main application loop.
 */
public class Main {

    public static void main(String[] args) throws Exception {
        System.out.println("Starting People Counting System...");

        // --- 1. INITIALIZATION ---
        // Create an instance of each of your components.
        System.out.println("Initializing components...");
        HeadDetector detector = new HeadDetector();
        CentroidTracker tracker = new CentroidTracker();
        DatabaseManager dbManager = new DatabaseManager();

        // --- Video Capture Setup ---
        // Use 0 for the default webcam. You can also pass a path to a video file.
        // e.g., "C:/Users/YourUser/Videos/test_video.mp4"
        FrameGrabber grabber = new FFmpegFrameGrabber("C:\\Users\\noobp\\Desktop\\MOT17-12-DPM-raw.webm");
        System.out.println("Starting video grabber...");
        grabber.start();

        // The Y-coordinate for the virtual line. We place it in the middle of the frame.
        final int VIRTUAL_LINE_Y = grabber.getImageHeight() / 2;
        EntryCounter counter = new EntryCounter(VIRTUAL_LINE_Y, dbManager);

        // A CanvasFrame is a simple window to display the processed video feed.
        CanvasFrame canvas = new CanvasFrame("Automated People Counter", CanvasFrame.getDefaultGamma() / grabber.getGamma());
        canvas.setCanvasSize(grabber.getImageWidth(), grabber.getImageHeight());

        // OpenCVFrameConverter allows us to easily convert between Frame and Mat objects.
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        System.out.println("Initialization complete. Starting main loop...");
        System.out.println("Press 'Q' in the video window to quit.");

        // --- 2. MAIN APPLICATION LOOP ---
        // The loop continues as long as the display window is open and frames are available.
        while (canvas.isVisible() && (grabber.grab()) != null) {
            Frame frame = grabber.getCurrentFrame();
            if (frame == null || frame.image == null) {
                continue;
            }

            // Convert the captured frame to an OpenCV Mat for processing and drawing.
            Mat mat = converter.convert(frame);

            // --- 3. ORCHESTRATION (The Core Logic Pipeline) ---

            // Step A: Detect heads in the current frame.
            List<BoundingBox> detections = detector.detectPersons(frame);

            // Step B: Update the tracker with the new detections.
            Map<Integer, Point> trackedObjects = tracker.update(detections);

            // Step C: Update the counts based on the movement of tracked objects.
            counter.updateCount(trackedObjects);

            // --- 4. VISUALIZATION ---
            // Draw all the visual elements onto the Mat.

            // Draw the virtual line.
            line(mat, new Point(0, VIRTUAL_LINE_Y), new Point(mat.cols(), VIRTUAL_LINE_Y), new Scalar(0, 255, 255, 0), 2, LINE_AA, 0);

            // Draw bounding boxes and IDs for each tracked object.
            for (Map.Entry<Integer, Point> entry : trackedObjects.entrySet()) {
                int id = entry.getKey();
                Point centroid = entry.getValue();

                // Find the original bounding box for this ID to draw it.
                // This is a simple approach; a more robust one might store the box in the tracker.
                for(BoundingBox box : detections) {
                    if(box.getCentroid().x == centroid.x && box.getCentroid().y == centroid.y) {
                        Rect rect = box.toRect();
                        rectangle(mat, rect, new Scalar(0, 255, 0, 0), 2); // Green box
                        String text = "ID " + id;
                        putText(mat, text, new Point(rect.x(), rect.y() - 10), FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 255, 0, 0), 2);
                        break;
                    }
                }
            }

            // Draw the statistics text on the top-left corner.
            String statsText = String.format("In: %d | Out: %d", counter.getEntryCount(), counter.getExitCount());
            putText(mat, statsText, new Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.2, new Scalar(255, 0, 0, 0), 3);

            // Convert the modified Mat back to a Frame to display it.
            Frame processedFrame = converter.convert(mat);
            canvas.showImage(processedFrame);

            // Allow quitting by pressing 'q'
            KeyEvent key = canvas.waitKey(20); // Wait 20ms for a key press
            if (key != null && (key.getKeyCode() == KeyEvent.VK_Q)) {
                System.out.println("'Q' pressed. Exiting...");
                break;
            }
            // Release the Mat to prevent memory leaks
            mat.release();
        }

        // --- 5. CLEANUP ---
        System.out.println("Cleaning up resources...");
        canvas.dispose();
        grabber.stop();
        grabber.release();
        System.out.println("System stopped.");
    }
}