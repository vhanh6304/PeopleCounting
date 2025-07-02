package com.peoplecounting;

import com.peoplecounting.database.DatabaseManager;
import com.peoplecounting.logic.EntryCounter;
import com.peoplecounting.vision.BoundingBox;
import com.peoplecounting.vision.CentroidTracker;
import com.peoplecounting.vision.HeadDetector;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import java.awt.event.KeyEvent;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Main class with CSV logging and on-screen count display.
 */
public class Main {

    public static void main(String[] args) throws Exception {
        System.out.println("Starting People Counting System...");

        // --- 1. INITIALIZATION ---
        HeadDetector detector = new HeadDetector();
        CentroidTracker tracker = new CentroidTracker();
        DatabaseManager dbManager = new DatabaseManager();

        // CSV logger
        CSVLogger csvLogger = new CSVLogger("counts.csv");

        // --- Video Capture Setup ---
        FrameGrabber grabber = new FFmpegFrameGrabber("/path/to/video.mp4");
        grabber.start();

        final int VIRTUAL_LINE_Y = grabber.getImageHeight() / 2;
        EntryCounter counter = new EntryCounter(VIRTUAL_LINE_Y, dbManager);

        CanvasFrame canvas = new CanvasFrame("Automated People Counter",
                CanvasFrame.getDefaultGamma() / grabber.getGamma());
        canvas.setCanvasSize(grabber.getImageWidth(), grabber.getImageHeight());

        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        int frameIndex = 0;

        // --- 2. MAIN LOOP ---
        while (canvas.isVisible() && grabber.grab() != null) {
            Frame frame = grabber.getCurrentFrame();
            if (frame == null || frame.image == null) continue;

            Mat mat = converter.convert(frame);

            // Detection and counting
            List<BoundingBox> detections = detector.detectPersons(frame);
            Map<Integer, Point> trackedObjects = tracker.update(detections);
            counter.updateCount(trackedObjects);
            int inCount = counter.getEntryCount();
            int outCount = counter.getExitCount();

            // Draw virtual line
            line(mat, new Point(0, VIRTUAL_LINE_Y), new Point(mat.cols(), VIRTUAL_LINE_Y),
                    new Scalar(0, 255, 255, 0), 2, LINE_AA, 0);

            // Draw detections
            for (BoundingBox box : detections) {
                Rect rect = box.toRect();
                rectangle(mat, rect, new Scalar(0, 255, 0, 0), 2);
            }

            // Display counts on screen
            String statsText = String.format("In: %d  Out: %d", inCount, outCount);
            putText(mat, statsText, new Point(20, 40), FONT_HERSHEY_SIMPLEX,
                    1.2, new Scalar(255, 0, 0, 0), 3);

            // Log to CSV
            csvLogger.log(frameIndex, inCount, outCount);

            // Show image
            canvas.showImage(converter.convert(mat));

            KeyEvent key = canvas.waitKey(20);
            if (key != null && key.getKeyCode() == KeyEvent.VK_Q) break;
            mat.release();
            frameIndex++;
        }

        // Cleanup
        csvLogger.close();
        canvas.dispose();
        grabber.stop();
        grabber.release();
        System.out.println("System stopped.");
    }
}

// CSV Logger class
class CSVLogger {
    private java.io.BufferedWriter writer;
    public CSVLogger(String filename) throws IOException {
        writer = new java.io.BufferedWriter(new java.io.FileWriter(filename));
        writer.write("frame,in_count,out_count\n");
    }
    public void log(int frameIdx, int inCount, int outCount) {
        try {
            writer.write(String.format("%d,%d,%d\n", frameIdx, inCount, outCount));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void close() {
        try { writer.close(); } catch (IOException e) { e.printStackTrace(); }
    }
}
