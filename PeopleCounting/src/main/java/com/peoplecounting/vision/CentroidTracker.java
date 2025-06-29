package com.peoplecounting.vision;

import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Tracks objects based on the Euclidean distance between centroids.
 * This is a robust, straightforward tracking algorithm perfect for this use case.
 */
public class CentroidTracker {

    private int nextObjectID = 0;
    // Map of object ID to its last known centroid
    private final Map<Integer, Point> trackedObjects = new HashMap<>();
    // Map of object ID to the number of consecutive frames it has been 'lost'
    private final Map<Integer, Integer> disappearedFrames = new HashMap<>();

    // Maximum number of frames an object can be lost before we deregister it
    private static final int MAX_DISAPPEARED_FRAMES = 30;

    /**
     * Updates the state of the tracker with a new set of detected bounding boxes.
     *
     * @param detectedBoxes A list of BoundingBox objects from the detector for the current frame.
     * @return A map of currently tracked object IDs to their centroid Points.
     */
    public Map<Integer, Point> update(List<BoundingBox> detectedBoxes) {

        // If there are no detections in the current frame
        if (detectedBoxes.isEmpty()) {
            // Mark all existing tracked objects as disappeared
            for (Integer objectID : new ArrayList<>(disappearedFrames.keySet())) {
                disappearedFrames.put(objectID, disappearedFrames.get(objectID) + 1);

                // If an object has been missing for too long, deregister it
                if (disappearedFrames.get(objectID) > MAX_DISAPPEARED_FRAMES) {
                    deregister(objectID);
                }
            }
            return Collections.unmodifiableMap(trackedObjects);
        }

        // --- Prepare data for the current frame ---
        // List of centroids from the new bounding boxes
        List<Point> inputCentroids = new ArrayList<>();
        for (BoundingBox box : detectedBoxes) {
            inputCentroids.add(box.getCentroid());
        }

        // If we are not currently tracking any objects, register all new detections
        if (trackedObjects.isEmpty()) {
            for (Point centroid : inputCentroids) {
                register(centroid);
            }
            return Collections.unmodifiableMap(trackedObjects);
        }

        // --- Match existing objects to new detections ---
        List<Integer> objectIDs = new ArrayList<>(trackedObjects.keySet());
        List<Point> objectCentroids = new ArrayList<>(trackedObjects.values());

        // Calculate the distance between each pair of old and new centroids
        // D[i][j] will be the distance between object i and detection j
        double[][] D = new double[objectCentroids.size()][inputCentroids.size()];
        for (int i = 0; i < objectCentroids.size(); i++) {
            for (int j = 0; j < inputCentroids.size(); j++) {
                D[i][j] = calculateEuclideanDistance(objectCentroids.get(i), inputCentroids.get(j));
            }
        }

        // Find the smallest value in each row and sort them by value
        // This gives us the best potential matches first
        List<int[]> sortedRows = new ArrayList<>();
        for (int i = 0; i < D.length; i++) {
            int minIndex = -1;
            double minValue = Double.MAX_VALUE;
            for (int j = 0; j < D[i].length; j++) {
                if (D[i][j] < minValue) {
                    minValue = D[i][j];
                    minIndex = j;
                }
            }
            sortedRows.add(new int[]{i, minIndex}); // Store as [row_index, col_index_of_min]
        }
        // Sort by the distance value itself
        sortedRows.sort((a, b) -> Double.compare(D[a[0]][a[1]], D[b[0]][b[1]]));


        Set<Integer> usedRows = new HashSet<>();
        Set<Integer> usedCols = new HashSet<>();

        // Loop over the potential matches and select the best ones
        for (int[] pair : sortedRows) {
            int row = pair[0];
            int col = pair[1];

            // If we have already used this row or column, skip it
            if (usedRows.contains(row) || usedCols.contains(col)) {
                continue;
            }

            // This is a good match: update the object's position
            int objectID = objectIDs.get(row);
            trackedObjects.put(objectID, inputCentroids.get(col));
            disappearedFrames.put(objectID, 0); // Reset disappeared counter

            usedRows.add(row);
            usedCols.add(col);
        }

        // --- Handle unmatched objects ---
        Set<Integer> allRows = new HashSet<>();
        for(int i = 0; i < objectCentroids.size(); i++) allRows.add(i);

        Set<Integer> allCols = new HashSet<>();
        for(int i = 0; i < inputCentroids.size(); i++) allCols.add(i);

        Set<Integer> unusedRows = new HashSet<>(allRows);
        unusedRows.removeAll(usedRows);

        Set<Integer> unusedCols = new HashSet<>(allCols);
        unusedCols.removeAll(usedCols);

        // If there are more old objects than new detections, some may have disappeared
        if (objectCentroids.size() >= inputCentroids.size()) {
            for (int row : unusedRows) {
                int objectID = objectIDs.get(row);
                disappearedFrames.put(objectID, disappearedFrames.get(objectID) + 1);

                if (disappearedFrames.get(objectID) > MAX_DISAPPEARED_FRAMES) {
                    deregister(objectID);
                }
            }
        }
        // Otherwise, if there are new detections left over, register them as new objects
        else {
            for (int col : unusedCols) {
                register(inputCentroids.get(col));
            }
        }

        return Collections.unmodifiableMap(trackedObjects);
    }

    /**
     * Registers a new object with a new ID.
     */
    private void register(Point centroid) {
        trackedObjects.put(nextObjectID, centroid);
        disappearedFrames.put(nextObjectID, 0);
        nextObjectID++;
    }

    /**
     * Deregisters an old object, removing it from tracking.
     */
    private void deregister(int objectID) {
        trackedObjects.remove(objectID);
        disappearedFrames.remove(objectID);
    }

    /**
     * Helper to calculate the distance between two points.
     */
    private double calculateEuclideanDistance(Point p1, Point p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
}