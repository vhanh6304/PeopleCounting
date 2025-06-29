package com.peoplecounting.logic;

import com.peoplecounting.database.DatabaseManager;
import org.opencv.core.Point;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Contains the logic for counting people crossing a virtual line.
 * It uses the tracked objects' positions to determine entry and exit events.
 */
public class EntryCounter {

    private int entryCount = 0;
    private int exitCount = 0;

    // The Y-coordinate of our virtual horizontal line.
    private final int virtualLineY;

    // A reference to the database manager to log events.
    private final DatabaseManager dbManager;

    // Stores the previous position of each tracked object to detect movement direction.
    private final Map<Integer, Point> positionHistory = new HashMap<>();

    // Stores the IDs of people who have already been counted to prevent double-counting.
    private final Set<Integer> countedObjectIDs = new HashSet<>();

    /**
     * Constructor for the EntryCounter.
     * @param virtualLineY The Y-coordinate for the virtual line.
     * @param dbManager A reference to the database manager for logging.
     */
    public EntryCounter(int virtualLineY, DatabaseManager dbManager) {
        this.virtualLineY = virtualLineY;
        this.dbManager = dbManager;
    }

    /**
     * Updates the counts based on the latest positions of tracked objects.
     * @param trackedObjects A map of object IDs to their current centroid positions from CentroidTracker.
     */
    public void updateCount(Map<Integer, Point> trackedObjects) {
        for (Map.Entry<Integer, Point> entry : trackedObjects.entrySet()) {
            int id = entry.getKey();
            Point currentPos = entry.getValue();
            Point prevPos = positionHistory.get(id);

            // We can only determine a crossing if we have a previous position.
            if (prevPos != null) {
                // Check for a crossing event only if this ID has not been counted yet.
                if (!countedObjectIDs.contains(id)) {
                    // Check for an ENTRY event (moving from top to bottom across the line)
                    if (prevPos.y < virtualLineY && currentPos.y >= virtualLineY) {
                        entryCount++;
                        dbManager.logEvent("entry", id); // Log the event
                        countedObjectIDs.add(id);         // Mark as counted
                    }
                    // Check for an EXIT event (moving from bottom to top across the line)
                    else if (prevPos.y > virtualLineY && currentPos.y <= virtualLineY) {
                        exitCount++;
                        dbManager.logEvent("exit", id);  // Log the event
                        countedObjectIDs.add(id);          // Mark as counted
                    }
                }
            }

            // Always update the position history for the next frame.
            positionHistory.put(id, currentPos);
        }
    }

    // --- Getter methods to retrieve current counts for the UI ---

    public int getEntryCount() {
        return entryCount;
    }

    public int getExitCount() {
        return exitCount;
    }
}