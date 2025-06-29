package com.peoplecounting.vision;

import org.opencv.core.Point;
import org.opencv.core.Rect;

/**
 * A simple data class to hold information about a detected object.
 * We use a Java Record for a concise, immutable data carrier.
 *
 * @param x The top-left x-coordinate of the box.
 * @param y The top-left y-coordinate of the box.
 * @param width The width of the box.
 * @param height The height of the box.
 * @param confidence The model's confidence in this detection (e.g., 0.95 for 95%).
 */
public record BoundingBox(int x, int y, int width, int height, double confidence) {

    /**
     * Calculates and returns the center point (centroid) of the bounding box.
     * @return The centroid as an OpenCV Point.
     */
    public Point getCentroid() {
        return new Point(x + width / 2.0, y + height / 2.0);
    }

    /**
     * Creates an OpenCV Rect object from this bounding box.
     * Useful for drawing on a frame.
     * @return The bounding box as an OpenCV Rect.
     */
    public Rect toRect() {
        return new Rect(x, y, width, height);
    }
}