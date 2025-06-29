package com.peoplecounting.database;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;

/**
 * Manages all database interactions.
 * This class handles the connection to the SQLite database and provides
 * methods to initialize the schema and log entry/exit events.
 */
public class DatabaseManager {
    // The path to the SQLite database file.
    // This will create a file named "entry_system.db" in the root of your project folder.
    private static final String DB_URL = "jdbc:sqlite:entry_system.db";

    public DatabaseManager() {
        // When the manager is created, ensure the necessary table exists.
        initializeDatabase();
    }

    /**
     * Creates the entry_logs table if it doesn't already exist.
     */
    private void initializeDatabase() {
        // This is the SQL command to create our table.
        // It includes a primary key, a timestamp that defaults to the current time,
        // the type of event, and the ID of the person tracked.
        String sql = "CREATE TABLE IF NOT EXISTS entry_logs ("
                + "    log_id INTEGER PRIMARY KEY AUTOINCREMENT,"
                + "    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,"
                + "    event_type TEXT NOT NULL,"
                + "    person_id INTEGER NOT NULL"
                + ");";

        try (Connection conn = DriverManager.getConnection(DB_URL);
             Statement stmt = conn.createStatement()) {
            // Execute the SQL command
            stmt.execute(sql);
        } catch (SQLException e) {
            System.err.println("Database initialization failed: " + e.getMessage());
            // In a real application, you might want to handle this more gracefully.
            // For this project, printing the error is sufficient.
        }
    }

    /**
     * Logs an event (like 'entry' or 'exit') to the database.
     * @param eventType A string describing the event (e.g., "entry").
     * @param personId The unique ID of the person from the CentroidTracker.
     */
    public void logEvent(String eventType, int personId) {
        String sql = "INSERT INTO entry_logs(event_type, person_id) VALUES(?, ?)";

        try (Connection conn = DriverManager.getConnection(DB_URL);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            // Set the parameters for the SQL query
            pstmt.setString(1, eventType);
            pstmt.setInt(2, personId);
            // Execute the command to insert the data
            pstmt.executeUpdate();
        } catch (SQLException e) {
            System.err.println("Failed to log event to database: " + e.getMessage());
        }
    }
}