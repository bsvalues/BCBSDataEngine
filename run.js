/**
 * BCBS Values - Application Startup Script
 * This script initializes the database and starts the Express server
 */

require('dotenv').config();
const { initializeDatabase } = require('./db/init');
const { createServer } = require('http');
const { Pool } = require('pg');
const fs = require('fs');
const path = require('path');

// Check if the public directory exists, create it if not
if (!fs.existsSync(path.join(__dirname, 'public'))) {
  fs.mkdirSync(path.join(__dirname, 'public'));
  console.log('Created public directory');
}

// Create a PostgreSQL connection pool
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

// Test the database connection
async function testDatabaseConnection() {
  try {
    const client = await pool.connect();
    console.log('Successfully connected to PostgreSQL database');
    
    // Check if tables exist
    const tablesResult = await client.query(`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public' 
      ORDER BY table_name;
    `);
    
    if (tablesResult.rows.length === 0) {
      console.log('No tables found in database, will initialize schema');
      await initializeDatabase();
    } else {
      console.log('Existing tables found:', tablesResult.rows.map(row => row.table_name).join(', '));
    }
    
    client.release();
    return true;
  } catch (error) {
    console.error('Database connection error:', error.message);
    return false;
  }
}

// Main startup function
async function startup() {
  console.log('Starting BCBS Values application...');
  
  // Test database connection
  const dbConnected = await testDatabaseConnection();
  if (!dbConnected) {
    console.error('Failed to connect to database. Please check your DATABASE_URL environment variable.');
    process.exit(1);
  }
  
  // Start the Express server
  const server = require('./server');
  const httpServer = createServer(server);
  
  const PORT = process.env.PORT || 5002;
  
  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

// Run the startup process
startup().catch(error => {
  console.error('Application startup failed:', error);
  process.exit(1);
});