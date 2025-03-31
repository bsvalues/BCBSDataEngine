/**
 * BCBS Values - Database Initialization Script
 * This script initializes the PostgreSQL database with the schema and sample data
 */

const { Pool } = require('pg');
const fs = require('fs');
const path = require('path');

// Create a PostgreSQL connection pool
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

// Initialize the database schema and sample data
async function initializeDatabase() {
  const client = await pool.connect();
  console.log('Creating database schema...');
  
  try {
    // Start a transaction
    await client.query('BEGIN');
    
    // Create Property table
    await client.query(`
      CREATE TABLE IF NOT EXISTS property (
        id SERIAL PRIMARY KEY,
        property_id VARCHAR(64) UNIQUE NOT NULL,
        address VARCHAR(256) NOT NULL,
        neighborhood VARCHAR(128),
        property_type VARCHAR(64) NOT NULL,
        year_built INTEGER,
        bedrooms INTEGER,
        bathrooms FLOAT,
        living_area FLOAT,
        land_area FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    // Create PropertyValuation table
    await client.query(`
      CREATE TABLE IF NOT EXISTS property_valuation (
        id SERIAL PRIMARY KEY,
        property_id INTEGER REFERENCES property(id) NOT NULL,
        estimated_value FLOAT NOT NULL,
        confidence_score FLOAT NOT NULL,
        valuation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        valuation_method VARCHAR(64) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    // Create EtlStatus table
    await client.query(`
      CREATE TABLE IF NOT EXISTS etl_status (
        id SERIAL PRIMARY KEY,
        status VARCHAR(64) NOT NULL DEFAULT 'idle',
        progress FLOAT DEFAULT 0.0,
        last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        records_processed INTEGER DEFAULT 0,
        success_rate FLOAT DEFAULT 0.0,
        average_processing_time FLOAT DEFAULT 0.0,
        completeness FLOAT DEFAULT 0.0,
        accuracy FLOAT DEFAULT 0.0,
        timeliness FLOAT DEFAULT 0.0
      )
    `);
    
    // Create DataSource table
    await client.query(`
      CREATE TABLE IF NOT EXISTS data_source (
        id SERIAL PRIMARY KEY,
        name VARCHAR(64) NOT NULL,
        status VARCHAR(64) NOT NULL DEFAULT 'idle',
        records INTEGER DEFAULT 0,
        etl_status_id INTEGER REFERENCES etl_status(id) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    // Create Agent table
    await client.query(`
      CREATE TABLE IF NOT EXISTS agent (
        id SERIAL PRIMARY KEY,
        agent_id VARCHAR(64) UNIQUE NOT NULL,
        agent_type VARCHAR(64) NOT NULL,
        status VARCHAR(64) NOT NULL DEFAULT 'idle',
        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        queue_size INTEGER DEFAULT 0,
        total_processed INTEGER DEFAULT 0,
        success_rate FLOAT DEFAULT 0.0,
        average_processing_time FLOAT DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    // Create AgentLog table
    await client.query(`
      CREATE TABLE IF NOT EXISTS agent_log (
        id SERIAL PRIMARY KEY,
        agent_id INTEGER REFERENCES agent(id) NOT NULL,
        level VARCHAR(16) NOT NULL DEFAULT 'info',
        message TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    // Create User table
    await client.query(`
      CREATE TABLE IF NOT EXISTS "user" (
        id SERIAL PRIMARY KEY,
        username VARCHAR(64) UNIQUE NOT NULL,
        email VARCHAR(120) UNIQUE NOT NULL,
        password_hash VARCHAR(256),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    // Create ApiKey table
    await client.query(`
      CREATE TABLE IF NOT EXISTS api_key (
        id SERIAL PRIMARY KEY,
        key VARCHAR(64) UNIQUE NOT NULL,
        name VARCHAR(64) NOT NULL,
        user_id INTEGER REFERENCES "user"(id) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_used TIMESTAMP,
        is_active BOOLEAN DEFAULT TRUE
      )
    `);
    
    // Insert initial ETL status record
    await client.query(`
      INSERT INTO etl_status (status, progress, records_processed)
      VALUES ('idle', 0.0, 0)
      ON CONFLICT DO NOTHING
    `);
    
    // Get the ETL status ID
    const etlResult = await client.query('SELECT id FROM etl_status LIMIT 1');
    const etlStatusId = etlResult.rows[0].id;
    
    // Insert sample data sources
    const dataSources = [
      { name: 'County Tax Records', status: 'idle', records: 0 },
      { name: 'MLS Listings', status: 'idle', records: 0 },
      { name: 'Census Data', status: 'idle', records: 0 },
      { name: 'GIS Data', status: 'idle', records: 0 }
    ];
    
    for (const source of dataSources) {
      await client.query(`
        INSERT INTO data_source (name, status, records, etl_status_id)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (name) DO UPDATE SET
          status = EXCLUDED.status,
          records = EXCLUDED.records
      `, [source.name, source.status, source.records, etlStatusId]);
    }
    
    // Insert sample agents
    const agents = [
      { agent_id: 'valuation-agent-1', agent_type: 'valuation', status: 'idle' },
      { agent_id: 'etl-agent-1', agent_type: 'etl', status: 'idle' },
      { agent_id: 'gis-agent-1', agent_type: 'gis', status: 'idle' }
    ];
    
    for (const agent of agents) {
      await client.query(`
        INSERT INTO agent (agent_id, agent_type, status)
        VALUES ($1, $2, $3)
        ON CONFLICT (agent_id) DO UPDATE SET
          agent_type = EXCLUDED.agent_type,
          status = EXCLUDED.status
      `, [agent.agent_id, agent.agent_type, agent.status]);
    }
    
    // Insert sample properties (Benton County, WA)
    const properties = [
      { 
        property_id: 'BCWA12345', 
        address: '123 Main St, Kennewick, WA 99336', 
        neighborhood: 'Downtown Kennewick', 
        property_type: 'Single Family', 
        year_built: 1985, 
        bedrooms: 3, 
        bathrooms: 2.5, 
        living_area: 2200, 
        land_area: 8500
      },
      { 
        property_id: 'BCWA12346', 
        address: '456 Oak Ave, Richland, WA 99352', 
        neighborhood: 'North Richland', 
        property_type: 'Single Family', 
        year_built: 1992, 
        bedrooms: 4, 
        bathrooms: 3, 
        living_area: 2800, 
        land_area: 10000
      },
      { 
        property_id: 'BCWA12347', 
        address: '789 Pine Ln, Pasco, WA 99301', 
        neighborhood: 'West Pasco', 
        property_type: 'Single Family', 
        year_built: 2005, 
        bedrooms: 3, 
        bathrooms: 2, 
        living_area: 1800, 
        land_area: 7500
      },
      { 
        property_id: 'BCWA12348', 
        address: '101 Columbia Dr, Richland, WA 99352', 
        neighborhood: 'South Richland', 
        property_type: 'Condo', 
        year_built: 2010, 
        bedrooms: 2, 
        bathrooms: 2, 
        living_area: 1500, 
        land_area: 0
      },
      { 
        property_id: 'BCWA12349', 
        address: '202 River Rd, Kennewick, WA 99336', 
        neighborhood: 'Clover Island', 
        property_type: 'Townhouse', 
        year_built: 2015, 
        bedrooms: 3, 
        bathrooms: 2.5, 
        living_area: 1900, 
        land_area: 3000
      }
    ];
    
    for (const property of properties) {
      // Insert property
      const propResult = await client.query(`
        INSERT INTO property 
          (property_id, address, neighborhood, property_type, year_built, bedrooms, bathrooms, living_area, land_area)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (property_id) DO UPDATE SET
          address = EXCLUDED.address,
          neighborhood = EXCLUDED.neighborhood,
          property_type = EXCLUDED.property_type,
          year_built = EXCLUDED.year_built,
          bedrooms = EXCLUDED.bedrooms,
          bathrooms = EXCLUDED.bathrooms,
          living_area = EXCLUDED.living_area,
          land_area = EXCLUDED.land_area
        RETURNING id
      `, [
        property.property_id, 
        property.address, 
        property.neighborhood, 
        property.property_type, 
        property.year_built, 
        property.bedrooms, 
        property.bathrooms, 
        property.living_area, 
        property.land_area
      ]);
      
      const propertyId = propResult.rows[0].id;
      
      // Add valuations for the property (one for each of the last 3 months)
      const baseValue = property.property_type === 'Single Family' ? 350000 : 
                       property.property_type === 'Condo' ? 250000 : 300000;
      
      // Adjust value based on property attributes
      const adjustedBaseValue = baseValue + 
                             (property.living_area - 2000) * 100 + 
                             (property.year_built - 2000) * 500 +
                             property.bedrooms * 15000;
      
      // Create valuations with slight variations for the last 3 months
      const valuations = [
        {
          estimated_value: adjustedBaseValue * 0.98, 
          confidence_score: 0.85, 
          valuation_date: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
          valuation_method: 'comparative'
        },
        {
          estimated_value: adjustedBaseValue, 
          confidence_score: 0.92, 
          valuation_date: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000),
          valuation_method: 'hybrid'
        },
        {
          estimated_value: adjustedBaseValue * 1.02, 
          confidence_score: 0.89, 
          valuation_date: new Date(),
          valuation_method: 'advanced-regression'
        }
      ];
      
      for (const valuation of valuations) {
        await client.query(`
          INSERT INTO property_valuation 
            (property_id, estimated_value, confidence_score, valuation_date, valuation_method)
          VALUES ($1, $2, $3, $4, $5)
        `, [
          propertyId, 
          valuation.estimated_value, 
          valuation.confidence_score, 
          valuation.valuation_date,
          valuation.valuation_method
        ]);
      }
    }
    
    // Commit the transaction
    await client.query('COMMIT');
    console.log('Database initialization completed successfully');
  } catch (error) {
    // Rollback the transaction on error
    await client.query('ROLLBACK');
    console.error('Database initialization failed:', error);
    throw error;
  } finally {
    // Release the client back to the pool
    client.release();
  }
}

module.exports = { initializeDatabase };