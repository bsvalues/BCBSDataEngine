/**
 * BCBS Values - Express Server
 * This server provides API endpoints for property valuations and dashboard data
 */

const express = require('express');
const path = require('path');
const { Pool } = require('pg');
const cors = require('cors');
const crypto = require('crypto');

// Create Express application
const app = express();

// Create PostgreSQL connection pool
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// Simple API key authentication middleware
const authenticateApiKey = async (req, res, next) => {
  // Skip authentication for public endpoints
  if (req.path === '/' || req.path.startsWith('/static') || req.path === '/index.html') {
    return next();
  }
  
  // Get API key from header
  const apiKey = req.header('X-API-Key');
  
  // If no API key provided, check if we are in development mode
  if (!apiKey) {
    if (process.env.NODE_ENV === 'development') {
      console.log('Warning: No API key provided, but allowing access in development mode');
      return next();
    }
    return res.status(401).json({ error: 'API key is required' });
  }
  
  try {
    // Check if API key exists in database
    const result = await pool.query(`
      SELECT a.*, u.username 
      FROM api_key a
      JOIN "user" u ON a.user_id = u.id
      WHERE a.key = $1 AND a.is_active = true
    `, [apiKey]);
    
    if (result.rows.length === 0) {
      return res.status(401).json({ error: 'Invalid API key' });
    }
    
    // Update last used timestamp
    await pool.query(`
      UPDATE api_key
      SET last_used = CURRENT_TIMESTAMP
      WHERE key = $1
    `, [apiKey]);
    
    // Store user info in request for later use
    req.user = {
      id: result.rows[0].user_id,
      username: result.rows[0].username,
      apiKeyId: result.rows[0].id
    };
    
    next();
  } catch (error) {
    console.error('API key validation error:', error);
    return res.status(500).json({ error: 'Internal server error during authentication' });
  }
};

// Apply authentication middleware
app.use(authenticateApiKey);

// Helper function to generate a pagination object
const generatePagination = (page, itemsPerPage, totalItems) => {
  const totalPages = Math.ceil(totalItems / itemsPerPage);
  
  return {
    currentPage: page,
    itemsPerPage: itemsPerPage,
    totalItems: totalItems,
    totalPages: totalPages,
    hasNextPage: page < totalPages,
    hasPrevPage: page > 1
  };
};

// Endpoint: Get all properties with valuations (paginated and filterable)
app.get('/api/valuations', async (req, res) => {
  try {
    // Extract query parameters
    const page = parseInt(req.query.page) || 1;
    const perPage = parseInt(req.query.per_page) || 10;
    const offset = (page - 1) * perPage;
    const sortBy = req.query.sort_by || 'estimated_value';
    const sortDirection = req.query.sort_direction?.toUpperCase() === 'ASC' ? 'ASC' : 'DESC';
    
    // Build WHERE clause for filters
    const whereConditions = [];
    const queryParams = [];
    let paramCounter = 1;
    
    // Filter by neighborhood
    if (req.query.neighborhood) {
      whereConditions.push(`p.neighborhood = $${paramCounter}`);
      queryParams.push(req.query.neighborhood);
      paramCounter++;
    }
    
    // Filter by property type
    if (req.query.property_type) {
      whereConditions.push(`p.property_type = $${paramCounter}`);
      queryParams.push(req.query.property_type);
      paramCounter++;
    }
    
    // Filter by value range
    if (req.query.min_value) {
      whereConditions.push(`pv.estimated_value >= $${paramCounter}`);
      queryParams.push(parseFloat(req.query.min_value));
      paramCounter++;
    }
    if (req.query.max_value) {
      whereConditions.push(`pv.estimated_value <= $${paramCounter}`);
      queryParams.push(parseFloat(req.query.max_value));
      paramCounter++;
    }
    
    // Filter by date range
    if (req.query.from_date) {
      whereConditions.push(`pv.valuation_date >= $${paramCounter}`);
      queryParams.push(req.query.from_date);
      paramCounter++;
    }
    if (req.query.to_date) {
      whereConditions.push(`pv.valuation_date <= $${paramCounter}`);
      queryParams.push(req.query.to_date);
      paramCounter++;
    }
    
    // Search by property ID or address
    if (req.query.search) {
      whereConditions.push(`(p.property_id ILIKE $${paramCounter} OR p.address ILIKE $${paramCounter})`);
      queryParams.push(`%${req.query.search}%`);
      paramCounter++;
    }
    
    // Combine all conditions
    const whereClause = whereConditions.length > 0 ? 'WHERE ' + whereConditions.join(' AND ') : '';
    
    // Query for total count
    const countQuery = `
      SELECT COUNT(*) AS total
      FROM property p
      JOIN (
        SELECT DISTINCT ON (property_id) property_id, estimated_value, confidence_score, valuation_date, valuation_method
        FROM property_valuation
        ORDER BY property_id, valuation_date DESC
      ) pv ON p.id = pv.property_id
      ${whereClause}
    `;
    
    const countResult = await pool.query(countQuery, queryParams);
    const totalItems = parseInt(countResult.rows[0].total);
    
    // Query for paginated data
    const dataQuery = `
      SELECT 
        p.property_id,
        p.address,
        p.neighborhood,
        p.property_type,
        p.year_built,
        p.bedrooms,
        p.bathrooms,
        p.living_area,
        p.land_area,
        pv.estimated_value,
        pv.confidence_score,
        pv.valuation_date,
        pv.valuation_method
      FROM property p
      JOIN (
        SELECT DISTINCT ON (property_id) property_id, estimated_value, confidence_score, valuation_date, valuation_method
        FROM property_valuation
        ORDER BY property_id, valuation_date DESC
      ) pv ON p.id = pv.property_id
      ${whereClause}
      ORDER BY ${sortBy} ${sortDirection}
      LIMIT $${paramCounter} OFFSET $${paramCounter + 1}
    `;
    
    queryParams.push(perPage, offset);
    
    const dataResult = await pool.query(dataQuery, queryParams);
    
    // Get unique neighborhoods for filtering
    const neighborhoodsQuery = `
      SELECT DISTINCT neighborhood
      FROM property
      WHERE neighborhood IS NOT NULL
      ORDER BY neighborhood
    `;
    
    const neighborhoodsResult = await pool.query(neighborhoodsQuery);
    const neighborhoods = neighborhoodsResult.rows.map(row => row.neighborhood);
    
    // Get unique property types for filtering
    const propertyTypesQuery = `
      SELECT DISTINCT property_type
      FROM property
      ORDER BY property_type
    `;
    
    const propertyTypesResult = await pool.query(propertyTypesQuery);
    const propertyTypes = propertyTypesResult.rows.map(row => row.property_type);
    
    // Format the response with pagination and filters
    const response = {
      properties: dataResult.rows,
      pagination: generatePagination(page, perPage, totalItems),
      metadata: {
        neighborhoods: neighborhoods,
        property_types: propertyTypes,
        filters: {
          neighborhood: req.query.neighborhood || null,
          property_type: req.query.property_type || null,
          min_value: req.query.min_value ? parseFloat(req.query.min_value) : null,
          max_value: req.query.max_value ? parseFloat(req.query.max_value) : null,
          from_date: req.query.from_date || null,
          to_date: req.query.to_date || null,
          search: req.query.search || null
        },
        sort: {
          by: sortBy,
          direction: sortDirection.toLowerCase()
        }
      },
      total_count: totalItems,
      total_pages: Math.ceil(totalItems / perPage)
    };
    
    res.json(response);
  } catch (error) {
    console.error('Error fetching property valuations:', error);
    res.status(500).json({ error: 'Internal server error', details: error.message });
  }
});

// Endpoint: Get ETL pipeline status
app.get('/api/etl-status', async (req, res) => {
  try {
    // Query for ETL status
    const etlQuery = `
      SELECT 
        id,
        status,
        progress,
        last_update,
        records_processed,
        success_rate,
        average_processing_time,
        completeness,
        accuracy,
        timeliness
      FROM etl_status
      ORDER BY id DESC
      LIMIT 1
    `;
    
    const etlResult = await pool.query(etlQuery);
    
    if (etlResult.rows.length === 0) {
      return res.status(404).json({ error: 'ETL status not found' });
    }
    
    const etlStatus = etlResult.rows[0];
    
    // Query for data sources
    const sourcesQuery = `
      SELECT 
        id,
        name,
        status,
        records,
        created_at,
        updated_at
      FROM data_source
      WHERE etl_status_id = $1
      ORDER BY name
    `;
    
    const sourcesResult = await pool.query(sourcesQuery, [etlStatus.id]);
    
    // Format the response
    const response = {
      status: etlStatus.status,
      progress: etlStatus.progress,
      last_update: etlStatus.last_update,
      metrics: {
        records_processed: etlStatus.records_processed,
        success_rate: etlStatus.success_rate,
        average_processing_time: etlStatus.average_processing_time
      },
      data_quality: {
        completeness: etlStatus.completeness,
        accuracy: etlStatus.accuracy,
        timeliness: etlStatus.timeliness
      },
      sources: sourcesResult.rows
    };
    
    res.json(response);
  } catch (error) {
    console.error('Error fetching ETL status:', error);
    res.status(500).json({ error: 'Internal server error', details: error.message });
  }
});

// Endpoint: Get agent statuses
app.get('/api/agent-status', async (req, res) => {
  try {
    // Query for agents
    const agentsQuery = `
      SELECT 
        id,
        agent_id,
        agent_type,
        status,
        last_active,
        queue_size,
        total_processed AS "total_processed",
        success_rate,
        average_processing_time
      FROM agent
      ORDER BY agent_type, agent_id
    `;
    
    const agentsResult = await pool.query(agentsQuery);
    
    // Format the response with metrics
    const agents = agentsResult.rows.map(agent => ({
      ...agent,
      metrics: {
        success_rate: agent.success_rate,
        total_processed: agent.total_processed,
        average_processing_time: agent.average_processing_time
      }
    }));
    
    // Remove the original metrics fields
    agents.forEach(agent => {
      delete agent.success_rate;
      delete agent.total_processed;
      delete agent.average_processing_time;
    });
    
    const response = {
      agents: agents,
      last_update: new Date()
    };
    
    res.json(response);
  } catch (error) {
    console.error('Error fetching agent status:', error);
    res.status(500).json({ error: 'Internal server error', details: error.message });
  }
});

// Endpoint: Get agent logs
app.get('/api/agent-logs/:agentId', async (req, res) => {
  try {
    const agentId = req.params.agentId;
    
    // Query for agent
    const agentQuery = `
      SELECT id FROM agent WHERE agent_id = $1
    `;
    
    const agentResult = await pool.query(agentQuery, [agentId]);
    
    if (agentResult.rows.length === 0) {
      return res.status(404).json({ error: 'Agent not found' });
    }
    
    const agentDbId = agentResult.rows[0].id;
    
    // Query for logs
    const logsQuery = `
      SELECT 
        level,
        message,
        timestamp
      FROM agent_log
      WHERE agent_id = $1
      ORDER BY timestamp DESC
      LIMIT 100
    `;
    
    const logsResult = await pool.query(logsQuery, [agentDbId]);
    
    // Format the response
    const response = {
      agent_id: agentId,
      logs: logsResult.rows
    };
    
    res.json(response);
  } catch (error) {
    console.error(`Error fetching logs for agent ${req.params.agentId}:`, error);
    res.status(500).json({ error: 'Internal server error', details: error.message });
  }
});

// Endpoint: Get detailed property information
app.get('/api/properties/:propertyId', async (req, res) => {
  try {
    const propertyId = req.params.propertyId;
    
    // Query for property details
    const propertyQuery = `
      SELECT 
        p.*,
        json_agg(json_build_object(
          'estimated_value', pv.estimated_value,
          'confidence_score', pv.confidence_score,
          'valuation_date', pv.valuation_date,
          'valuation_method', pv.valuation_method
        ) ORDER BY pv.valuation_date DESC) AS valuations
      FROM property p
      LEFT JOIN property_valuation pv ON p.id = pv.property_id
      WHERE p.property_id = $1
      GROUP BY p.id
    `;
    
    const propertyResult = await pool.query(propertyQuery, [propertyId]);
    
    if (propertyResult.rows.length === 0) {
      return res.status(404).json({ error: 'Property not found' });
    }
    
    // Format response
    const property = propertyResult.rows[0];
    
    res.json(property);
  } catch (error) {
    console.error(`Error fetching property ${req.params.propertyId}:`, error);
    res.status(500).json({ error: 'Internal server error', details: error.message });
  }
});

// Endpoint: Register a new user
app.post('/api/users/register', async (req, res) => {
  try {
    const { username, email, password } = req.body;
    
    // Validate inputs
    if (!username || !email || !password) {
      return res.status(400).json({ error: 'Username, email, and password are required' });
    }
    
    // Check if username or email already exists
    const checkQuery = `
      SELECT id FROM "user" WHERE username = $1 OR email = $2
    `;
    
    const checkResult = await pool.query(checkQuery, [username, email]);
    
    if (checkResult.rows.length > 0) {
      return res.status(409).json({ error: 'Username or email already exists' });
    }
    
    // Hash the password (in a real app, use bcrypt)
    const passwordHash = crypto.createHash('sha256').update(password).digest('hex');
    
    // Insert the new user
    const insertQuery = `
      INSERT INTO "user" (username, email, password_hash)
      VALUES ($1, $2, $3)
      RETURNING id, username, email, created_at
    `;
    
    const insertResult = await pool.query(insertQuery, [username, email, passwordHash]);
    
    // Generate an API key for the new user
    const apiKey = crypto.randomBytes(32).toString('hex');
    const apiKeyName = 'Default API Key';
    
    const apiKeyQuery = `
      INSERT INTO api_key (key, name, user_id, is_active)
      VALUES ($1, $2, $3, true)
      RETURNING id, key, name, created_at
    `;
    
    const apiKeyResult = await pool.query(apiKeyQuery, [apiKey, apiKeyName, insertResult.rows[0].id]);
    
    // Return the user and API key
    res.status(201).json({
      user: insertResult.rows[0],
      api_key: apiKeyResult.rows[0]
    });
  } catch (error) {
    console.error('Error registering user:', error);
    res.status(500).json({ error: 'Internal server error', details: error.message });
  }
});

// Endpoint: Login
app.post('/api/users/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    
    // Validate inputs
    if (!username || !password) {
      return res.status(400).json({ error: 'Username and password are required' });
    }
    
    // Hash the password
    const passwordHash = crypto.createHash('sha256').update(password).digest('hex');
    
    // Find the user
    const userQuery = `
      SELECT id, username, email, created_at
      FROM "user"
      WHERE username = $1 AND password_hash = $2
    `;
    
    const userResult = await pool.query(userQuery, [username, passwordHash]);
    
    if (userResult.rows.length === 0) {
      return res.status(401).json({ error: 'Invalid username or password' });
    }
    
    // Get the user's API keys
    const apiKeysQuery = `
      SELECT id, key, name, created_at, last_used, is_active
      FROM api_key
      WHERE user_id = $1 AND is_active = true
    `;
    
    const apiKeysResult = await pool.query(apiKeysQuery, [userResult.rows[0].id]);
    
    // Return user and API keys
    res.json({
      user: userResult.rows[0],
      api_keys: apiKeysResult.rows
    });
  } catch (error) {
    console.error('Error logging in:', error);
    res.status(500).json({ error: 'Internal server error', details: error.message });
  }
});

// Serve static index.html for all non-API routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Export the Express app
module.exports = app;