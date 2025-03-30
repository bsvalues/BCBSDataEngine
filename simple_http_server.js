// Simple HTTP server using Node.js
const http = require('http');

// Create sample data
const sampleProperties = [
  {
    "property_id": "BENT-12345",
    "address": "123 Main St, Richland, WA 99352",
    "estimated_value": 345000,
    "confidence_score": 0.85,
    "last_updated": new Date().toISOString().split('T')[0],
    "bedrooms": 3,
    "bathrooms": 2,
    "square_feet": 1850,
    "year_built": 2005,
    "property_type": "Single Family"
  },
  {
    "property_id": "BENT-67890",
    "address": "456 Oak Ave, Kennewick, WA 99336",
    "estimated_value": 289000,
    "confidence_score": 0.78,
    "last_updated": new Date().toISOString().split('T')[0],
    "bedrooms": 2,
    "bathrooms": 1.5,
    "square_feet": 1550,
    "year_built": 1995,
    "property_type": "Single Family"
  },
  {
    "property_id": "BENT-23456",
    "address": "789 Pine Ln, Pasco, WA 99301",
    "estimated_value": 425000,
    "confidence_score": 0.92,
    "last_updated": new Date().toISOString().split('T')[0],
    "bedrooms": 4,
    "bathrooms": 3,
    "square_feet": 2200,
    "year_built": 2018,
    "property_type": "Single Family"
  }
];

const validationResults = {
  "validation_passed": true,
  "total_records": 150,
  "valid_records": 148,
  "invalid_records": 2,
  "validation_timestamp": new Date().toISOString().replace('T', ' ').substring(0, 19),
  "validation_results": {
    "missing_values": {
      "count": 1,
      "details": ["Property BENT-45678 missing square_feet value"]
    },
    "invalid_values": {
      "count": 1,
      "details": ["Property BENT-98765 has invalid year_built (value: 3005)"]
    }
  }
};

// Create the HTTP server
const server = http.createServer((req, res) => {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Handle OPTIONS requests for CORS preflight
  if (req.method === 'OPTIONS') {
    res.statusCode = 200;
    res.end();
    return;
  }
  
  // Parse the URL
  const url = new URL(req.url, `http://${req.headers.host}`);
  const path = url.pathname;
  
  console.log(`Request received: ${req.method} ${path}`);
  
  // Set content type to JSON by default
  res.setHeader('Content-Type', 'application/json');
  
  // Routes
  if (path === '/' || path === '/index') {
    // API root
    const response = {
      status: "running",
      message: "BCBS Values API is running!",
      version: "1.0",
      endpoints: [
        "/api/properties",
        "/api/validation"
      ]
    };
    res.statusCode = 200;
    res.end(JSON.stringify(response));
  } 
  else if (path === '/api/properties' || path === '/api/valuations') {
    // Return all properties
    res.statusCode = 200;
    res.end(JSON.stringify(sampleProperties));
  } 
  else if (path.startsWith('/api/property/') || path.startsWith('/api/valuation/')) {
    // Get a specific property by ID
    const property_id = path.split('/').pop();
    
    // Find the property
    const property = sampleProperties.find(p => p.property_id === property_id);
    
    if (property) {
      res.statusCode = 200;
      res.end(JSON.stringify(property));
    } else {
      res.statusCode = 404;
      res.end(JSON.stringify({ error: "Property not found" }));
    }
  }
  else if (path === '/api/validation' || path === '/api/etl-status') {
    // Return validation results
    res.statusCode = 200;
    res.end(JSON.stringify(validationResults));
  }
  else {
    // Not found
    res.statusCode = 404;
    res.end(JSON.stringify({ error: "Resource not found" }));
  }
});

// Start the server
const PORT = process.env.PORT || 5000;
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running at http://0.0.0.0:${PORT}/`);
  console.log(`API endpoints:`);
  console.log(`  - GET /api/properties`);
  console.log(`  - GET /api/property/:id`);
  console.log(`  - GET /api/validation`);
});