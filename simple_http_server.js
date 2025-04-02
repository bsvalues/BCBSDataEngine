/**
 * Simple HTTP server to redirect requests to the appropriate backend services
 * - Flask web application on port 5000
 * - FastAPI REST API on port 8000
 */
const http = require('http');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration
const PORT = process.env.PORT || 3000;
const FLASK_PORT = 5001;
const FASTAPI_PORT = 8000;

// Start the Flask web application
console.log('Starting Flask web application...');
const flaskProcess = spawn('python', ['main.py', '--web'], {
  detached: true,
  stdio: 'pipe'
});

// Pipe Flask output to console and a log file
flaskProcess.stdout.on('data', (data) => {
  console.log(`[Flask] ${data.toString().trim()}`);
});

flaskProcess.stderr.on('data', (data) => {
  console.error(`[Flask Error] ${data.toString().trim()}`);
});

flaskProcess.on('close', (code) => {
  console.log(`Flask process exited with code ${code}`);
});

// Start the FastAPI server
console.log('Starting FastAPI server...');
const apiProcess = spawn('python', ['run_api.py'], {
  detached: true,
  stdio: 'pipe'
});

// Pipe API output to console and a log file
apiProcess.stdout.on('data', (data) => {
  console.log(`[FastAPI] ${data.toString().trim()}`);
});

apiProcess.stderr.on('data', (data) => {
  console.error(`[FastAPI Error] ${data.toString().trim()}`);
});

apiProcess.on('close', (code) => {
  console.log(`FastAPI process exited with code ${code}`);
});

// Create HTTP server that redirects to the appropriate service
const server = http.createServer((req, res) => {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');
  res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type');
  
  // Respond to OPTIONS method for CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }
  
  // Check if the request is for the API (starts with /api)
  if (req.url.startsWith('/api')) {
    // Proxy request to FastAPI
    const options = {
      hostname: 'localhost',
      port: FASTAPI_PORT,
      path: req.url,
      method: req.method,
      headers: req.headers
    };
    
    console.log(`Proxying API request to FastAPI: ${req.url}`);
    
    const apiReq = http.request(options, (apiRes) => {
      res.writeHead(apiRes.statusCode, apiRes.headers);
      apiRes.pipe(res);
    });
    
    apiReq.on('error', (e) => {
      console.error(`Error proxying to API: ${e.message}`);
      res.writeHead(503, { 'Content-Type': 'text/plain' });
      res.end('API server unavailable. Please try again later.');
    });
    
    if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
      req.pipe(apiReq);
    } else {
      apiReq.end();
    }
  } else {
    // Proxy request to Flask web application
    const options = {
      hostname: 'localhost',
      port: FLASK_PORT,
      path: req.url,
      method: req.method,
      headers: req.headers
    };
    
    console.log(`Proxying web request to Flask: ${req.url}`);
    
    const flaskReq = http.request(options, (flaskRes) => {
      res.writeHead(flaskRes.statusCode, flaskRes.headers);
      flaskRes.pipe(res);
    });
    
    flaskReq.on('error', (e) => {
      console.error(`Error proxying to Flask: ${e.message}`);
      res.writeHead(503, { 'Content-Type': 'text/plain' });
      res.end('Web application unavailable. Please try again later.');
    });
    
    if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
      req.pipe(flaskReq);
    } else {
      flaskReq.end();
    }
  }
});

// Start proxy server
server.listen(PORT, () => {
  console.log(`Gateway server running at http://localhost:${PORT}/`);
  console.log(`Proxying to Flask application at http://localhost:${FLASK_PORT}/`);
  console.log(`Proxying to FastAPI server at http://localhost:${FASTAPI_PORT}/api`);
  console.log('\nPress Ctrl+C to stop the server.');
});

// Handle process termination
process.on('SIGINT', () => {
  console.log('Shutting down services...');
  
  // Kill child processes
  process.kill(-flaskProcess.pid);
  process.kill(-apiProcess.pid);
  
  process.exit(0);
});