/**
 * Simple HTTP server for BCBS Values API diagnostic
 * This is a fallback if Python servers fail to start
 */
const http = require('http');
const fs = require('fs');
const os = require('os');

// Configuration
const PORT = process.env.PORT || 5000;

// Create a server
const server = http.createServer((req, res) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  
  // Basic routing
  if (req.url === '/' || req.url === '/index.html') {
    // Try to read index.html file
    try {
      const content = fs.readFileSync('index.html', 'utf8');
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(content);
    } catch (err) {
      // If index.html doesn't exist, generate HTML
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(generateDiagnosticHtml());
    }
  } else if (req.url === '/api/health') {
    // Health check endpoint
    res.writeHead(200, { 'Content-Type': 'application/json' });
    const healthData = {
      status: 'minimal_diagnostic_mode',
      message: 'Running in minimal Node.js diagnostic mode',
      timestamp: new Date().toISOString(),
      serverInfo: {
        nodeVersion: process.version,
        platform: process.platform
      }
    };
    res.end(JSON.stringify(healthData, null, 2));
  } else {
    // 404 Not Found
    res.writeHead(404, { 'Content-Type': 'text/html' });
    res.end(`
      <html>
        <head>
          <title>404 - Not Found</title>
          <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 650px; margin: 40px auto; padding: 0 10px; }
            h1 { color: #e74c3c; }
            a { color: #3498db; }
          </style>
        </head>
        <body>
          <h1>404 - Not Found</h1>
          <p>The requested resource "${req.url}" was not found on this server.</p>
          <p><a href="/">Return to home page</a></p>
        </body>
      </html>
    `);
  }
});

// Generate diagnostic HTML
function generateDiagnosticHtml() {
  // Filter environment variables to only show safe ones
  const envVars = {};
  for (const key in process.env) {
    if (key.startsWith('DATABASE') || key.startsWith('PG') || 
        key.startsWith('NODE') || key === 'PORT') {
      // Redact passwords
      if (key === 'DATABASE_URL') {
        // Try to redact password from DATABASE_URL
        let value = process.env[key] || '';
        envVars[key] = value.replace(/:[^:]*@/, ':[REDACTED]@');
      } else {
        envVars[key] = process.env[key];
      }
    }
  }

  // Basic HTML content
  return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>BCBS Values Minimal Diagnostic</title>
      <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 40px auto; padding: 0 10px; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; margin-top: 30px; }
        pre { background: #f5f5f5; padding: 15px; overflow-x: auto; border-radius: 4px; }
        .card { border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .error { color: #e74c3c; }
        .warning { color: #f39c12; }
      </style>
    </head>
    <body>
      <h1>BCBS Values Minimal Diagnostic</h1>
      <p>This is the minimal Node.js diagnostic page for the BCBS Values application.</p>
      
      <div class="card">
        <h2>System Information</h2>
        <table>
          <tr><td>Date and Time:</td><td>${new Date().toISOString()}</td></tr>
          <tr><td>Node.js Version:</td><td>${process.version}</td></tr>
          <tr><td>Platform:</td><td>${process.platform}</td></tr>
          <tr><td>Hostname:</td><td>${os.hostname()}</td></tr>
          <tr><td>Server Port:</td><td>${PORT}</td></tr>
        </table>
      </div>
      
      <div class="card">
        <h2>Diagnostic Status</h2>
        <p><span class="warning">⚠️ WARNING:</span> Running in minimal Node.js diagnostic mode.</p>
        <p>The full application functionality is not available.</p>
      </div>
      
      <div class="card">
        <h2>Environment Variables</h2>
        <pre>${JSON.stringify(envVars, null, 2)}</pre>
        <p><i>Note: Secret values are hidden for security reasons.</i></p>
      </div>
      
      <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; text-align: center;">
        <p>BCBS Values Minimal Diagnostic Server</p>
        <p>Generated at: ${new Date().toISOString()}</p>
      </footer>
    </body>
    </html>
  `;
}

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Minimal diagnostic server running on http://0.0.0.0:${PORT}`);
  console.log(`Server started at: ${new Date().toISOString()}`);
});