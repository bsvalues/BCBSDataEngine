/**
 * Direct server for BCBS Values
 * This is a basic Node.js server to display diagnostic information
 */
const http = require('http');
const fs = require('fs');
const os = require('os');

// Port configuration
const port = process.env.PORT || 5000;

// Create the server
const server = http.createServer((req, res) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  
  if (req.url === '/' || req.url === '/index.html') {
    // Try to serve index.html if it exists
    try {
      if (fs.existsSync('index.html')) {
        const content = fs.readFileSync('index.html', 'utf8');
        res.writeHead(200, {'Content-Type': 'text/html'});
        res.end(content);
        return;
      }
    } catch (error) {
      console.error('Error reading index.html:', error);
    }
    
    // Otherwise, serve generated HTML
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.end(generateHTML());
  } else if (req.url === '/api/health') {
    // Health endpoint
    res.writeHead(200, {'Content-Type': 'application/json'});
    const healthData = {
      status: 'diagnostic_mode',
      message: 'Running in Node.js diagnostic mode',
      timestamp: new Date().toISOString(),
      nodejs_version: process.version,
      platform: process.platform,
      hostname: os.hostname()
    };
    res.end(JSON.stringify(healthData, null, 2));
  } else {
    // 404 Not Found
    res.writeHead(404, {'Content-Type': 'text/html'});
    res.end(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>404 - Not Found</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 650px; margin: 40px auto; padding: 0 10px; }
          h1 { color: #e74c3c; }
          a { color: #3498db; }
        </style>
      </head>
      <body>
        <h1>404 - Not Found</h1>
        <p>The requested resource was not found on this server.</p>
        <p><a href="/">Return to home page</a></p>
      </body>
      </html>
    `);
  }
});

function generateHTML() {
  // Get filtered environment variables
  const envVars = {};
  for (const key in process.env) {
    if (key.startsWith('DATABASE') || key.startsWith('PG') || key === 'PORT') {
      if (key === 'DATABASE_URL') {
        // Mask the password in DATABASE_URL
        let value = process.env[key] || '';
        if (value.includes('@') && value.includes('://')) {
          try {
            const parts = value.split('@');
            const prefix = parts[0].split('://');
            if (prefix.length > 1) {
              const maskedUrl = `${prefix[0]}://${prefix[1].split(':')[0]}:****@${parts[1]}`;
              envVars[key] = maskedUrl;
            } else {
              envVars[key] = '[REDACTED]';
            }
          } catch (e) {
            envVars[key] = '[REDACTED]';
          }
        } else {
          envVars[key] = '[REDACTED]';
        }
      } else if (key.includes('PASSWORD')) {
        envVars[key] = '[REDACTED]';
      } else {
        envVars[key] = process.env[key];
      }
    }
  }

  return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>BCBS Values Diagnostic</title>
      <style>
        body { 
          font-family: Arial, sans-serif; 
          line-height: 1.6; 
          color: #333; 
          max-width: 900px; 
          margin: 40px auto; 
          padding: 0 20px; 
          background-color: #f9f9f9;
        }
        header {
          background-color: #2c3e50;
          color: white;
          padding: 20px;
          border-radius: 5px;
          margin-bottom: 30px;
          box-shadow: 0 3px 5px rgba(0,0,0,0.1);
        }
        h1 { 
          margin: 0;
          color: white;
        }
        h2 { 
          color: #3498db; 
          margin-top: 30px; 
          border-bottom: 2px solid #eee;
          padding-bottom: 10px;
        }
        .header-subtitle {
          margin-top: 10px;
          color: #ecf0f1;
          font-size: 1.1em;
        }
        pre { 
          background: #f5f5f5; 
          padding: 15px; 
          overflow-x: auto; 
          border-radius: 4px; 
          border: 1px solid #ddd;
        }
        .card { 
          background-color: white;
          border: 1px solid #ddd; 
          border-radius: 5px; 
          padding: 25px; 
          margin-bottom: 25px; 
          box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
        }
        .error { color: #e74c3c; }
        .warning { color: #f39c12; }
        .success { color: #27ae60; }
        table { 
          width: 100%; 
          border-collapse: collapse; 
          margin-bottom: 20px; 
        }
        th, td { 
          border: 1px solid #ddd; 
          padding: 12px; 
          text-align: left; 
        }
        th { 
          background-color: #f8f9fa; 
          font-weight: bold;
        }
        tr:nth-child(even) {
          background-color: #f8f9fa;
        }
        .dashboard-section {
          margin-top: 40px;
        }
        .status-badge {
          display: inline-block;
          padding: 5px 10px;
          border-radius: 3px;
          font-size: 0.85em;
          font-weight: bold;
        }
        .status-ok {
          background-color: #d4edda;
          color: #155724;
        }
        .status-warning {
          background-color: #fff3cd;
          color: #856404;
        }
        .status-error {
          background-color: #f8d7da;
          color: #721c24;
        }
        .footer {
          margin-top: 50px;
          padding-top: 20px;
          border-top: 1px solid #eee;
          text-align: center;
          color: #7f8c8d;
        }
      </style>
    </head>
    <body>
      <header>
        <h1>BCBS Values Diagnostic</h1>
        <div class="header-subtitle">System Status & Diagnostic Information</div>
      </header>

      <div class="card">
        <h2>System Status</h2>
        <p><span class="status-badge status-warning">DIAGNOSTIC MODE</span> This is a diagnostic page for the BCBS Values application.</p>
        <p>The application is currently running in diagnostic mode. This means that the full application functionality may not be available. This page is designed to help diagnose any issues with the application environment.</p>
      </div>
      
      <div class="card">
        <h2>Environment Information</h2>
        <table>
          <tr>
            <th>Component</th>
            <th>Status</th>
            <th>Details</th>
          </tr>
          <tr>
            <td>Node.js Runtime</td>
            <td><span class="status-badge status-ok">AVAILABLE</span></td>
            <td>${process.version}</td>
          </tr>
          <tr>
            <td>Operating System</td>
            <td><span class="status-badge status-ok">AVAILABLE</span></td>
            <td>${os.platform()} ${os.release()}</td>
          </tr>
          <tr>
            <td>Database</td>
            <td><span class="status-badge status-ok">AVAILABLE</span></td>
            <td>PostgreSQL database is available via DATABASE_URL environment variable.</td>
          </tr>
          <tr>
            <td>Application Server</td>
            <td><span class="status-badge status-warning">DIAGNOSTIC MODE</span></td>
            <td>Running in Node.js diagnostic mode.</td>
          </tr>
          <tr>
            <td>Memory</td>
            <td><span class="status-badge status-ok">AVAILABLE</span></td>
            <td>Total: ${Math.round(os.totalmem() / (1024 * 1024))} MB, Free: ${Math.round(os.freemem() / (1024 * 1024))} MB</td>
          </tr>
        </table>
      </div>

      <div class="card">
        <h2>Environment Variables</h2>
        <pre>${JSON.stringify(envVars, null, 2)}</pre>
        <p><i>Note: Secret values are redacted for security.</i></p>
      </div>

      <div class="dashboard-section">
        <div class="card">
          <h2>BCBS Values Dashboard</h2>
          <p>The BCBS Values application includes several dashboards for monitoring and analyzing real estate property data.</p>
          
          <h3>Dashboard Features</h3>
          <ul>
            <li><strong>Property Search and Filtering</strong> - Search and filter properties by various criteria</li>
            <li><strong>Valuation Analytics</strong> - View detailed valuation data and analytics</li>
            <li><strong>Agent Status Monitoring</strong> - Monitor the status of the BS Army of Agents</li>
            <li><strong>ETL Pipeline Status</strong> - Monitor the status of the data extraction, transformation, and loading pipeline</li>
            <li><strong>What-If Analysis</strong> - Perform what-if analysis to see how changes in property features affect valuations</li>
          </ul>
          
          <p><em>Note: Dashboard features are not available in diagnostic mode.</em></p>
        </div>
      </div>

      <div class="footer">
        <p>BCBS Values Diagnostic Page</p>
        <p>Generated on: <span id="date-generated">${new Date().toLocaleDateString()}</span></p>
      </div>
    </body>
    </html>
  `;
}

// Start the server
server.listen(port, '0.0.0.0', () => {
  console.log(`BCBS Values Diagnostic Server running at http://0.0.0.0:${port}/`);
  console.log(`Server started at: ${new Date().toISOString()}`);
});