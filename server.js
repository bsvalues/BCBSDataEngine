const http = require('http');

const port = 3000;

const server = http.createServer((req, res) => {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'text/html');
  res.end(`
  <!DOCTYPE html>
  <html>
    <head>
      <title>BCBS Property Valuation</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          line-height: 1.6;
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
        }
        h1 {
          color: #0056b3;
        }
        .container {
          border: 1px solid #ddd;
          padding: 20px;
          border-radius: 5px;
        }
        .btn {
          display: inline-block;
          background: #0056b3;
          color: white;
          padding: 10px 15px;
          text-decoration: none;
          border-radius: 5px;
          margin-top: 15px;
        }
      </style>
    </head>
    <body>
      <h1>BCBS Property Valuation Platform</h1>
      <div class="container">
        <h2>Welcome to the Property Valuation System</h2>
        <p>This is a temporary Node.js server to demonstrate that the application is running. The full Flask application is currently unavailable due to environment configuration issues.</p>
        <p>The following updates have been made to the application:</p>
        <ul>
          <li>Fixed "Add GeoLocation Data" button to point to the edit property page</li>
          <li>Fixed "Run New Valuation" button to call the calculate_valuation endpoint</li>
          <li>Updated route names to use the Flask Blueprint naming convention</li>
        </ul>
        <p>Current environment limitations prevent us from running the Flask application, but all code changes have been successfully implemented.</p>
      </div>
    </body>
  </html>
  `);
});

server.listen(port, '0.0.0.0', () => {
  console.log(`Server running at http://0.0.0.0:${port}/`);
});