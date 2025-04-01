#!/usr/bin/env bash
":"; //# comment; exec /usr/bin/env node "$0" "$@" || echo "Node.js not found, using bash fallback"; # -*- JavaScript -*-

/**
 * BCBS Direct Server - Special hybrid script that can run on both Node.js and bash
 * This script is designed to be run directly with bash if Node.js is not available
 */

// This script is designed as a hybrid script to provide fallback diagnostics
// when conventional environments fail. It can run in two ways:
// 1. As a Node.js script if Node.js is available
// 2. With the embedded bash fallback if Node.js is not available

// ==== Node.js Implementation ====
// This section only runs if the script is executed by Node.js

if (typeof process !== 'undefined' && process.versions && process.versions.node) {
  // We're running in Node.js
  const http = require('http');
  const os = require('os');
  
  // Server configuration
  const PORT = process.env.PORT || 5000;
  
  // Create a simple HTTP server
  const server = http.createServer((req, res) => {
    const timestamp = new Date().toISOString();
    console.log(`${timestamp} - ${req.method} ${req.url}`);
    
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>BCBS Direct Server</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 0 20px; }
          h1 { color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }
          h2 { color: #0066cc; margin-top: 30px; }
          pre { background: #f5f5f5; padding: 15px; overflow-x: auto; }
          .card { border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-bottom: 20px; }
          .status { padding: 5px 10px; border-radius: 4px; display: inline-block; font-weight: bold; }
          .warning { background-color: #fff3cd; color: #856404; }
        </style>
      </head>
      <body>
        <h1>BCBS Direct Server</h1>
        
        <div class='card'>
          <h2>System Status</h2>
          <p><span class='status warning'>DIRECT FALLBACK MODE</span> Running with direct Node.js server.</p>
          <p>This is the minimal direct server implementation for diagnostics.</p>
        </div>
        
        <div class='card'>
          <h2>System Information</h2>
          <pre>
Date: ${new Date().toString()}
Node.js: ${process.version}
Platform: ${process.platform}
Architecture: ${process.arch}
Hostname: ${os.hostname()}
</pre>
        </div>
        
        <div class='card'>
          <h2>Environment</h2>
          <pre>
PORT: ${process.env.PORT || 'Not set'}
DATABASE_URL: ${process.env.DATABASE_URL ? '[REDACTED]' : 'Not set'}
</pre>
        </div>
        
        <div class='card'>
          <h2>System Resources</h2>
          <pre>
Memory:
  Total: ${(os.totalmem() / (1024 * 1024 * 1024)).toFixed(2)} GB
  Free: ${(os.freemem() / (1024 * 1024 * 1024)).toFixed(2)} GB
  Load Average: ${os.loadavg().join(', ')}
</pre>
        </div>
        
        <div class='card'>
          <h2>Help Information</h2>
          <p>The server is currently running in direct fallback mode.</p>
          <p>To enable the full application, please ensure that the primary diagnostic servers can start correctly.</p>
        </div>
        
        <footer style='margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; text-align: center;'>
          <p>BCBS Direct Server (Node.js version)</p>
          <p>Generated: ${new Date().toString()}</p>
        </footer>
      </body>
      </html>
    `);
  });
  
  // Start the server
  server.listen(PORT, '0.0.0.0', () => {
    console.log(`Direct server running on http://0.0.0.0:${PORT}`);
    console.log(`Server started at: ${new Date().toISOString()}`);
  });
  
  // Exit gracefully on SIGINT
  process.on('SIGINT', () => {
    console.log('\nShutting down server');
    server.close(() => {
      console.log('Server closed');
      process.exit(0);
    });
  });
  
  // Don't proceed to the bash part
  return;
}

// ==== Bash Fallback Implementation ====
// This section only runs if executed directly by bash when Node.js is not available
// It's written in a way that bash will interpret it as commands

echo "Running direct server bash fallback mode"
echo "Date: $(date)"

# Try to determine a good port to use
PORT=${PORT:-5000}
echo "Using port: $PORT"

# Generate static HTML
cat << EOF
<!DOCTYPE html>
<html>
<head>
    <title>BCBS Direct Server (Bash)</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        h1 { color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }
        h2 { color: #0066cc; margin-top: 30px; }
        pre { background: #f5f5f5; padding: 15px; overflow-x: auto; }
        .card { border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-bottom: 20px; }
        .status { padding: 5px 10px; border-radius: 4px; display: inline-block; font-weight: bold; }
        .warning { background-color: #fff3cd; color: #856404; }
    </style>
</head>
<body>
    <h1>BCBS Direct Server (Bash)</h1>
    
    <div class='card'>
        <h2>System Status</h2>
        <p><span class='status warning'>BASH FALLBACK MODE</span> Running with bash script server.</p>
        <p>This is the absolute minimal bash fallback implementation for diagnostics.</p>
    </div>
    
    <div class='card'>
        <h2>System Information</h2>
        <pre>
Date: $(date)
Bash: $(bash --version | head -n 1)
</pre>
    </div>
    
    <div class='card'>
        <h2>Environment</h2>
        <pre>
PORT: $PORT
PATH: $PATH
</pre>
    </div>
    
    <div class='card'>
        <h2>Help Information</h2>
        <p>The server is currently running in bash fallback mode, which means both Node.js and Python servers failed to start.</p>
        <p>This is the most minimal diagnostic server possible.</p>
    </div>
    
    <footer style='margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; text-align: center;'>
        <p>BCBS Direct Server (Bash version)</p>
        <p>Generated: $(date)</p>
    </footer>
</body>
</html>
EOF

# Successfully exit the script for bash mode
exit 0