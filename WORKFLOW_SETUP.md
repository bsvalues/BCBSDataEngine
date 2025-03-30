# Setting Up the Valuation API Workflow

This guide explains how to run the BCBS_Values Valuation API.

## Valuation API Options

We have two API implementations available:

1. **Simple Valuation API** (Flask-based):
   - Lightweight API that doesn't require database access
   - Provides core property valuation functionality
   - Runs on port 5002 by default

2. **Full Valuation API** (FastAPI-based):
   - Comprehensive API with database integration
   - Requires PostgreSQL database connection
   - Includes ETL status and agent management endpoints
   - Runs on port 8000 by default

## Starting the Simple Valuation API

You can start the simple API with the provided scripts:

### Using the Run Script (Easiest)

```bash
# Run the user-friendly script
./run_simple_api.sh
```

This script provides a nice overview of the API endpoints and starts the server.

### Using the Server Script Directly

For more control over the server configuration, you can use:

```bash
# Make the script executable (if needed)
chmod +x start_simple_api.sh

# Start the API
./start_simple_api.sh
```

This starts the Gunicorn server on port 5002 with 4 workers by default.

### Customizing Server Settings

You can customize the API server by setting environment variables:

```bash
# Example: Using 2 workers and debug log level
PORT=5002 WORKERS=2 LOG_LEVEL=debug ./start_simple_api.sh
```

Available options:
- `PORT`: The port to listen on (default: 5002)
- `WORKERS`: Number of Gunicorn worker processes (default: 4)
- `TIMEOUT`: Request timeout in seconds (default: 120)
- `LOG_LEVEL`: Logging level (default: info)

## Testing the Simple Valuation API

Once the API is running, you can test it using the provided test script:

```bash
# Make the script executable
chmod +x test_simple_api.sh

# Run the test (includes starting/stopping the API)
./test_simple_api.sh
```

The test script will:
1. Start the API server
2. Send a sample property valuation request
3. Display the results
4. Stop the API server

## Using the Full Valuation API

If you need the full API with database integration:

```bash
# Make sure the script is executable
chmod +x test_new_valuation_endpoint.sh

# Run the test (which includes starting the API)
./test_new_valuation_endpoint.sh
```

## API Endpoints

### Simple Valuation API (port 5002)

- `GET /api/health`: Health check endpoint
- `POST /api/valuation`: Generate a property valuation
- `GET /api/neighborhoods`: Get neighborhood quality ratings
- `GET /api/reference-points`: Get GIS reference points

### Full Valuation API (port 8000)

- `GET /api/valuations`: Get property valuations with filtering options
- `GET /api/valuations/{property_id}`: Get valuation for a specific property
- `POST /api/valuations`: Generate a new property valuation (requires API key)
- `GET /api/etl-status`: Get the current status of the ETL process
- `GET /api/agent-status`: Get the status of the BS Army of Agents

For more details, see the API documentation in the README.md file.