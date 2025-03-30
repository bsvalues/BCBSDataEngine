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

You can start the simple API either using the workflow or directly with the script:

### Using Replit Workflow (Recommended)

1. Click on the **Workflow** option in the Replit sidebar
2. Select the **SimpleValuationAPI** workflow
3. Click **Run**

This will start the Gunicorn server on port 5002.

### Using the Script Directly

Alternatively, you can run the API directly with the provided script:

```bash
# Make the script executable
chmod +x start_simple_api.sh

# Start the API
./start_simple_api.sh
```

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