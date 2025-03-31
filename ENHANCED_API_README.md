# Enhanced API Documentation

## Overview

The BCBS Values API provides programmatic access to property valuation data, ETL pipeline status, and valuation agent monitoring. This document describes the enhanced API features including:

- Token-based authentication
- Advanced property valuation endpoints
- ETL pipeline status monitoring
- Detailed agent status metrics

## Authentication

The API supports two authentication methods:

1. **API Key Authentication** (legacy)
   - Include the API key in the `X-API-Key` header or as an `api_key` query parameter
   - Example: `curl -H "X-API-Key: your_api_key" https://api.bcbsvalues.com/api/valuations`

2. **JWT Token Authentication** (recommended)
   - Request a token using the `/api/auth/token` endpoint
   - Include the token in the `Authorization` header with the `Bearer` prefix
   - Example: `curl -H "Authorization: Bearer your_token" https://api.bcbsvalues.com/api/valuations`

### Obtaining a JWT Token

```
POST /api/auth/token
```

Request body:
```json
{
  "agent_id": "your_agent_id",
  "agent_type": "your_agent_type",
  "api_key": "your_api_key"
}
```

Response:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 86400,
  "token_type": "Bearer"
}
```

## Enhanced Property Valuations

The enhanced valuations endpoint provides advanced filtering, sorting, and pagination capabilities.

```
GET /api/valuations
```

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| method | string | Filter by valuation method (e.g., 'enhanced_regression', 'lightgbm') |
| min_confidence | float | Minimum confidence score (0.0-1.0) |
| after_date | string | Only valuations after this date (ISO format) |
| before_date | string | Only valuations before this date (ISO format) |
| property_id | string | Filter by property ID |
| neighborhood | string | Filter by neighborhood |
| page | integer | Page number for pagination (default: 1) |
| limit | integer | Results per page (default: 20, max: 100) |
| sort_by | string | Field to sort by (default: 'valuation_date') |
| sort_dir | string | Sort direction ('asc' or 'desc', default: 'desc') |

### Example

```
GET /api/valuations?method=enhanced_regression&min_confidence=0.8&sort_by=estimated_value&sort_dir=desc&limit=5
```

Response:
```json
{
  "valuations": [
    {
      "valuation_id": 123,
      "property_id": "PROP-12345",
      "address": "123 Main St",
      "city": "Anytown",
      "state": "WA",
      "zip_code": "98101",
      "neighborhood": "Downtown",
      "estimated_value": 500000,
      "valuation_date": "2025-03-15T14:30:00Z",
      "valuation_method": "enhanced_regression",
      "confidence_score": 0.92,
      "model_metrics": {
        "adj_r2": 0.85,
        "rmse": 12500,
        "mae": 9500
      },
      "gis_features": {
        "school_district_impact": 0.05,
        "flood_zone_impact": -0.02
      }
    },
    ...
  ],
  "page": 1,
  "limit": 5,
  "total": 120,
  "pages": 24
}
```

## ETL Pipeline Status

Monitor the status of ETL jobs with filtering by job type, status, and timeframe.

```
GET /api/etl-status
```

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| job_type | string | Filter by job type |
| status | string | Filter by status |
| timeframe | string | Filter by timeframe ('today', 'yesterday', 'this_week', 'last_week', 'this_month') |
| limit | integer | Maximum number of jobs to return (default: 20, max: 100) |

### Example

```
GET /api/etl-status?timeframe=this_week&status=completed
```

Response:
```json
{
  "jobs": [
    {
      "id": 45,
      "job_type": "property_import",
      "status": "completed",
      "start_time": "2025-03-15T10:00:00Z",
      "end_time": "2025-03-15T10:15:30Z",
      "progress": 1.0,
      "records_processed": 500,
      "records_total": 500,
      "source": "county_data_feed",
      "message": "Successfully imported 500 properties",
      "error": null,
      "duration_seconds": 930
    },
    ...
  ],
  "stats": {
    "total_jobs": 10,
    "completed_jobs": 10,
    "failed_jobs": 0,
    "running_jobs": 0,
    "pending_jobs": 0,
    "total_records_processed": 5230,
    "average_progress": 1.0
  },
  "health": {
    "status": "healthy",
    "pipeline_active": true,
    "last_successful_job": "2025-03-15T10:15:30Z"
  },
  "timeframe": "this_week",
  "timestamp": "2025-03-15T14:30:00Z"
}
```

## Agent Status

Monitor the status of valuation agents with detailed metrics and filtering.

```
GET /api/agent-status
```

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| agent_type | string | Filter by agent type (e.g., 'regression', 'ensemble', 'gis') |
| status | string | Filter by status (e.g., 'idle', 'processing', 'error') |
| active_only | boolean | If set to 'true', only return active agents |

### Example

```
GET /api/agent-status?agent_type=regression&active_only=true
```

Response:
```json
{
  "agents": [
    {
      "id": 1,
      "name": "Enhanced Regression Agent",
      "agent_type": "regression",
      "description": "Enhanced regression model for property valuation",
      "status": "idle",
      "is_active": true,
      "version": "1.2.0",
      "created_at": "2025-01-15T10:00:00Z",
      "last_active": "2025-03-15T14:25:00Z",
      "success_rate": 0.98,
      "performance_metrics": {
        "recent_valuations": 10,
        "average_confidence": 0.91,
        "methods_used": ["enhanced_regression", "ridge_regression"]
      },
      "configuration": {
        "features_enabled": ["neighborhood_trends", "gis_features"],
        "max_comparables": 20
      },
      "latest_log": {
        "level": "info",
        "message": "Processed valuation request for property PROP-12345",
        "timestamp": "2025-03-15T14:25:00Z",
        "details": {
          "property_id": "PROP-12345",
          "execution_time_ms": 250
        }
      }
    },
    ...
  ],
  "count": 3,
  "timestamp": "2025-03-15T14:30:00Z",
  "metrics": {
    "total_agents": 3,
    "active_agents": 3,
    "idle_agents": 2,
    "processing_agents": 1,
    "error_agents": 0,
    "system_health": "healthy"
  }
}
```

## Agent Logs

Retrieve detailed logs for a specific agent.

```
GET /api/agent-logs/{agent_id}
```

Response:
```json
{
  "agent_id": "1",
  "agent_name": "Enhanced Regression Agent",
  "logs": [
    {
      "level": "info",
      "message": "Processed valuation request for property PROP-12345",
      "timestamp": "2025-03-15T14:25:00Z"
    },
    {
      "level": "info",
      "message": "Processed valuation request for property PROP-54321",
      "timestamp": "2025-03-15T14:20:00Z"
    },
    ...
  ],
  "count": 100
}
```

## Error Handling

All API endpoints return standard HTTP status codes:

- 200 OK: Request was successful
- 400 Bad Request: Invalid parameters
- 401 Unauthorized: Missing or invalid authentication
- 404 Not Found: Requested resource not found
- 500 Internal Server Error: Server-side error

Error responses include a JSON object with an error message:

```json
{
  "error": "Invalid API key"
}
```

## Testing

Use the included test scripts to verify API functionality:

```bash
# Start the API server
./start_api_server.sh

# Run the API tests
./test_enhanced_api.sh

# Stop the API server
./stop_api_server.sh
```

The test script will verify all enhanced API endpoints and report any issues.