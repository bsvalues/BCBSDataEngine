# BCBS Values Workflow Setup Guide

This document provides instructions for setting up and running the BCBS Values application workflows in Replit.

## Available Workflows

The project has several workflows configured:

1. **API** - The main FastAPI REST API server on port 5000
2. **WebApp** - The Flask web application UI on port 5001
3. **ValuationAPI** - Specialized valuation API service
4. **EnhancedAPI** - Advanced valuation API with GIS features

## Setting Up Environment Variables

Before running the workflows, you should set up the required environment variables.

### API Key Setup

The API requires authentication using an API key. You can set this up using:

```bash
# Set API key in environment
./set_api_key.sh your_secret_key

# For testing, you can use the sample key
./set_api_key.sh sample_test_key
```

### Database Configuration

The application uses PostgreSQL. Make sure your database is properly configured:

```bash
# Check database status
echo $DATABASE_URL

# For local testing, the URL should look like:
# postgresql://username:password@localhost:5432/bcbs_values
```

### Web Application Setup

The web application requires a session secret:

```bash
# Set a secure session secret
export SESSION_SECRET="your_secure_secret_key"
```

## Starting Workflows

To start a workflow, use the Replit workflow system or run the appropriate script:

### API Server

```bash
# Start via workflow
# or
./start_api_server.sh
```

### Web Application

```bash
# Start via workflow
# or
python start_web_app.py
```

## Testing the API

You can test the API using the included script:

```bash
# Make the script executable
chmod +x test_api_with_auth.sh

# Run the tests
./test_api_with_auth.sh
```

## Workflow Configuration Files

Workflow configurations are stored in:

- `.replit.workflow/API.json`
- `.replit.workflow/WebApp.json`
- `.replit.workflow/ValuationAPI.json`
- `.replit.workflow/EnhancedAPI.json`

These files define how each workflow operates, what ports are used, and what environment variables are required.

## Advanced Configuration

For more advanced configuration needs, edit the workflow JSON files in the `.replit.workflow/` directory.