# BCBS Values Platform Workflow Automation

## Overview

This document describes the workflow automation system for the BCBS Values Platform. The workflow automation system provides reliable startup and monitoring of the web application and related services.

## Workflows

### WebApp Workflow

The primary workflow that starts the web application server.

**Command:** `./run_webapp.sh`  
**Port:** 5002  
**Description:** Web application server for BCBS Values Platform

#### Features:
- Automatic Python executable detection
- Fallback to Node.js server if Python is unavailable
- Intelligent script selection based on availability
- Server monitoring and health checks
- Detailed logging

### TestWorkflow

A workflow for running tests on the workflow automation system.

**Command:** `/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3 test_workflow.py`  
**Description:** Run workflow tests for BCBS Values Platform

## Scripts

### run_webapp.sh

The main script for starting the web application. This script:

1. Searches for a working Python executable
2. Selects the appropriate server script
3. Starts the server and monitors its health
4. Provides detailed logging
5. Falls back to a Node.js server if Python is unavailable

### simple_http_server.py

A simple HTTP server implementation that doesn't require external dependencies. This provides a reliable server option when Flask or other dependencies are not available.

### test_workflow.py

A test suite for validating the workflow automation system. It checks:

1. Python environment availability
2. Required files
3. Workflow configuration
4. Server startup capability
5. API functionality

### test_python_env.py

A diagnostic tool that checks the Python environment, including:

1. Python version and path
2. Available modules
3. Network port status
4. File system status
5. Flask availability

## Troubleshooting

If the WebApp workflow fails to start:

1. Check the server logs in `server.log`
2. Run the test workflow to identify issues: `/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3 test_workflow.py`
3. Check the Python environment: `/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3 test_python_env.py`
4. Verify that `run_webapp.sh` is executable: `chmod +x run_webapp.sh`
5. Verify that `simple_http_server.py` is executable: `chmod +x simple_http_server.py`

## Configuration

The workflow configuration is defined in `workflow_config.json`. This file contains the configuration for each workflow, including:

- Name and description
- Command to run
- Port to use
- Environment variables
- Dependencies
- Monitoring settings

## Best Practices

1. Always use `run_webapp.sh` to start the server, not direct Python commands
2. Check script permissions before running workflows
3. Run tests before making changes to the workflow system
4. Monitor server logs for errors
5. Keep the fallback server options available for reliability