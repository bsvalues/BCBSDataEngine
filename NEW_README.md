# BCBS Values Platform

Welcome to the BCBS Values Platform. This platform is an advanced real estate valuation and exploration system that provides intelligent, interactive property insights through cutting-edge technology.

## Core Features

- **Advanced Property Valuation**: Leveraging AI and ML technologies to provide accurate property valuations
- **Interactive Dashboards**: Visualize property data with interactive charts and tables
- **What-If Analysis**: Test different scenarios and parameters to see their impact on property values
- **Agent-Based Infrastructure**: Monitor and manage the BS Army of Agents for property assessment

## Architecture

The platform is built with a modular architecture across multiple layers:

1. **Data Acquisition Layer**: ETL pipelines for property data collection
2. **Persistence Layer**: PostgreSQL database for reliable data storage
3. **Processing Layer**: Valuation engine with BS Army of Agents
4. **Presentation Layer**: Web interface and API endpoints

## Directory Structure

- `/dashboard.html`: Main property valuation dashboard
- `/what-if-analysis.html`: Interface for scenario testing
- `/agent-dashboard.html`: Monitor agent status and performance
- `/api/`: RESTful API endpoints
- `/etl/`: Data acquisition scripts
- `/valuation/`: Property valuation engine

## Running the Application

### Method 1: Using the Diagnostic Script

Run the diagnostic script which will automatically find the best runtime environment:

```bash
./run_diagnosis.sh
```

### Method 2: Using Python Server

If you have Python 3.11+ installed:

```bash
python3 simple_python_server.py
```

### Method 3: Using Node.js Server

If you have Node.js installed:

```bash
node server.js
```

### Access the Application

Once the server is running, visit:
- http://localhost:5000/ - Home page
- http://localhost:5000/dashboard.html - Property valuation dashboard
- http://localhost:5000/what-if-analysis.html - Scenario testing
- http://localhost:5000/agent-dashboard.html - Agent monitoring

## Technical Requirements

- Python 3.11+ or Node.js
- Web browser with JavaScript enabled
- PostgreSQL database (for full functionality)

## Data Sources

The platform is designed to work with Benton County, WA property data, including neighborhoods such as:
- West Richland
- Kennewick
- Pasco
- Finley
- Richland