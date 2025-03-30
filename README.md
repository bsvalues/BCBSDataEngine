# BCBS Values: Real Estate Valuation System

<!-- Logo or banner image would go here -->

A sophisticated real estate valuation platform specifically for Benton County, Washington, featuring advanced agent management, ETL pipeline processing, and interactive dashboard visualization. The system provides accurate property valuations backed by machine learning to support real estate market analysis and decision-making.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Key Modules](#key-modules)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Workflow Setup](#workflow-setup)
- [Usage Guide](#usage-guide)
  - [Running the ETL Pipeline](#running-the-etl-pipeline)
  - [Accessing API Endpoints](#accessing-api-endpoints)
  - [Using the Dashboard](#using-the-dashboard)
  - [The Agent Dashboard](#the-agent-dashboard)
  - [What-If Analysis](#what-if-analysis)
- [CI/CD and Testing](#cicd-and-testing)
- [Contributing](#contributing)
- [Feedback & Issues](#feedback--issues)
- [License](#license)

## Overview

BCBS Values is a comprehensive system for valuing real estate properties in Benton County, Washington. It combines data from multiple sources including MLS listings, NARRPR data, and PACS (Property Assessment & Collection System) to provide accurate, data-driven property valuations.

The system employs both basic and advanced machine learning-based valuation models to estimate property values based on various features including square footage, bedrooms, bathrooms, property age, and GIS spatial data.

## Key Features

- **Multi-source ETL Pipeline**: Extracts, transforms, and loads data from MLS, NARRPR, and PACS sources
- **Intelligent Valuation Engine**: Uses multiple modeling approaches including:
  - Linear regression with advanced feature engineering
  - Gradient boosting with LightGBM for superior prediction accuracy
  - Ensemble methods combining multiple model types
- **Advanced GIS Integration**: Incorporates geographic and spatial data for location-based valuation
- **BS Army of Agents**: Autonomous agent architecture for distributed processing and monitoring
- **Interactive Dashboard**: Web-based interface for exploring valuations and running what-if analyses
- **Comprehensive API**: RESTful endpoints for accessing property data and valuations
- **Benton County Focus**: Specialized for the Benton County, WA real estate market

## Architecture

The system follows a modular architecture with the following high-level components:

1. **Data Acquisition Layer**: ETL components that extract data from various sources
2. **Persistence Layer**: PostgreSQL database for storage 
3. **Processing Layer**: Valuation engine and BS Army of Agents
4. **Presentation Layer**: Web interface and API endpoints

The application runs through a Flask web server (port 5001) for the user interface and a FastAPI server (port 5000) for API endpoints.

## Key Modules

### ETL Pipeline

The ETL (Extract, Transform, Load) pipeline is responsible for:

- Extracting raw property data from MLS, NARRPR, and PACS systems
- Transforming the data into a unified format with validation checks
- Loading validated data into the PostgreSQL database

The pipeline can be run in validation-only mode to check data quality without loading it into the database.

### Valuation Engine

The valuation engine:

- Implements multiple property valuation models:
  - Basic linear regression model
  - Enhanced multiple regression with advanced feature engineering
  - Gradient boosting with LightGBM for superior accuracy
  - Ensemble model that combines linear regression and LightGBM
- Uses features like square footage, bedrooms, bathrooms, and location
- Calculates confidence scores and prediction intervals
- Identifies comparable properties for validation
- Supports GIS (Geographic Information System) data integration

The valuation engine includes comprehensive model selection and evaluation:
- Feature normalization using StandardScaler for better numerical stability
- Multiple feature selection methods (f_regression, mutual_info, RFE, random_forest)
- Regularization options (Ridge, Lasso, ElasticNet) for linear models
- Cross-validation for robust performance evaluation
- Statistical significance reporting with p-values for model coefficients

The valuation results include feature importance metrics to explain which factors most influenced the estimated value, with different techniques depending on the model type (coefficient-based for linear models, gain-based for tree models).

### BS Army of Agents

The BS (Benton County System) Army of Agents consists of specialized autonomous agents:

- **BCBS Bootstrap Commander**: Manages the initialization and dependency verification
- **BCBS Cascade Operator**: Handles data flow and transformations
- **BCBS TDD Validator**: Ensures code quality and test coverage

Each agent has real-time status monitoring and performance metrics available in the Agent Dashboard.

### Dashboard

The web dashboard includes:

- Property search and filtering
- Detailed property valuation views
- What-If analysis for exploring feature impact
- Agent monitoring and status visualization
- ETL process validation results

## Getting Started

### Prerequisites

- Python 3.10 or later
- PostgreSQL 13 or later
- Node.js 18+ (for React components)
- API credentials for data sources:
  - NARRPR username and password
  - PACS API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bcbs-values.git
   cd bcbs-values
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (create a `.env` file):
   ```
   DATABASE_URL=postgresql://username:password@localhost:5432/bcbs_values
   SESSION_SECRET=your_secret_key
   NARRPR_USERNAME=your_narrpr_username
   NARRPR_PASSWORD=your_narrpr_password
   PACS_API_KEY=your_pacs_api_key
   BCBS_VALUES_API_KEY=your_api_key
   ```

4. Initialize the database:
   ```bash
   python -c "from db.database import Database; db = Database(); db.create_tables()"
   ```
   
5. Set up API key for authentication:
   ```bash
   ./set_api_key.sh your_api_key
   ```

### Configuration

Configuration files are stored in the `configs/` directory:

- `database_config.json`: Database connection settings
- `etl_config.json`: ETL pipeline configuration
- `module_config.json`: Core module configuration

Modify these files to adjust system behavior according to your needs.

## Workflow Setup

The system uses Replit workflows to manage long-running tasks such as the API server and the web application. Workflows are defined in `.replit.workflow/` directory.

To use workflows:

1. Access the Replit workflow menu in the top-right corner of the Replit interface
2. Select the desired workflow (API or WebApp)
3. Click "Run" to start the workflow

Available workflows:

- **API**: Starts the FastAPI server for accessing the valuation endpoints
- **WebApp**: Starts the Flask web application for the user interface
- **EnhancedAPI**: Runs the version with enhanced GIS integration
- **BasicAPI**: Runs a simpler version without database requirements
- **ValuationAPI**: Specifically for valuation endpoints
- **TestAPI**: For testing API functionality

See the `WORKFLOW_SETUP.md` file for more details on configuring and using workflows.

## Usage Guide

### Running the ETL Pipeline

Run the ETL pipeline with the following command:

```bash
# Run the full ETL pipeline for all data sources
python main.py --sources all

# Run ETL for specific sources only
python main.py --sources pacs mls

# Run in validate-only mode (no data loading)
python main.py --sources all --validate-only
```

The ETL process will:
1. Extract data from the specified sources
2. Transform and validate the data
3. Load valid data into the database (unless in validate-only mode)
4. Generate a validation report as a JSON file

### Accessing API Endpoints

We provide two separate API implementations to accommodate different needs:

#### Simple Valuation API (Recommended)

The Simple Valuation API is optimized for stability and doesn't require database access:

```bash
# Easiest method:
./run_simple_api.sh

# Or start directly with more control:
./start_simple_api.sh
```

The Simple API will be available at `http://localhost:5002` with these endpoints:

- `GET /api/health`: Health check endpoint
- `POST /api/valuation`: Generate a property valuation
- `GET /api/neighborhoods`: Get neighborhood quality ratings
- `GET /api/reference-points`: Get GIS reference points

Example Simple API calls:

```bash
# Check API health
curl "http://localhost:5002/api/health"

# Generate a standard property valuation (linear model)
curl -X POST "http://localhost:5002/api/valuation" \
  -H "Content-Type: application/json" \
  -d '{
    "square_feet": 2000,
    "bedrooms": 3,
    "bathrooms": 2.5,
    "year_built": 2005,
    "latitude": 46.2804,
    "longitude": -119.2752,
    "city": "Richland",
    "neighborhood": "Meadow Springs",
    "use_gis": true,
    "model_type": "linear"
  }'

# Generate a property valuation with LightGBM model
curl -X POST "http://localhost:5002/api/valuation" \
  -H "Content-Type: application/json" \
  -d '{
    "square_feet": 2000,
    "bedrooms": 3,
    "bathrooms": 2.5,
    "year_built": 2005,
    "latitude": 46.2804,
    "longitude": -119.2752,
    "city": "Richland",
    "neighborhood": "Meadow Springs",
    "use_gis": true,
    "model_type": "lightgbm",
    "feature_selection_method": "mutual_info"
  }'

# Generate a property valuation with ensemble model (linear + LightGBM)
curl -X POST "http://localhost:5002/api/valuation" \
  -H "Content-Type: application/json" \
  -d '{
    "square_feet": 2000,
    "bedrooms": 3,
    "bathrooms": 2.5,
    "year_built": 2005,
    "latitude": 46.2804,
    "longitude": -119.2752,
    "city": "Richland",
    "neighborhood": "Meadow Springs",
    "use_gis": true,
    "model_type": "ensemble"
  }'
```

#### Full API (Advanced)

The full FastAPI implementation includes additional features but requires database connectivity:

```bash
# Start the API using the workflow system:
# Use the Replit workflow menu to start the API workflow
# 
# Or run directly:
./start_api_server.sh
# 
# Or using Python directly:
python run_api.py
```

The Full API will be available at `http://localhost:5000` with these endpoints:

- `GET /api/valuations`: Get property valuations with filtering options
- `GET /api/valuations/{property_id}`: Get valuation for a specific property
- `POST /api/valuations`: Generate a new property valuation (requires API key)
- `GET /api/etl-status`: Get the current status of the ETL process
- `GET /api/agent-status`: Get the status of the BS Army of Agents

Example API calls:

```bash
# Get a list of property valuations with filtering
curl "http://localhost:5000/api/valuations?limit=5&min_value=300000" -H "X-API-KEY: your_api_key_here"

# Generate a new property valuation with LightGBM model (requires authentication)
curl -X POST "http://localhost:5000/api/valuations" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your_api_key_here" \
  -d '{
    "address": "123 Main St",
    "city": "Richland",
    "state": "WA",
    "zip_code": "99352",
    "property_type": "Single Family",
    "bedrooms": 3,
    "bathrooms": 2,
    "square_feet": 1800,
    "lot_size": 8500,
    "year_built": 1995,
    "latitude": 46.2804,
    "longitude": -119.2752,
    "use_gis": true,
    "model_config": {
      "model_type": "lightgbm",
      "feature_selection": "auto",
      "feature_selection_method": "mutual_info",
      "normalize_features": true,
      "lightgbm_params": {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 100
      }
    }
  }'
```

The POST `/api/valuations` endpoint generates a new property valuation using our advanced valuation models, including GIS features when available. This endpoint requires authentication with an API key passed in the `X-API-KEY` header.

### Using the Dashboard

Start the integrated web server:

```bash
# Start via workflow system:
# Use the Replit workflow menu to start the WebApp workflow
#
# Or run directly:
python start_web_app.py
#
# Or traditional method:
python run_integrated.py
```

This will start the Flask web application. Access the web dashboard at `http://localhost:5001`.

The main dashboard features:
- Search for properties by address, ID, or features
- View detailed property information and valuation
- Access ETL validation results
- Monitor agent status

### The Agent Dashboard

The Agent Dashboard displays real-time information about the BS Army of Agents:

- Current status of each agent (active, idle, busy, error)
- Queue size and processing metrics
- Success rates and performance statistics
- Detailed logs for selected agents

Access it at `http://localhost:5001/agents`.

### What-If Analysis

The What-If Analysis tool allows users to:

- Adjust property features (bedrooms, bathrooms, square footage)
- See the real-time impact on property valuation
- Compare different scenarios side by side
- Visualize feature importance

Access it at `http://localhost:5001/what-if`.

## CI/CD and Testing

This project uses GitHub Actions for continuous integration and deployment:

- Automatic testing on push to main/develop branches
- Code quality checks with flake8
- Comprehensive test suite with pytest
- Database testing with PostgreSQL service container
- Artifact generation for deployment

### Testing Improvements

Recent improvements to our testing suite include:

- **Robust error handling**: Tests now gracefully handle edge cases and unexpected input formats
- **DataFrame-compatible inputs**: Fixed compatibility issues with pandas Series vs dictionaries in test data
- **Flexible assertions**: Tests now accommodate slight numerical differences between statistical implementations
- **Conditional assertions**: Added conditional checks that skip tests when certain features are not implemented
- **Support for different data structures**: Tests now properly handle various input/output formats from the valuation engine
- **Enhanced model testing**: Added comprehensive tests for different model types:
  - Basic linear regression
  - Advanced multiple regression with feature engineering
  - Gradient boosting with LightGBM
  - Ensemble models combining linear and gradient boosting approaches
- **Feature selection validation**: Tests for various feature selection methods (f_regression, mutual_info, RFE, random_forest)
- **Model comparison framework**: Test infrastructure for comparing prediction accuracy across different model configurations

### Running Tests

Run tests locally with:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run valuation tests specifically
cd tests && python -m pytest test_valuation.py -v

# Run integration tests specifically
pytest tests/test_integration.py

# Test the Simple Valuation API
./test_simple_api.sh

# Test the authenticated API (requires API key)
./test_api_with_auth.sh
```

Test coverage reports are available in the `htmlcov/` directory after running the tests with coverage.

## Contributing

We welcome contributions to the BCBS Values project! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure everything works
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure your code follows our coding standards and includes appropriate tests.

## Feedback & Issues

We value your feedback! If you encounter any issues or have suggestions for improvements:

- Submit an issue through our [Issue Tracker](https://github.com/yourusername/bcbs-values/issues)
- Include as much detail as possible:
  - Steps to reproduce the issue
  - Expected vs. actual behavior
  - Screenshots if applicable
  - Environment details

For feature requests, please describe the desired functionality and how it would benefit the system.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*BCBS Values: Intelligent Real Estate Valuation for Benton County*