#!/usr/bin/env python
"""
Environment Diagnostic Tool for BCBS Values Application

This script checks:
1. Required environment variables
2. Database connection
3. Installed dependencies
4. Access to external APIs (NARRPR, MLS, PACS)
5. System configuration

Author: BCBS Engineering Team
Date: March 31, 2025
"""

import os
import sys
import json
import logging
import subprocess
import importlib
import traceback
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set

# Setup logging
LOG_FILE = f"bcbs_env_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger("env_diagnostics")

# Define color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message: str) -> None:
    """Print a formatted header message."""
    logger.info(f"{Colors.HEADER}{Colors.BOLD}{message}{Colors.ENDC}")
    print(f"\n{Colors.HEADER}{Colors.BOLD}{message}{Colors.ENDC}")


def print_success(message: str) -> None:
    """Print a success message."""
    logger.info(f"SUCCESS: {message}")
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    logger.warning(message)
    print(f"{Colors.YELLOW}! {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    """Print an error message."""
    logger.error(message)
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")


def print_info(message: str) -> None:
    """Print an info message."""
    logger.info(message)
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")


def check_environment_variables() -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Check for required environment variables.
    
    Returns:
        Tuple containing:
        - Boolean indicating if all critical variables are present
        - List of missing critical variables
        - List of missing recommended variables
        - List of present variables
    """
    print_header("Checking Environment Variables")
    
    # Define critical environment variables (application won't function without these)
    critical_vars = [
        "DATABASE_URL",      # Database connection string
        "PGDATABASE",        # PostgreSQL database name
        "PGUSER",            # PostgreSQL username
        "PGPASSWORD",        # PostgreSQL password
        "PGHOST",            # PostgreSQL host
        "PGPORT",            # PostgreSQL port
        "SESSION_SECRET"     # Flask session secret key
    ]
    
    # Define recommended variables (application may function with reduced capabilities)
    recommended_vars = [
        "API_KEY",                  # API key for authentication
        "BCBS_VALUES_API_KEY",      # Alternative API key name
        "JWT_SECRET",               # JWT token secret
        "NARRPR_API_KEY",           # NARRPR API access key
        "NARRPR_API_SECRET",        # NARRPR API secret
        "MLS_API_KEY",              # MLS API access key
        "PACS_API_KEY",             # PACS API access key
        "LOG_LEVEL",                # Logging level
        "ENABLE_CACHING",           # Enable API response caching
        "NODE_ENV",                 # Node.js environment
        "PORT"                      # Port for the server
    ]
    
    # Check critical variables
    missing_critical = []
    for var in critical_vars:
        if not os.environ.get(var):
            missing_critical.append(var)
            print_error(f"Missing critical environment variable: {var}")
        else:
            # Don't log the actual values of sensitive variables
            if var in ["PGPASSWORD", "SESSION_SECRET", "JWT_SECRET"]:
                print_success(f"Found critical environment variable: {var} (value hidden)")
            else:
                print_success(f"Found critical environment variable: {var}")
    
    # Check recommended variables
    missing_recommended = []
    for var in recommended_vars:
        if not os.environ.get(var):
            missing_recommended.append(var)
            print_warning(f"Missing recommended environment variable: {var}")
        else:
            # Don't log the actual values of sensitive variables
            if any(sensitive in var for sensitive in ["KEY", "SECRET", "PASSWORD"]):
                print_success(f"Found recommended environment variable: {var} (value hidden)")
            else:
                print_success(f"Found recommended environment variable: {var}")
    
    # Get a list of present variables (without sensitive values)
    present_vars = []
    for var in critical_vars + recommended_vars:
        if os.environ.get(var):
            present_vars.append(var)
    
    # Summary
    if missing_critical:
        print_error(f"Missing {len(missing_critical)} critical environment variables. Application may not function correctly.")
    else:
        print_success("All critical environment variables are set.")
    
    if missing_recommended:
        print_warning(f"Missing {len(missing_recommended)} recommended environment variables. Some features may be limited.")
    
    # Check for secure SESSION_SECRET
    session_secret = os.environ.get("SESSION_SECRET", "")
    if session_secret and (len(session_secret) < 16 or session_secret == "bcbs_values_session_secret_key_2025"):
        print_warning("SESSION_SECRET appears to be the default value or too short. Consider setting a stronger secret.")
    
    # Return results
    all_critical_present = len(missing_critical) == 0
    return all_critical_present, missing_critical, missing_recommended, present_vars


def check_database_connection() -> bool:
    """
    Test the database connection using environment variables.
    
    Returns:
        Boolean indicating if the database connection is successful
    """
    print_header("Checking Database Connection")

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print_error("DATABASE_URL environment variable is not set.")
        return False
    
    # Try to import required modules
    try:
        import psycopg2
    except ImportError:
        print_error("psycopg2 is not installed. Install it with 'pip install psycopg2-binary'.")
        return False
    
    # Try to connect to the database
    try:
        print_info(f"Attempting to connect to the database...")
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Check if we can execute a simple query
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        
        print_success(f"Successfully connected to PostgreSQL database.")
        print_info(f"Database version: {db_version[0]}")
        
        # Check for essential tables
        table_check = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        """
        cursor.execute(table_check)
        tables = cursor.fetchall()
        if tables:
            print_success(f"Found {len(tables)} tables in the database:")
            for table in tables:
                print_info(f"  - {table[0]}")
        else:
            print_warning("No tables found in the database. Schema may not be initialized.")
        
        # Close the connection
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print_error(f"Failed to connect to the database: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Check if specific parts of the connection string work
        try:
            # Try individual PostgreSQL environment variables
            pg_host = os.environ.get("PGHOST")
            pg_port = os.environ.get("PGPORT")
            pg_database = os.environ.get("PGDATABASE")
            pg_user = os.environ.get("PGUSER")
            pg_password = os.environ.get("PGPASSWORD")
            
            # Validate each component
            if not pg_host:
                print_error("PGHOST is not set or empty.")
            if not pg_port:
                print_error("PGPORT is not set or empty.")
            elif not pg_port.isdigit():
                print_error(f"PGPORT is not a valid number: {pg_port}")
            if not pg_database:
                print_error("PGDATABASE is not set or empty.")
            if not pg_user:
                print_error("PGUSER is not set or empty.")
            if not pg_password:
                print_error("PGPASSWORD is not set or empty.")
                
            # Check if DATABASE_URL is properly constructed from PG* variables
            expected_db_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
            sanitized_expected = f"postgresql://{pg_user}:******@{pg_host}:{pg_port}/{pg_database}"
            sanitized_actual = re.sub(r':[^@:]+@', ':******@', db_url) if db_url else "None"
            
            if db_url and expected_db_url != db_url:
                print_warning(f"DATABASE_URL may not match individual PostgreSQL environment variables.")
                print_info(f"Expected format: {sanitized_expected}")
                print_info(f"Actual format:   {sanitized_actual}")
        except Exception as detail_error:
            print_error(f"Error while checking database variables: {str(detail_error)}")
        
        return False


def check_installed_dependencies() -> Tuple[bool, List[str], List[Dict[str, str]]]:
    """
    Check if all required Python dependencies are installed.
    
    Returns:
        Tuple containing:
        - Boolean indicating if all critical dependencies are installed
        - List of missing dependencies
        - List of installed dependencies with versions
    """
    print_header("Checking Installed Dependencies")
    
    # Define required dependencies
    required_deps = [
        # Core dependencies
        "flask",
        "flask-sqlalchemy",
        "flask-login",
        "werkzeug",
        "pyjwt",
        "psycopg2",
        "gunicorn",
        "python-dotenv",
        "fastapi",
        "uvicorn",
        "pydantic",
        # Database
        "sqlalchemy",
        # Testing
        "pytest",
        "pytest-flask",
        # ETL
        "pandas",
        "numpy",
        "requests",
        # Valuation
        "scikit-learn"
    ]
    
    # Check if each dependency is installed
    missing_deps = []
    installed_deps = []
    
    for dep in required_deps:
        try:
            # Try to import the module
            module = importlib.import_module(dep.replace('-', '_'))
            
            # Get the version if available
            version = getattr(module, '__version__', 'unknown')
            
            # Record the installed dependency
            installed_deps.append({
                "name": dep,
                "version": version
            })
            
            print_success(f"Found dependency: {dep} (version {version})")
        except ImportError:
            missing_deps.append(dep)
            print_error(f"Missing dependency: {dep}")
        except Exception as e:
            print_warning(f"Error checking dependency {dep}: {str(e)}")
    
    # Summary
    if missing_deps:
        print_error(f"Missing {len(missing_deps)} dependencies. Install them using pip or poetry.")
        print_info("For pip: pip install " + " ".join(missing_deps))
        print_info("For poetry: poetry add " + " ".join(missing_deps))
    else:
        print_success("All required dependencies are installed.")
    
    # Check pyproject.toml dependency version compatibility
    try:
        if os.path.exists('pyproject.toml'):
            with open('pyproject.toml', 'r') as f:
                content = f.read()
                
            # Check Python version
            python_version_match = re.search(r'python\s*=\s*"\^?(\d+\.\d+)"', content)
            if python_version_match:
                required_python = python_version_match.group(1)
                current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
                
                if required_python != current_python:
                    print_warning(f"Python version mismatch: required {required_python}, current {current_python}")
                else:
                    print_success(f"Using correct Python version: {current_python}")
    except Exception as e:
        print_warning(f"Error checking pyproject.toml: {str(e)}")
    
    # Return results
    all_deps_installed = len(missing_deps) == 0
    return all_deps_installed, missing_deps, installed_deps


def check_api_keys() -> bool:
    """
    Check if the API keys for external services are configured.
    
    Returns:
        Boolean indicating if the essential API keys are configured
    """
    print_header("Checking API Keys")
    
    # Check for BCBS Values API key
    api_key = os.environ.get("API_KEY") or os.environ.get("BCBS_VALUES_API_KEY")
    if not api_key:
        print_warning("No API key found (API_KEY or BCBS_VALUES_API_KEY). API authentication may fail.")
    else:
        print_success("API key is configured.")
    
    # Check for NARRPR API credentials
    narrpr_key = os.environ.get("NARRPR_API_KEY")
    narrpr_secret = os.environ.get("NARRPR_API_SECRET")
    
    if not narrpr_key or not narrpr_secret:
        print_warning("NARRPR API credentials are not configured. NARRPR data source may not work.")
    else:
        print_success("NARRPR API credentials are configured.")
    
    # Check for MLS API key
    mls_key = os.environ.get("MLS_API_KEY")
    if not mls_key:
        print_warning("MLS API key is not configured. MLS data source may not work.")
    else:
        print_success("MLS API key is configured.")
    
    # Check for PACS API key
    pacs_key = os.environ.get("PACS_API_KEY")
    if not pacs_key:
        print_warning("PACS API key is not configured. PACS data source may not work.")
    else:
        print_success("PACS API key is configured.")
    
    # Return true if at least the main API key is configured
    return api_key is not None


def check_config_files() -> bool:
    """
    Check if all required configuration files exist and are valid.
    
    Returns:
        Boolean indicating if all required configuration files are valid
    """
    print_header("Checking Configuration Files")
    
    config_files = [
        "configs/database_config.json",
        "configs/etl_config.json",
        "configs/module_config.json"
    ]
    
    all_valid = True
    
    # Check each configuration file
    for config_file in config_files:
        if not os.path.exists(config_file):
            print_error(f"Configuration file not found: {config_file}")
            all_valid = False
            continue
        
        # Check if the file is valid JSON
        try:
            with open(config_file, 'r') as f:
                json.load(f)
            print_success(f"Configuration file is valid: {config_file}")
        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON in configuration file {config_file}: {str(e)}")
            all_valid = False
        except Exception as e:
            print_error(f"Error reading configuration file {config_file}: {str(e)}")
            all_valid = False
    
    # Check .env file
    if os.path.exists('.env'):
        print_success(".env file exists.")
        
        # Check if .env file has basic required variables
        try:
            with open('.env', 'r') as f:
                env_content = f.read()
                
            # Check for key variables
            for var in ["DATABASE_URL", "SESSION_SECRET"]:
                if var not in env_content:
                    print_warning(f"{var} is not defined in .env file.")
        except Exception as e:
            print_warning(f"Error reading .env file: {str(e)}")
    else:
        print_warning(".env file does not exist. Environment variables should be set using system environment.")
    
    return all_valid


def diagnostic_summary(env_vars_status: Tuple, db_status: bool, deps_status: Tuple, api_keys_status: bool, config_status: bool) -> None:
    """
    Print a summary of all diagnostic checks.
    
    Args:
        env_vars_status: Results from check_environment_variables
        db_status: Results from check_database_connection
        deps_status: Results from check_installed_dependencies
        api_keys_status: Results from check_api_keys
        config_status: Results from check_config_files
    """
    print_header("DIAGNOSTIC SUMMARY")
    
    env_vars_ok, missing_critical, missing_recommended, _ = env_vars_status
    deps_ok, missing_deps, _ = deps_status
    
    # Check overall status
    all_ok = env_vars_ok and db_status and deps_ok and api_keys_status and config_status
    
    # Print summary table
    print_info("╔═══════════════════════════════════╦═══════════╗")
    print_info("║ Diagnostic Check                  ║ Status     ║")
    print_info("╠═══════════════════════════════════╬═══════════╣")
    print_info(f"║ Environment Variables             ║ {Colors.GREEN + 'PASS' + Colors.ENDC if env_vars_ok else Colors.RED + 'FAIL' + Colors.ENDC}       ║")
    print_info(f"║ Database Connection               ║ {Colors.GREEN + 'PASS' + Colors.ENDC if db_status else Colors.RED + 'FAIL' + Colors.ENDC}       ║")
    print_info(f"║ Dependencies                      ║ {Colors.GREEN + 'PASS' + Colors.ENDC if deps_ok else Colors.RED + 'FAIL' + Colors.ENDC}       ║")
    print_info(f"║ API Keys                          ║ {Colors.GREEN + 'PASS' + Colors.ENDC if api_keys_status else Colors.YELLOW + 'WARNING' + Colors.ENDC}   ║")
    print_info(f"║ Configuration Files               ║ {Colors.GREEN + 'PASS' + Colors.ENDC if config_status else Colors.RED + 'FAIL' + Colors.ENDC}       ║")
    print_info("╚═══════════════════════════════════╩═══════════╝")
    
    # Print action items if there are issues
    if not all_ok:
        print_header("ACTION ITEMS")
        
        if missing_critical:
            print_error(f"Set the following critical environment variables:")
            for var in missing_critical:
                print_error(f"  - {var}")
        
        if missing_recommended:
            print_warning(f"Consider setting these recommended environment variables:")
            for var in missing_recommended:
                print_warning(f"  - {var}")
        
        if not db_status:
            print_error("Fix database connection issues:")
            print_error("  - Check database credentials in environment variables")
            print_error("  - Ensure PostgreSQL is running and accessible")
            print_error("  - Verify the DATABASE_URL format")
        
        if missing_deps:
            print_error("Install missing dependencies:")
            print_error(f"  pip install {' '.join(missing_deps)}")
            print_error("  or")
            print_error(f"  poetry add {' '.join(missing_deps)}")
        
        if not api_keys_status:
            print_warning("Set API keys for external services:")
            print_warning("  - API_KEY or BCBS_VALUES_API_KEY for authentication")
            print_warning("  - NARRPR_API_KEY and NARRPR_API_SECRET for NARRPR data source")
            print_warning("  - MLS_API_KEY for MLS data source")
            print_warning("  - PACS_API_KEY for PACS data source")
        
        if not config_status:
            print_error("Fix configuration files:")
            print_error("  - Ensure all configuration files exist and contain valid JSON")
            print_error("  - Check for syntax errors in configuration files")
    
    # Print success message if all checks pass
    if all_ok:
        print_success("All diagnostic checks passed! The environment is correctly configured.")
    else:
        print_warning("Some diagnostic checks failed. Fix the issues above before proceeding.")
    
    # Final note on log file
    print_info(f"\nDetailed diagnostic logs are available in: {LOG_FILE}")


def main() -> int:
    """
    Main function to run all diagnostic checks.
    
    Returns:
        Exit code (0 for success, 1 for failures)
    """
    print_header("BCBS VALUES ENVIRONMENT DIAGNOSTICS")
    print_info(f"Running diagnostics at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Python version: {sys.version}")
    print_info(f"Platform: {sys.platform}")
    
    try:
        # Run all checks
        env_vars_status = check_environment_variables()
        db_status = check_database_connection()
        deps_status = check_installed_dependencies()
        api_keys_status = check_api_keys()
        config_status = check_config_files()
        
        # Print summary
        diagnostic_summary(env_vars_status, db_status, deps_status, api_keys_status, config_status)
        
        # Determine exit code based on critical checks
        env_vars_ok = env_vars_status[0]
        deps_ok = deps_status[0]
        
        if not env_vars_ok or not db_status or not deps_ok or not config_status:
            return 1  # Return non-zero exit code if any critical check fails
        return 0  # Return zero exit code if all critical checks pass
        
    except Exception as e:
        print_error(f"Error running diagnostics: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())