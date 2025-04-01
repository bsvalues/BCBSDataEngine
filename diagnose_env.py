#!/usr/bin/env python3
"""
Environment Diagnostic Tool for BCBS Values System

This script checks the Python environment for required modules,
environment variables, and configuration.
"""
import os
import sys
import importlib
import platform
import datetime
import json
import socket
import tempfile
import subprocess
from pathlib import Path

# List of required modules for the BCBS Values system
REQUIRED_MODULES = [
    # Core modules
    ('fastapi', 'FastAPI web framework'),
    ('uvicorn', 'ASGI server'),
    ('sqlalchemy', 'SQL toolkit and ORM'),
    ('psycopg2', 'PostgreSQL adapter'),
    ('pandas', 'Data analysis library'),
    ('numpy', 'Numerical computing library'),
    ('scikit-learn', 'Machine learning library'),
    ('lightgbm', 'Gradient boosting framework'),
    ('matplotlib', 'Plotting library'),
    ('httpx', 'HTTP client'),
    ('pydantic', 'Data validation'),
    ('pytest', 'Testing framework'),
    ('jinja2', 'Template engine'),
    
    # Specialized modules
    ('geopandas', 'Geospatial data handling'),
    ('shapely', 'Geometric operations'),
    ('pyproj', 'Cartographic projections'),
    ('rtree', 'Spatial indexing'),
    ('flask', 'Web framework (for diagnostics)'),
    ('requests', 'HTTP library'),
    ('websockets', 'WebSocket implementation'),
    ('aiohttp', 'Async HTTP client/server'),
    ('asyncpg', 'Async PostgreSQL client'),
]

# Required environment variables
REQUIRED_ENV_VARS = [
    ('DATABASE_URL', 'PostgreSQL connection string'),
    ('SESSION_SECRET', 'Session encryption key'),
    ('API_KEY', 'API authentication key'),
]

# Optional environment variables
OPTIONAL_ENV_VARS = [
    ('NARRPR_API_KEY', 'NARRPR API access key'),
    ('MLS_USERNAME', 'MLS system username'),
    ('MLS_PASSWORD', 'MLS system password'),
    ('PACS_URL', 'PACS system access URL'),
    ('LOG_LEVEL', 'Logging level'),
    ('PORT', 'HTTP server port'),
]

def check_module(module_name):
    """Try to import a module and return its version"""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None
    except Exception as e:
        return False, f"Error: {e}"

def check_database_connection():
    """Try to connect to the database"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        return False, "DATABASE_URL not set"
    
    try:
        # Try to import sqlalchemy
        import sqlalchemy
        from sqlalchemy import create_engine, text
        
        # Create engine
        engine = create_engine(database_url)
        
        # Try to connect
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            if result and result[0] == 1:
                return True, "Connection successful"
            else:
                return False, "Unexpected result from database"
    except ImportError:
        return False, "SQLAlchemy not installed"
    except Exception as e:
        return False, f"Error: {e}"

def check_file_permissions():
    """Check if we can write to temporary files"""
    try:
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(b"Test content")
            tmp.flush()
            
            # Try to read it back
            with open(tmp.name, 'rb') as f:
                content = f.read()
                if content == b"Test content":
                    return True, "File operations successful"
                else:
                    return False, "File content mismatch"
    except Exception as e:
        return False, f"Error: {e}"

def check_network():
    """Check if we can connect to the internet"""
    try:
        # Try to resolve google.com
        socket.gethostbyname('google.com')
        
        # Try to connect to port 80
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(('google.com', 80))
        s.close()
        
        return True, "Network connection successful"
    except Exception as e:
        return False, f"Error: {e}"

def check_gpu():
    """Check if GPU is available (via subprocess)"""
    try:
        # Try nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            # Process the output to get GPU info
            lines = result.stdout.split('\n')
            gpu_info = ' '.join([line for line in lines if 'NVIDIA' in line and 'GPU' in line])
            return True, f"GPU available: {gpu_info}"
        else:
            return False, "No NVIDIA GPU detected"
    except FileNotFoundError:
        # nvidia-smi not found
        return False, "nvidia-smi not found"
    except subprocess.TimeoutExpired:
        return False, "GPU check timed out"
    except Exception as e:
        return False, f"Error checking GPU: {e}"

def main():
    """Run all diagnostic checks and print results"""
    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'system': {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'python_executable': sys.executable,
            'cpu_count': os.cpu_count(),
            'hostname': platform.node(),
            'username': os.environ.get('USER', os.environ.get('USERNAME', 'unknown')),
            'cwd': os.getcwd(),
        },
        'modules': {},
        'environment': {},
        'checks': {}
    }
    
    # Check for required modules
    print("Checking required modules...")
    for module_name, description in REQUIRED_MODULES:
        installed, version = check_module(module_name)
        results['modules'][module_name] = {
            'installed': installed,
            'version': version,
            'description': description
        }
        status = "✓" if installed else "✗"
        version_str = f"v{version}" if installed and version != 'unknown' else ""
        print(f"  {status} {module_name:<20} {version_str:<10} {description}")
    
    # Check for required environment variables
    print("\nChecking required environment variables...")
    for var_name, description in REQUIRED_ENV_VARS:
        value = os.environ.get(var_name)
        is_set = value is not None
        results['environment'][var_name] = {
            'is_set': is_set,
            'description': description
        }
        status = "✓" if is_set else "✗"
        value_str = "Set (hidden)" if is_set else "Not set"
        print(f"  {status} {var_name:<20} {value_str:<15} {description}")
    
    # Check for optional environment variables
    print("\nChecking optional environment variables...")
    for var_name, description in OPTIONAL_ENV_VARS:
        value = os.environ.get(var_name)
        is_set = value is not None
        results['environment'][var_name] = {
            'is_set': is_set,
            'description': description
        }
        status = "✓" if is_set else "-"
        value_str = "Set (hidden)" if is_set else "Not set"
        print(f"  {status} {var_name:<20} {value_str:<15} {description}")
    
    # Additional checks
    print("\nRunning additional checks...")
    
    # Database connection
    db_success, db_message = check_database_connection()
    results['checks']['database'] = {
        'success': db_success,
        'message': db_message
    }
    status = "✓" if db_success else "✗"
    print(f"  {status} Database connection: {db_message}")
    
    # File permissions
    file_success, file_message = check_file_permissions()
    results['checks']['file_permissions'] = {
        'success': file_success,
        'message': file_message
    }
    status = "✓" if file_success else "✗"
    print(f"  {status} File permissions: {file_message}")
    
    # Network connection
    net_success, net_message = check_network()
    results['checks']['network'] = {
        'success': net_success,
        'message': net_message
    }
    status = "✓" if net_success else "✗"
    print(f"  {status} Network connection: {net_message}")
    
    # GPU check
    gpu_success, gpu_message = check_gpu()
    results['checks']['gpu'] = {
        'success': gpu_success,
        'message': gpu_message
    }
    status = "✓" if gpu_success else "-"  # Not critical
    print(f"  {status} GPU check: {gpu_message}")
    
    # Output summary
    print("\nDiagnostic Summary:")
    total_modules = len(REQUIRED_MODULES)
    installed_modules = sum(1 for mod in results['modules'].values() if mod['installed'])
    print(f"  Modules: {installed_modules}/{total_modules} installed")
    
    total_required_vars = len(REQUIRED_ENV_VARS)
    set_required_vars = sum(1 for var in results['environment'].values() if var.get('is_set', False))
    print(f"  Required environment variables: {set_required_vars}/{total_required_vars} set")
    
    critical_checks = ['database', 'file_permissions', 'network']
    passed_checks = sum(1 for check in critical_checks if results['checks'].get(check, {}).get('success', False))
    print(f"  Critical checks: {passed_checks}/{len(critical_checks)} passed")
    
    # Calculate overall health percentage
    health_score = (installed_modules / total_modules * 40 + 
                   set_required_vars / total_required_vars * 30 + 
                   passed_checks / len(critical_checks) * 30)
    print(f"  Overall health: {health_score:.1f}%")
    
    # Save results to a JSON file
    output_file = Path('diagnostic_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    
    # Return appropriate exit code
    if health_score < 60:
        print("\nCRITICAL: Environment has significant issues that need to be resolved.")
        return 2
    elif health_score < 90:
        print("\nWARNING: Environment has some issues that should be addressed.")
        return 1
    else:
        print("\nOK: Environment is healthy.")
        return 0

if __name__ == '__main__':
    sys.exit(main())