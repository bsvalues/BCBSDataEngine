#!/usr/bin/env python3
"""
Enhanced Python Environment Test Script
This script provides a comprehensive check of the Python environment
"""
import os
import sys
import platform
import json
import socket
import datetime
import subprocess
import importlib.util

def section_header(title):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def check_module(module_name):
    """Check if a module is available and return its version if possible"""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False, None
        
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "Unknown")
        return True, version
    except ImportError:
        return False, None
    except Exception as e:
        return False, f"Error: {str(e)}"

def run_command(command):
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=False, 
            capture_output=True, 
            text=True
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_port(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def main():
    """Run the environment test"""
    print("BCBS Values Platform - Python Environment Test")
    print("=" * 60)
    print(f"Test run at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System Information
    section_header("System Information")
    print(f"Platform: {platform.platform()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Python Implementation: {platform.python_implementation()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Path: {sys.path}")
    
    # Environment Variables
    section_header("Environment Variables")
    for key, value in sorted(os.environ.items()):
        # Skip displaying some long environment variables
        if key in ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH"]:
            print(f"{key}: [truncated for brevity]")
        else:
            print(f"{key}: {value}")
    
    # Python Modules
    section_header("Required Python Modules")
    required_modules = [
        "flask", "sqlalchemy", "requests", "numpy", "pandas", 
        "matplotlib", "seaborn", "scikit-learn", "tensorflow", "torch"
    ]
    
    for module in required_modules:
        available, version = check_module(module)
        status = f"{'✅' if available else '❌'} {module:<15}"
        if available:
            status += f" (Version: {version})"
        else:
            status += " (Not installed)"
        print(status)
    
    # Network Ports
    section_header("Network Port Status")
    ports_to_check = [5000, 5001, 5002, 8000, 8080]
    
    for port in ports_to_check:
        in_use = check_port(port)
        status = "In use" if in_use else "Available"
        print(f"Port {port:<6} - {status}")
    
    # File System Check
    section_header("File System Check")
    important_files = [
        "app.py", "main.py", "start_webapp.py", "run_webapp.sh", 
        ".replit", "index.html", "dashboard.html"
    ]
    
    for filename in important_files:
        exists = os.path.exists(filename)
        status = "✅ Exists" if exists else "❌ Missing"
        if exists and os.path.isfile(filename):
            size = os.path.getsize(filename)
            status += f" ({size} bytes)"
        print(f"{filename:<20} - {status}")
    
    # Try to import Flask
    section_header("Flask Availability Test")
    returncode, stdout, stderr = run_command(f"{sys.executable} -m flask --version")
    
    if returncode == 0:
        print("Flask is available:")
        print(stdout)
    else:
        print("Flask is not available:")
        if stderr:
            print(stderr)
    
    # Test Summary
    section_header("Test Summary")
    print("Environment tests completed.")
    print(f"Python executable is available: {'✅' if sys.executable else '❌'}")
    
    flask_available, _ = check_module('flask')
    print(f"Flask is available: {'✅' if flask_available else '❌'}")
    
    app_exists = os.path.exists('app.py')
    print(f"app.py exists: {'✅' if app_exists else '❌'}")
    
    port_available = not check_port(5002)
    print(f"Port 5002 is available: {'✅' if port_available else '❌ (port is in use)'}")
    
    shell_scripts_executable = all(os.access(script, os.X_OK) 
                                for script in ["run_webapp.sh"] 
                                if os.path.exists(script))
    print(f"Shell scripts are executable: {'✅' if shell_scripts_executable else '❌'}")
    

if __name__ == "__main__":
    main()