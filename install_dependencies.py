#!/usr/bin/env python3
"""
Dependency installer script for BCBS Values Platform
This script installs required Python packages using pip
"""
import sys
import subprocess
import os
import time

def log(message):
    """Print a log message with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_pip_install(package):
    """Run pip install for a specific package"""
    log(f"Installing {package}...")
    try:
        cmd = [sys.executable, "-m", "pip", "install", "--user", package]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            log(f"Successfully installed {package}")
            return True
        else:
            log(f"Failed to install {package}")
            log(f"Error: {result.stderr}")
            return False
    except Exception as e:
        log(f"Exception while installing {package}: {e}")
        return False

def main():
    """Install required dependencies"""
    log("Starting dependency installation")
    log(f"Python executable: {sys.executable}")
    
    # Required packages
    packages = [
        "flask",
        "sqlalchemy",
        "requests",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn"
    ]
    
    # Install each package
    success_count = 0
    for package in packages:
        if run_pip_install(package):
            success_count += 1
    
    # Summary
    log(f"Installation complete: {success_count}/{len(packages)} packages installed successfully")
    
    # Verify installations
    log("Verifying installations:")
    for package in packages:
        try:
            __import__(package)
            log(f"✅ {package} is available")
        except ImportError:
            log(f"❌ {package} is not available")

if __name__ == "__main__":
    main()