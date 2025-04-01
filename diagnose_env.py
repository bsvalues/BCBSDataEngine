#!/usr/bin/env python3
"""
Diagnostic script to check Python environment and available packages
"""

import sys
import os
import platform
import socket

print("Python Diagnostic Information")
print("=============================")
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print(f"Platform: {platform.platform()}")
print(f"Working Directory: {os.getcwd()}")
print("\nEnvironment Variables:")
for key, value in os.environ.items():
    print(f"  {key}: {value}")

print("\nChecking for basic HTTP server capabilities:")
try:
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    print("  HTTP server modules available ✓")
except ImportError as e:
    print(f"  HTTP server modules not available: {e}")

print("\nChecking network:")
try:
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    print(f"  Hostname: {hostname}")
    print(f"  IP Address: {ip}")
    
    # Check if port 5000 is available
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(('localhost', 5000))
        if result == 0:
            print("  Port 5000 is in use")
        else:
            print("  Port 5000 is available")
except Exception as e:
    print(f"  Network check error: {e}")

print("\nFile System Check:")
for file in ['index.html', 'dashboard.html', 'static_fallback.html', 'start_webapp.py']:
    if os.path.exists(file):
        print(f"  {file}: Found ✓")
    else:
        print(f"  {file}: Not found ✗")