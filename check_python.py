#!/usr/bin/env python3
"""
Python availability checker
This script simply prints system information to verify Python is working
"""

import os
import sys
import platform
import datetime

def main():
    """Print system information to verify Python is working"""
    print("=" * 50)
    print("PYTHON AVAILABILITY CHECK")
    print("=" * 50)
    print(f"Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Python Implementation: {platform.python_implementation()}")
    print(f"Python Path: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Environment Variables:")
    for key in sorted(os.environ.keys()):
        if key.startswith(('DATABASE_URL', 'PG', 'API_KEY', 'SESSION_SECRET', 'PORT', 'PYTHONPATH')):
            if 'PASSWORD' in key or 'SECRET' in key:
                value = '*****'
            else:
                value = os.environ.get(key)
            print(f"  {key}: {value}")
    print("=" * 50)
    print("Python is available and working!")
    print("=" * 50)
    return 0

if __name__ == "__main__":
    sys.exit(main())