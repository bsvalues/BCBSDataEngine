#!/usr/bin/env python3
# Simple Python script to check if Python is working

import sys
import os
import datetime

def main():
    """Print basic Python environment information."""
    print("Python Check Script")
    print("===================")
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    
    print("\nEnvironment Variables:")
    for key, value in sorted(os.environ.items()):
        if any(x in key for x in ['DATABASE', 'PG', 'SESSION', 'API']):
            if 'SECRET' in key or 'PASSWORD' in key:
                print(f"{key}: [HIDDEN]")
            else:
                print(f"{key}: {value}")
    
    print("\nPython Path:")
    for path in sys.path:
        print(f"  {path}")
    
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()