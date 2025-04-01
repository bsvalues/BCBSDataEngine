#!/usr/bin/env python3
"""
Test Python script to verify environment
"""
import sys
import os

def main():
    """Print Python version and environment information"""
    print("Python Version Test")
    print("==================")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Current Working Directory: {os.getcwd()}")
    print("\nEnvironment Variables:")
    for key, value in sorted(os.environ.items()):
        print(f"  {key}={value}")
    
    print("\nPython Path:")
    for path in sys.path:
        print(f"  {path}")
    
    print("\nTest Successful!")
    print("===============")

if __name__ == "__main__":
    main()