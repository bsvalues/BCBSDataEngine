"""
Main entry point for the BCBS Property Valuation Dashboard application
"""

import os
import subprocess
import sys

def main():
    """
    Start the BCBS Property Valuation Dashboard web application
    """
    # Set Python path to include current directory
    os.environ["PYTHONPATH"] = "."
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("Error: app.py not found.")
        sys.exit(1)
    
    # Start the Flask web application
    print("Starting BCBS Property Valuation Dashboard...")
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\nShutting down BCBS Property Valuation Dashboard...")
    except Exception as e:
        print(f"Error running web application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()