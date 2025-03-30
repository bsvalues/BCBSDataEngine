"""
Startup script for the BCBS Property Valuation Dashboard web application.
This script is used by Replit workflows to start the web application.
"""

import os
import sys
import subprocess

def main():
    """
    Start the BCBS Property Valuation Dashboard web application
    """
    # Set Python path to include current directory
    os.environ["PYTHONPATH"] = "."
    
    # Add session secret if not present
    if "SESSION_SECRET" not in os.environ:
        os.environ["SESSION_SECRET"] = "bcbs-valuation-dashboard-secret"
    
    # Start the Flask web application
    print("Starting BCBS Property Valuation Dashboard...")
    try:
        if os.path.exists("app.py"):
            subprocess.run([sys.executable, "app.py"], check=True)
        else:
            print("Error: app.py not found.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down BCBS Property Valuation Dashboard...")
    except Exception as e:
        print(f"Error running web application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()