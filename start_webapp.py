#!/usr/bin/env python3
"""
Launcher script for BCBS Values Platform web server
This script auto-detects the environment and starts the appropriate server
"""

import app

if __name__ == "__main__":
    print("BCBS Values Platform Dashboard Server")
    print("====================================")
    app.start_server()