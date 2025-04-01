#!/usr/bin/env python3
"""
Unified diagnostic server launcher with fallback to simple server
"""
import os
import sys
import subprocess
import time

def main():
    """Run the diagnostic server with fallback to simple server if needed"""
    port = os.environ.get('PORT', '5000')
    
    print("=" * 60)
    print("BCBS VALUES DIAGNOSTIC LAUNCHER")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print("=" * 60)
    
    # Try to run the advanced diagnostic server first
    try:
        print("Attempting to start advanced diagnostic server...")
        
        # Create a special environment with the needed variables
        env = os.environ.copy()
        env['DATABASE_URL'] = env.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/postgres')
        env['SESSION_SECRET'] = env.get('SESSION_SECRET', 'bcbs_values_session_secret_key_2025')
        env['API_KEY'] = env.get('API_KEY', 'bcbs_values_api_key_2025')
        
        # Start the advanced server
        advanced_process = subprocess.Popen(
            [sys.executable, 'quick_diagnostic_server.py'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait a moment to see if it starts
        time.sleep(2)
        
        # Check if the process is still running
        if advanced_process.poll() is None:
            print("Advanced diagnostic server started successfully.")
            print(f"You can access it at http://localhost:{port}/")
            # Transfer control to the subprocess
            stdout, stderr = advanced_process.communicate()
            print(stdout)
            if stderr:
                print("Error output:", stderr, file=sys.stderr)
            return advanced_process.returncode
        else:
            # Process exited, check output
            stdout, stderr = advanced_process.communicate()
            print("Advanced server failed to start.")
            print("Output:", stdout)
            if stderr:
                print("Error:", stderr)
    except Exception as e:
        print(f"Error launching advanced diagnostic server: {e}")
    
    # If we get here, the advanced server failed to start
    print("\n" + "=" * 60)
    print("FALLBACK: Starting simple diagnostic server")
    print("=" * 60)
    
    try:
        # Start the simple server
        simple_process = subprocess.Popen(
            [sys.executable, 'simple_diagnostic.py'],
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        print("Simple diagnostic server started.")
        print(f"You can access it at http://localhost:{port}/")
        
        # Transfer control to the subprocess
        stdout, stderr = simple_process.communicate()
        print(stdout)
        if stderr:
            print("Error output:", stderr, file=sys.stderr)
        return simple_process.returncode
    except Exception as e:
        print(f"Error launching simple diagnostic server: {e}")
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nDiagnostic server stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error in launcher: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)