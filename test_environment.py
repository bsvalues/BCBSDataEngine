#!/usr/bin/env python3
"""
Environment Testing Module

This module tests the environment to detect available tools,
languages, and utilities for the BCBS Values Platform.
"""

import os
import sys
import subprocess
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_environment():
    """
    Detect available tools and languages in the environment.
    
    Returns:
        dict: Dictionary containing environment information
    """
    environment = {
        "platform": sys.platform,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "python_available": True,
        "node_available": False,
        "bash_available": False,
        "bash_utilities": {},
        "allowed_ports": []
    }
    
    # Check for Node.js
    try:
        node_result = subprocess.run(
            ["node", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if node_result.returncode == 0:
            environment["node_available"] = True
            environment["node_version"] = node_result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Node.js not detected")
    
    # Check for bash
    try:
        bash_result = subprocess.run(
            ["bash", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if bash_result.returncode == 0:
            environment["bash_available"] = True
            environment["bash_version"] = bash_result.stdout.split("\n")[0]
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Bash not detected")
    
    # Check for common utilities
    utilities = ["curl", "wget", "nc", "netcat", "socat", "telnet"]
    environment["bash_utilities"] = {
        util: check_executable(util) for util in utilities
    }
    
    # Check available ports
    for port in [5000, 5001, 5002, 5003, 5004, 8000, 8080, 3000]:
        if check_port_available(port):
            environment["allowed_ports"].append(port)
    
    return environment

def check_executable(executable):
    """
    Check if an executable is available in PATH
    
    Args:
        executable (str): Name of the executable to check
        
    Returns:
        bool: True if executable is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["which", executable], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_port_available(port):
    """
    Check if a port is available for binding
    
    Args:
        port (int): Port number to check
        
    Returns:
        bool: True if port is available, False otherwise
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('0.0.0.0', port))
        sock.listen(1)
        sock.close()
        return True
    except (socket.error, OSError):
        try:
            sock.close()
        except:
            pass
        return False

def find_executables(commands):
    """
    Find paths to executables for specified commands
    
    Args:
        commands (list): List of command names to find
        
    Returns:
        dict: Dictionary of command -> path mappings
    """
    result = {}
    for cmd in commands:
        try:
            which_result = subprocess.run(
                ["which", cmd], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if which_result.returncode == 0:
                path = which_result.stdout.strip()
                if os.path.exists(path) and os.access(path, os.X_OK):
                    result[cmd] = path
                else:
                    result[cmd] = None
            else:
                result[cmd] = None
        except (subprocess.SubprocessError, FileNotFoundError):
            result[cmd] = None
    
    return result

def test_environment_detection():
    """Test if the environment detection correctly identifies available tools."""
    env = detect_environment()
    assert isinstance(env, dict)
    assert "python_available" in env
    assert "node_available" in env
    assert "bash_utilities" in env
    return True

def test_path_executable():
    """Test if executables are properly detected."""
    executable_paths = find_executables(["python", "node", "bash"])
    for command, path in executable_paths.items():
        if path:  # If a path was found
            assert os.path.exists(path)
            assert os.access(path, os.X_OK)
    return True

def main():
    """Main function for environment testing"""
    logger.info("Starting environment detection...")
    env = detect_environment()
    print(json.dumps(env, indent=2))
    
    logger.info("Running tests...")
    test_results = {
        "test_environment_detection": test_environment_detection(),
        "test_path_executable": test_path_executable()
    }
    
    print("Test Results:")
    for test, result in test_results.items():
        print(f"  {test}: {'PASS' if result else 'FAIL'}")
    
    all_passed = all(test_results.values())
    if all_passed:
        logger.info("All tests passed")
    else:
        logger.error("Some tests failed")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())