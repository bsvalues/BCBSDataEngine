#!/bin/bash
# Diagnostic script for BCBS Values Platform workflow system

# Set up logging with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "BCBS Values Platform Workflow Diagnostics"
log "=========================================="

# Find Python executable
PYTHON_PATH="/mnt/nixmodules/nix/store/fj3r91wy2ggvriazbkl24vyarny6qb1s-python3-3.11.10-env/bin/python3"

# Check Python availability
log "Checking Python availability..."
if [ -x "$PYTHON_PATH" ]; then
    log "✅ Python executable found at: $PYTHON_PATH"
else
    log "❌ Python executable not found at expected path"
    PYTHON_PATH=""
    
    # Try to find Python elsewhere
    for alt_path in "python3" "python" "/usr/bin/python3" "/usr/bin/python"; do
        if command -v $alt_path >/dev/null 2>&1; then
            PYTHON_PATH=$alt_path
            log "✅ Alternative Python executable found: $PYTHON_PATH"
            break
        fi
    done
    
    if [ -z "$PYTHON_PATH" ]; then
        log "❌ No Python executable found"
    fi
fi

# Check script files
log "Checking script files..."
for script in "run_webapp.sh" "simple_http_server.py" "test_workflow.py" "test_python_env.py"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            log "✅ $script exists and is executable"
        else
            log "⚠️ $script exists but is not executable (fixing...)"
            chmod +x "$script"
            log "  → Made $script executable"
        fi
    else
        log "❌ $script does not exist"
    fi
done

# Check HTML files
log "Checking HTML files..."
for html in "index.html" "dashboard.html"; do
    if [ -f "$html" ]; then
        log "✅ $html exists"
    else
        log "⚠️ $html does not exist"
    fi
done

# Check ports
log "Checking port availability..."
for port in 5002 5001; do
    if (echo > /dev/tcp/localhost/$port) 2>/dev/null; then
        log "⚠️ Port $port is already in use"
    else
        log "✅ Port $port is available"
    fi
done

# Run Python environment test if Python is available
if [ -n "$PYTHON_PATH" ] && [ -f "test_python_env.py" ]; then
    log "Running Python environment test..."
    $PYTHON_PATH test_python_env.py > python_env_test.log
    
    # Extract summary results
    if grep -q "Test Summary" python_env_test.log; then
        log "Python environment test summary:"
        grep -A 10 "Test Summary" python_env_test.log | grep -v "=" | sed 's/^/  /'
    else
        log "⚠️ Python environment test did not produce expected output"
    fi
else
    log "⚠️ Skipping Python environment test (Python not available or test script missing)"
fi

# Check run_webapp.sh
log "Checking run_webapp.sh functionality..."
if grep -q "find_python()" run_webapp.sh && 
   grep -q "check_server()" run_webapp.sh && 
   grep -q "simple_http_server.py" run_webapp.sh; then
    log "✅ run_webapp.sh contains expected functions and references"
else
    log "⚠️ run_webapp.sh may be missing key components"
fi

# Check workflow configuration
log "Checking workflow configuration..."
if [ -f "workflow_config.json" ]; then
    log "✅ workflow_config.json exists"
    
    # Check if it contains required sections
    if grep -q "WebApp" workflow_config.json && 
       grep -q "TestWorkflow" workflow_config.json; then
        log "✅ workflow_config.json contains expected sections"
    else
        log "⚠️ workflow_config.json may be missing key sections"
    fi
else
    log "❌ workflow_config.json does not exist"
fi

# Check documentation
log "Checking documentation..."
if [ -f "WORKFLOW_README.md" ]; then
    log "✅ WORKFLOW_README.md exists"
else
    log "❌ WORKFLOW_README.md does not exist"
fi

# Check for server.log from previous runs
if [ -f "server.log" ]; then
    log "Previous server.log found. Last few lines:"
    tail -n 5 server.log | sed 's/^/  /'
fi

# Summary
log ""
log "Diagnostic Summary"
log "================="
log "1. Python executable: $([ -n "$PYTHON_PATH" ] && echo "✅ Found" || echo "❌ Not found")"
log "2. Required scripts: $([ -f "run_webapp.sh" ] && [ -f "simple_http_server.py" ] && echo "✅ Available" || echo "❌ Missing")"
log "3. HTML files: $([ -f "index.html" ] && [ -f "dashboard.html" ] && echo "✅ Available" || echo "⚠️ Incomplete")"
log "4. Configuration: $([ -f "workflow_config.json" ] && echo "✅ Available" || echo "❌ Missing")"
log "5. Documentation: $([ -f "WORKFLOW_README.md" ] && echo "✅ Available" || echo "❌ Missing")"

log ""
log "Next Steps:"
log "1. Run the WebApp workflow to start the server"
log "2. If the server fails to start, check server.log"
log "3. For more information, see WORKFLOW_README.md"