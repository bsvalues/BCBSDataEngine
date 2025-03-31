#!/bin/bash
# Simple diagnostic script for BCBS Values

echo "======================= BCBS Values Diagnostics ======================="
echo "Date: $(date)"
echo ""

# Check environment variables
echo "==== Environment Variables ===="
echo "Checking critical environment variables..."

check_env_var() {
  if [ -z "${!1}" ]; then
    echo "❌ $1 is not set"
    return 1
  else
    echo "✅ $1 is set"
    return 0
  fi
}

# Database variables
check_env_var "DATABASE_URL" 
check_env_var "PGHOST"
check_env_var "PGPORT"
check_env_var "PGDATABASE"
check_env_var "PGUSER"
check_env_var "PGPASSWORD"

# Session variables
check_env_var "SESSION_SECRET"

# API keys
check_env_var "API_KEY" || check_env_var "BCBS_VALUES_API_KEY"
check_env_var "NARRPR_API_KEY"
check_env_var "MLS_API_KEY"
check_env_var "PACS_API_KEY"

echo ""
echo "==== File System Check ===="
echo "Checking for critical files and directories..."

check_file() {
  if [ -f "$1" ]; then
    echo "✅ $1 exists"
    return 0
  else
    echo "❌ $1 does not exist"
    return 1
  fi
}

check_dir() {
  if [ -d "$1" ]; then
    echo "✅ $1 directory exists"
    return 0
  else
    echo "❌ $1 directory does not exist"
    return 1
  fi
}

# Check main files
check_file "diagnose_env.py"
check_file ".env"
check_file "pyproject.toml"
check_file "configs/database_config.json"

# Check directories
check_dir "src"
check_dir "api"
check_dir "utils"
check_dir "etl"
check_dir "agents"
check_dir "tests"

echo ""
echo "==== Database Connectivity ===="
echo "Attempting to connect to PostgreSQL database using Python..."

# Try to connect using Python
pg_working=0

# Create a temporary Python script
cat > test_db_connection.py << EOF
import os
import sys
from urllib.parse import urlparse

try:
    # Try to import necessary modules
    import psycopg2
    from sqlalchemy import create_engine, text
    
    # Get database URL from environment
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("❌ DATABASE_URL environment variable is not set")
        sys.exit(1)
    
    # Parse connection details
    result = urlparse(db_url)
    print(f"Attempting to connect to {result.hostname}:{result.port}/{result.path[1:]}")
    
    # Try direct psycopg2 connection
    try:
        conn = psycopg2.connect(
            database=result.path[1:],
            user=result.username,
            password=result.password,
            host=result.hostname,
            port=result.port
        )
        conn.close()
        print("✅ Successfully connected using psycopg2")
    except Exception as e:
        print(f"❌ Failed to connect using psycopg2: {str(e)}")
        raise
    
    # Try SQLAlchemy connection
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            assert result.scalar() == 1
        print("✅ Successfully connected using SQLAlchemy")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to connect using SQLAlchemy: {str(e)}")
        sys.exit(1)
except ImportError as e:
    print(f"❌ Missing required Python module: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {str(e)}")
    sys.exit(1)
EOF

# Run the Python script
echo "Running Python database connectivity test..."
if command -v python3 &> /dev/null; then
    if python3 test_db_connection.py; then
        echo "✅ Database connection successful"
        pg_working=1
    else
        echo "❌ Database connection failed"
    fi
elif command -v python &> /dev/null; then
    if python test_db_connection.py; then
        echo "✅ Database connection successful"
        pg_working=1
    else
        echo "❌ Database connection failed"
    fi
else
    echo "❌ Python not found. Database connection test skipped."
fi

# Clean up temporary file
rm -f test_db_connection.py

echo ""
echo "==== Summary ===="
if [ $pg_working -eq 1 ]; then
  echo "Database connection: ✅ Working"
else
  echo "Database connection: ❌ Not working"
fi

echo ""
echo "==== Next Steps ===="
echo "1. If environment variables are missing, run source setup_env.sh"
echo "2. If database connection failed, check database credentials"
echo "3. If files or directories are missing, check the repository structure"

echo ""
echo "===================== Diagnostics Complete ====================="