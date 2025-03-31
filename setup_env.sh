#!/bin/bash
# Set up environment variables for the application

# Session secret
export SESSION_SECRET="bcbs_values_session_secret_key_2025"

# Database credentials (from the create_postgresql_database_tool)
# Use DATABASE_URL from the environment if it exists, otherwise use a default
if [ -z "$DATABASE_URL" ]; then
  export DATABASE_URL="postgresql://${PGUSER}:${PGPASSWORD}@${PGHOST}:${PGPORT}/${PGDATABASE}"
fi

# Parse DATABASE_URL to set individual environment variables if they're not already set
if [ -z "$PGHOST" ] || [ -z "$PGPORT" ] || [ -z "$PGDATABASE" ] || [ -z "$PGUSER" ] || [ -z "$PGPASSWORD" ]; then
  # Extract components from DATABASE_URL
  regex="postgresql://([^:]+):([^@]+)@([^:]+):([^/]+)/(.+)"
  if [[ $DATABASE_URL =~ $regex ]]; then
    export PGUSER="${BASH_REMATCH[1]}"
    export PGPASSWORD="${BASH_REMATCH[2]}"
    export PGHOST="${BASH_REMATCH[3]}"
    export PGPORT="${BASH_REMATCH[4]}"
    export PGDATABASE="${BASH_REMATCH[5]}"
  else
    # Fallback values
    export PGHOST="${PGHOST:-localhost}"
    export PGPORT="${PGPORT:-5432}"
    export PGDATABASE="${PGDATABASE:-postgres}"
    export PGUSER="${PGUSER:-postgres}"
    export PGPASSWORD="${PGPASSWORD:-postgres}"
  fi
fi

# API keys
export API_KEY="bcbs_values_api_key_2025"
export BCBS_VALUES_API_KEY="bcbs_values_api_key_2025"

# External API keys
export NARRPR_API_KEY="narrpr_api_key_2025"
export NARRPR_API_SECRET="narrpr_api_secret_2025"
export MLS_API_KEY="mls_api_key_2025"
export PACS_API_KEY="pacs_api_key_2025"

# Other settings
export LOG_LEVEL="INFO"
export ENABLE_CACHING="true"
export NODE_ENV="development"
export PORT="5002"

# Echo the key environment variables
echo "Environment variables set!"
echo "DATABASE_URL=$DATABASE_URL"
echo "PGHOST=$PGHOST"
echo "PGPORT=$PGPORT"
echo "PGDATABASE=$PGDATABASE"
echo "PGUSER=$PGUSER"
echo "SESSION_SECRET=<hidden>"
echo "API_KEY=<hidden>"