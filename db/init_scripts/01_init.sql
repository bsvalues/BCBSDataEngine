-- Initialize database extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Set up application schema if not exists
CREATE SCHEMA IF NOT EXISTS bcbs_values;

-- Create role for application if not exists
DO
$$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'bcbs_app_role') THEN
    CREATE ROLE bcbs_app_role WITH LOGIN PASSWORD 'secure_password_placeholder';
  END IF;
END
$$;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON SCHEMA bcbs_values TO bcbs_app_role;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA bcbs_values TO bcbs_app_role;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA bcbs_values TO bcbs_app_role;

-- These tables will be created by SQLAlchemy when the application starts
-- This script only ensures the database is properly configured with extensions and permissions