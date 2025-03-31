-- BCBS Values Database Schema

-- Properties table
CREATE TABLE IF NOT EXISTS properties (
  property_id SERIAL PRIMARY KEY,
  address TEXT NOT NULL,
  neighborhood TEXT,
  property_type TEXT NOT NULL,
  land_area NUMERIC,
  living_area NUMERIC,
  bedrooms INTEGER,
  bathrooms NUMERIC,
  year_built INTEGER,
  estimated_value NUMERIC NOT NULL,
  confidence_score NUMERIC NOT NULL,
  valuation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  valuation_method TEXT,
  latitude NUMERIC,
  longitude NUMERIC,
  last_sale_date TIMESTAMP,
  last_sale_price NUMERIC,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
  agent_id TEXT PRIMARY KEY,
  agent_type TEXT NOT NULL,
  status TEXT NOT NULL,
  last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  error TEXT,
  total_processed INTEGER DEFAULT 0,
  success_rate NUMERIC DEFAULT 0,
  average_processing_time NUMERIC DEFAULT 0,
  queue_size INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent logs table
CREATE TABLE IF NOT EXISTS agent_logs (
  log_id SERIAL PRIMARY KEY,
  agent_id TEXT NOT NULL REFERENCES agents(agent_id),
  level TEXT NOT NULL,
  message TEXT NOT NULL,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_agent FOREIGN KEY (agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE
);

-- ETL Jobs table
CREATE TABLE IF NOT EXISTS etl_jobs (
  job_id SERIAL PRIMARY KEY,
  job_type TEXT NOT NULL,
  status TEXT NOT NULL,
  start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  end_time TIMESTAMP,
  records_processed INTEGER DEFAULT 0,
  success_rate NUMERIC DEFAULT 0,
  average_processing_time NUMERIC DEFAULT 0,
  error TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ETL Sources table
CREATE TABLE IF NOT EXISTS etl_sources (
  source_id SERIAL PRIMARY KEY,
  job_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  records INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_job FOREIGN KEY (job_id) REFERENCES etl_jobs(job_id) ON DELETE CASCADE
);

-- ETL Data Quality table
CREATE TABLE IF NOT EXISTS etl_data_quality (
  quality_id SERIAL PRIMARY KEY,
  job_id INTEGER NOT NULL,
  completeness NUMERIC DEFAULT 0,
  accuracy NUMERIC DEFAULT 0,
  timeliness NUMERIC DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_job FOREIGN KEY (job_id) REFERENCES etl_jobs(job_id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_properties_neighborhood ON properties(neighborhood);
CREATE INDEX IF NOT EXISTS idx_properties_property_type ON properties(property_type);
CREATE INDEX IF NOT EXISTS idx_properties_valuation_date ON properties(valuation_date);
CREATE INDEX IF NOT EXISTS idx_properties_estimated_value ON properties(estimated_value);
CREATE INDEX IF NOT EXISTS idx_properties_coordinates ON properties(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_logs_agent_id ON agent_logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_logs_timestamp ON agent_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_etl_jobs_status ON etl_jobs(status);
CREATE INDEX IF NOT EXISTS idx_etl_jobs_job_type ON etl_jobs(job_type);