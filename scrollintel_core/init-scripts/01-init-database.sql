-- ================================
-- ScrollIntel Core - Database Initialization
-- Creates database and user with proper permissions
-- ================================

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE scrollintel_core'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'scrollintel_core')\gexec

-- Connect to the database
\c scrollintel_core;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for better performance
-- These will be created by SQLAlchemy, but we can add custom ones here if needed

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE scrollintel_core TO scrollintel;