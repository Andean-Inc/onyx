-- init_andean_db.sql

-- Create the database if it doesn't already exist
CREATE DATABASE andean_db;

-- Connect to the newly created database to ensure subsequent commands apply to it.
\c andean_db;

-- app user table
CREATE TABLE IF NOT EXISTS users (
    uid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    owned_projects UUID[] DEFAULT '{}',
    edit_projects UUID[] DEFAULT '{}',
    view_projects UUID[] DEFAULT '{}',
    templates JSONB DEFAULT '[]'::jsonb
);


CREATE ROLE andean_admin WITH LOGIN PASSWORD 'comparch-18447';

-- Grant schema permissions
GRANT CREATE, USAGE ON SCHEMA public TO andean_admin;

-- Grant privileges to the andean_admin role
GRANT ALL PRIVILEGES ON DATABASE andean_db TO andean_admin;
GRANT ALL PRIVILEGES ON TABLE users TO andean_admin;
GRANT CONNECT ON DATABASE andean_db TO andean_admin;