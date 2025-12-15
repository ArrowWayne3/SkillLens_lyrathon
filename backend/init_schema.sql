-- 1. Enable extensions for AI (Vectors)
CREATE EXTENSION IF NOT EXISTS vector;

-- ==========================================
-- 2. USERS & ROLES
-- ==========================================
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('candidate', 'recruiter')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for faster email lookups
CREATE INDEX idx_users_email ON users(email);

-- ==========================================
-- 3. PROFILES & EMBEDDINGS (For AI Matching)
-- ==========================================
CREATE TABLE candidate_profiles (
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    skills TEXT[],
    experience_years INT,
    salary_min INT,
    salary_max INT,
    work_mode TEXT CHECK (work_mode IN ('remote', 'hybrid', 'onsite')),
    bio TEXT,
    embedding vector(384),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id)
);

CREATE TABLE recruiter_profiles (
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    company_name TEXT,
    industry TEXT,
    team_size TEXT,
    hiring_priority TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id)
);

CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    recruiter_id INT REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    salary_min INT,
    salary_max INT,
    required_skills TEXT[],
    work_mode TEXT CHECK (work_mode IN ('remote', 'hybrid', 'onsite')),
    is_active BOOLEAN DEFAULT TRUE,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for vector similarity search
CREATE INDEX idx_candidate_embedding ON candidate_profiles USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_jobs_embedding ON jobs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ==========================================
-- 4. APPLICATIONS & MATCHES
-- ==========================================
CREATE TABLE applications (
    id SERIAL PRIMARY KEY,
    candidate_id INT REFERENCES users(id) ON DELETE CASCADE,
    job_id INT REFERENCES jobs(id) ON DELETE CASCADE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'reviewed', 'interviewing', 'accepted', 'rejected')),
    match_score DECIMAL(5,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(candidate_id, job_id)
);

CREATE TABLE recruiter_interests (
    id SERIAL PRIMARY KEY,
    recruiter_id INT REFERENCES users(id) ON DELETE CASCADE,
    candidate_id INT REFERENCES users(id) ON DELETE CASCADE,
    job_id INT REFERENCES jobs(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(recruiter_id, candidate_id, job_id)
);

-- ==========================================
-- 5. MARKET TRENDS (Time-Series Hypertable)
-- ==========================================
CREATE TABLE skill_trends (
    time TIMESTAMPTZ NOT NULL,
    skill_name TEXT NOT NULL,
    source TEXT,
    weight INT DEFAULT 1
);

-- Convert standard table to Timescale Hypertable
SELECT create_hypertable('skill_trends', 'time');

-- Create continuous aggregate for faster trend queries
CREATE MATERIALIZED VIEW skill_trends_daily
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS bucket,
    skill_name,
    SUM(weight) AS total_weight,
    COUNT(*) AS occurrences
FROM skill_trends
GROUP BY bucket, skill_name;

-- ==========================================
-- 6. MESSAGES (Chat)
-- ==========================================
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    sender_id INT REFERENCES users(id) ON DELETE CASCADE,
    receiver_id INT REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_messages_conversation ON messages(sender_id, receiver_id, created_at DESC);