-- database/add_indexes.sql
-- Performance indexes for documents table metadata queries
-- Run this in Supabase SQL Editor

-- Index for class_name filtering (used in RAG retrieval)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_metadata_class_name
ON documents ((metadata->>'class_name'));

-- Index for source_url lookups (used in deduplication)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_metadata_source_url
ON documents ((metadata->>'source_url'));

-- Index for section filtering (used in relevance boosting)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_metadata_section
ON documents ((metadata->>'section'));

-- Index for retrieval_date (used in recent content checks)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_metadata_retrieval_date
ON documents ((metadata->>'retrieval_date'));

-- Composite index for common query pattern: class + recent date
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_class_date
ON documents ((metadata->>'class_name'), (metadata->>'retrieval_date'));

-- Analyze table to update query planner statistics
ANALYZE documents;

-- Verify indexes were created
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'documents'
ORDER BY indexname;