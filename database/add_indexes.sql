-- This file contains SQL commands to add indexes to the 'documents' table
-- in your Supabase database. These indexes are crucial for optimizing RAG
-- performance when filtering by class or checking for existing URLs, as 
-- required by src/refinery/embedding.py and src/shared/utils.py.

-- 1. Index on class_name metadata field (for faster RAG filtering)
CREATE INDEX CONCURRENTLY idx_documents_metadata_class ON documents 
  ((metadata->>'class_name'));

-- 2. Index on source_url metadata field (for faster de-duplication checks)
CREATE INDEX CONCURRENTLY idx_documents_metadata_url ON documents 
  ((metadata->>'source_url'));