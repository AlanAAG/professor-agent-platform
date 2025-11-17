-- ============================================================================
-- CRITICAL CONSTRAINT: This function expects 1024-dimensional embedding vectors
--
-- Compatible models:
--   - Mistral mistral-embed (1024 dimensions) - DEFAULT
--
-- INCOMPATIBLE models (will cause errors):
--   - OpenAI text-embedding-3-small (1536 dimensions)
--   - OpenAI text-embedding-3-large (3072 dimensions)
--
-- To change dimensions:
--   1. Update vector(1024) to vector(NEW_DIM) in this file
--   2. Update EXPECTED_EMBEDDING_DIM in src/shared/utils.py
--   3. Recreate the documents table with new vector dimension
-- ============================================================================
-- Supabase RPC: Semantic document retrieval using pgvector
-- Requires the `vector` extension and a `documents` table with an `embedding` column (vector(1024)).

CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(1024),
  match_threshold float,
  match_count int,
  filter_class text DEFAULT NULL
)
RETURNS TABLE (
  id uuid,
  content text,
  metadata jsonb,
  embedding vector(1024),
  similarity double precision,
  class_name text,
  title text,
  section text,
  url text
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
  effective_limit int := LEAST(GREATEST(COALESCE(match_count, 10), 1), 200);
  effective_threshold float := GREATEST(LEAST(COALESCE(match_threshold, 0.0), 1.0), -1.0);
  normalized_filter text := NULLIF(TRIM(filter_class), '');
BEGIN
  IF query_embedding IS NULL THEN
    RETURN;
  END IF;

  RETURN QUERY
  WITH scored AS (
    SELECT
      d.id,
      d.content,
      d.metadata,
      d.embedding,
      1 - (d.embedding <=> query_embedding) AS similarity,
      d.embedding <=> query_embedding AS distance,
      d.metadata->>'class_name' AS class_name,
      d.metadata->>'title' AS title,
      COALESCE(
        NULLIF(d.metadata->>'section', ''),
        NULLIF(d.metadata->>'chapter', ''),
        NULLIF(d.metadata->>'heading', '')
      ) AS section,
      COALESCE(
        NULLIF(d.metadata->>'url', ''),
        NULLIF(d.metadata->>'source_url', '')
      ) AS url
    FROM documents d
    WHERE d.embedding IS NOT NULL
      AND (normalized_filter IS NULL OR d.metadata->>'class_name' = normalized_filter)
  )
  SELECT
    s.id,
    s.content,
    s.metadata,
    s.embedding,
    s.similarity,
    s.class_name,
    s.title,
    s.section,
    s.url
  FROM scored AS s
  WHERE s.similarity >= effective_threshold
  ORDER BY s.distance ASC
  LIMIT effective_limit;
END;
$$;

