-- ============================================================================
-- Hybrid Search (RRF) Function for documents_v2
-- Combines Vector Search (Semantics) and Keyword Search (BM25/TF-IDF via ts_rank)
-- using Reciprocal Rank Fusion (RRF).
-- ============================================================================

CREATE OR REPLACE FUNCTION match_documents_hybrid(
  query_embedding vector(768),
  query_text text,
  match_count int,
  filter_class text DEFAULT NULL,
  rrf_k int DEFAULT 60
)
RETURNS TABLE (
  id uuid,
  content text,
  metadata jsonb,
  embedding vector(768),
  similarity float,
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
  normalized_filter text := NULLIF(TRIM(filter_class), '');
BEGIN
  RETURN QUERY
  WITH vector_search AS (
    SELECT
      d.id,
      ROW_NUMBER() OVER (ORDER BY d.embedding <=> query_embedding ASC) as rank_v
    FROM documents_v2 d
    WHERE d.embedding IS NOT NULL
      AND (normalized_filter IS NULL OR d.metadata->>'class_name' = normalized_filter)
    ORDER BY d.embedding <=> query_embedding ASC
    LIMIT effective_limit * 2
  ),
  keyword_search AS (
    SELECT
      d.id,
      ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', d.content), plainto_tsquery('english', query_text)) DESC) as rank_k
    FROM documents_v2 d
    WHERE (normalized_filter IS NULL OR d.metadata->>'class_name' = normalized_filter)
      AND to_tsvector('english', d.content) @@ plainto_tsquery('english', query_text)
    ORDER BY ts_rank_cd(to_tsvector('english', d.content), plainto_tsquery('english', query_text)) DESC
    LIMIT effective_limit * 2
  ),
  combined_scores AS (
    SELECT
      COALESCE(v.id, k.id) as id,
      (COALESCE(1.0 / (rrf_k + v.rank_v), 0.0) + COALESCE(1.0 / (rrf_k + k.rank_k), 0.0))::float as similarity
    FROM vector_search v
    FULL OUTER JOIN keyword_search k ON v.id = k.id
  )
  SELECT
    d.id,
    d.content,
    d.metadata,
    d.embedding,
    c.similarity,
    d.metadata->>'class_name' as class_name,
    d.metadata->>'title' as title,
    COALESCE(
        NULLIF(d.metadata->>'section', ''),
        NULLIF(d.metadata->>'chapter', ''),
        NULLIF(d.metadata->>'heading', '')
      ) AS section,
      COALESCE(
        NULLIF(d.metadata->>'url', ''),
        NULLIF(d.metadata->>'source_url', '')
      ) AS url
  FROM combined_scores c
  JOIN documents_v2 d ON c.id = d.id
  ORDER BY c.similarity DESC
  LIMIT effective_limit;
END;
$$;
