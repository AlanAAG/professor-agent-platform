# Database Maintenance

## Adding Indexes

Run `add_indexes.sql` in your Supabase SQL Editor:

1. Go to Supabase Dashboard > SQL Editor
2. Paste contents of `add_indexes.sql`
3. Click "Run"
4. Wait 30-60 seconds for CONCURRENTLY indexes to build
5. Verify with the SELECT query at the end

## Expected Performance Gains

| Query Type | Before | After | Speedup |
|------------|--------|-------|---------|
| `url_exists_in_db_sync()` | ~200ms | ~5ms | 40x |
| `check_if_embedded_recently_sync()` | ~150ms | ~8ms | 18x |
| RAG retrieval with class filter | ~100ms | ~15ms | 6x |

## Monitoring Index Usage
```sql
-- Check if indexes are being used
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read
FROM pg_stat_user_indexes
WHERE tablename = 'documents'
ORDER BY idx_scan DESC;
```

## Index Maintenance

Indexes are automatically maintained by PostgreSQL. However, if you see performance degradation after many inserts:
```sql
REINDEX TABLE CONCURRENTLY documents;
ANALYZE documents;
```