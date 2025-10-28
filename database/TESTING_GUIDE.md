# Database Index Testing Guide

## Pre-Testing Setup

1. **Backup your database** (recommended for production environments)
2. **Note current query times** by running these queries in Supabase SQL Editor:

```sql
-- Test query performance BEFORE adding indexes
EXPLAIN ANALYZE SELECT COUNT(*) FROM documents WHERE metadata->>'source_url' = 'https://example.com';
EXPLAIN ANALYZE SELECT COUNT(*) FROM documents WHERE metadata->>'class_name' = 'business_fundamentals';
EXPLAIN ANALYZE SELECT COUNT(*) FROM documents WHERE metadata->>'retrieval_date' > '2024-01-01T00:00:00';
```

## Step 1: Run the Index Migration

1. Open Supabase Dashboard → SQL Editor
2. Copy the entire contents of `database/add_indexes.sql`
3. Paste and click "Run"
4. Wait 30-60 seconds for `CONCURRENTLY` indexes to build
5. Verify no errors in the output

## Step 2: Verify Index Creation

Run this query to confirm all 5 indexes were created:

```sql
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'documents'
AND indexname LIKE 'idx_documents_%'
ORDER BY indexname;
```

**Expected Output:** You should see 5 indexes:
- `idx_documents_class_date`
- `idx_documents_metadata_class_name`
- `idx_documents_metadata_retrieval_date`
- `idx_documents_metadata_section`
- `idx_documents_metadata_source_url`

## Step 3: Test Query Performance

Re-run the same queries from pre-testing setup:

```sql
-- Test query performance AFTER adding indexes
EXPLAIN ANALYZE SELECT COUNT(*) FROM documents WHERE metadata->>'source_url' = 'https://example.com';
EXPLAIN ANALYZE SELECT COUNT(*) FROM documents WHERE metadata->>'class_name' = 'business_fundamentals';
EXPLAIN ANALYZE SELECT COUNT(*) FROM documents WHERE metadata->>'retrieval_date' > '2024-01-01T00:00:00';
```

**Success Criteria:**
- Execution time should be 10-50x faster
- Query plans should show "Index Scan" instead of "Seq Scan"
- Cost estimates should be significantly lower

## Step 4: Test Pipeline Integration

1. **Restart your application** to trigger the index verification
2. **Check logs** for this message:
   ```
   ✅ Database indexes verified (5 found)
   ```
3. **If you see a warning**, the indexes weren't created properly:
   ```
   ⚠️  Only X/4 recommended indexes found on 'documents' table
   ```

## Step 5: Functional Testing

Test the key functions that benefit from indexes:

```python
# Test in Python console or create a test script
from src.refinery.embedding import url_exists_in_db_sync, check_if_embedded_recently_sync
import time

# Test URL existence check (should be very fast now)
start = time.time()
exists = url_exists_in_db_sync("https://example.com")
duration = time.time() - start
print(f"URL check took {duration*1000:.1f}ms (should be <10ms)")

# Test recent embedding check (should be very fast now)
start = time.time()
recent = check_if_embedded_recently_sync({"class_name": "business_fundamentals"})
duration = time.time() - start
print(f"Recent check took {duration*1000:.1f}ms (should be <15ms)")
```

## Success Criteria Checklist

- ✅ All 5 indexes visible in `pg_indexes`
- ✅ Query times reduced by at least 10x
- ✅ No errors in pipeline startup logs
- ✅ Index verification passes on startup
- ✅ `url_exists_in_db_sync()` completes in <10ms
- ✅ `check_if_embedded_recently_sync()` completes in <15ms

## Troubleshooting

### If indexes aren't created:
- Check for typos in the SQL
- Ensure you have sufficient database permissions
- Try running without `CONCURRENTLY` if there are lock issues

### If verification fails:
- Check that the `exec_sql` RPC function exists in Supabase
- Verify the index names match exactly
- Check Supabase logs for any errors

### If performance doesn't improve:
- Run `ANALYZE documents;` to update statistics
- Check that your test queries actually use the indexed columns
- Verify you have enough data to see meaningful performance differences

## Monitoring Index Usage

After deployment, monitor index effectiveness:

```sql
-- Check index usage statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
WHERE tablename = 'documents'
ORDER BY idx_scan DESC;
```

Indexes with low `idx_scan` values may not be providing value and could be candidates for removal.