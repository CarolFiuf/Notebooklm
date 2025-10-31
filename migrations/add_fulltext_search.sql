-- Migration: Add Full-Text Search support to document_chunks table
-- This improves keyword search performance by 10-100x
-- Idempotent: Safe to run multiple times

-- ============================================
-- STEP 1: Check if migration already applied
-- ============================================
DO $$
BEGIN
    -- Check if migration already exists
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'document_chunks' AND column_name = 'content_tsv'
    ) THEN
        RAISE NOTICE 'Full-Text Search migration already applied. Skipping...';
    ELSE
        RAISE NOTICE 'Applying Full-Text Search migration...';

        -- Step 1: Add tsvector column for full-text search
        ALTER TABLE document_chunks
        ADD COLUMN content_tsv tsvector;

        RAISE NOTICE '✓ Added content_tsv column';

        -- Step 2: Create GIN index on tsvector column
        CREATE INDEX idx_document_chunks_content_tsv
        ON document_chunks USING GIN(content_tsv);

        RAISE NOTICE '✓ Created GIN index';

        -- Step 3: Create function to update tsvector
        CREATE OR REPLACE FUNCTION document_chunks_content_tsv_trigger() RETURNS trigger AS $trigger$
        BEGIN
          NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
          RETURN NEW;
        END
        $trigger$ LANGUAGE plpgsql;

        RAISE NOTICE '✓ Created trigger function';

        -- Step 4: Create trigger to automatically update tsvector on INSERT/UPDATE
        CREATE TRIGGER tsvector_update
        BEFORE INSERT OR UPDATE ON document_chunks
        FOR EACH ROW EXECUTE FUNCTION document_chunks_content_tsv_trigger();

        RAISE NOTICE '✓ Created trigger';

        -- Step 5: Populate existing rows with tsvector data (if any)
        UPDATE document_chunks
        SET content_tsv = to_tsvector('english', COALESCE(content, ''))
        WHERE content_tsv IS NULL;

        RAISE NOTICE '✓ Populated existing rows';

        RAISE NOTICE '✅ Full-Text Search migration completed successfully!';
    END IF;
END $$;

-- ============================================
-- STEP 2: Add additional performance indexes
-- ============================================
CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id
ON document_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id_chunk_idx
ON document_chunks(document_id, chunk_index);

-- ============================================
-- Verification Query
-- ============================================
-- Run this to verify the migration:
-- SELECT
--   column_name,
--   data_type,
--   is_nullable
-- FROM information_schema.columns
-- WHERE table_name = 'document_chunks' AND column_name = 'content_tsv';

-- ============================================
-- Performance Test Query
-- ============================================
-- Test FTS performance:
-- EXPLAIN ANALYZE
-- SELECT id, content
-- FROM document_chunks
-- WHERE content_tsv @@ plainto_tsquery('english', 'machine learning')
-- LIMIT 10;