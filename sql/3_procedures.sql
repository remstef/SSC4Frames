-- ============================================================================
-- ============================================================================
-- ============================================================================

-- Procedure to remove clusterid as foreign key from clusterembedding tables
CREATE OR REPLACE PROCEDURE drop_clusterembeddings_fkclusterid()
LANGUAGE plpgsql
AS $$
DECLARE _records CURSOR FOR
  select * 
  from clusterings;
BEGIN
  FOR _record IN _records LOOP
    -- RAISE NOTICE '%', _record.id;
    -- OBJECT_ID(format('clusterembeddings__%s_clusterid_fkey', _record.id), 'F')
    IF EXISTS (
      select 1 
      from information_schema.table_constraints 
      where table_schema='public' 
        and table_name=format('clusterembeddings__%s', _record.id) 
        and constraint_name=format('clusterembeddings__%s_clusterid_fkey', _record.id) 
    ) THEN
      RAISE NOTICE 'Dropping clusterembeddings foreign key if exists: clusterembeddings__%', _record.id;
      EXECUTE format('alter table if exists public.clusterembeddings__%s drop CONSTRAINT clusterembeddings__%s_clusterid_fkey', _record.id,_record.id);
      COMMIT;
    END IF;
  END LOOP;
END
$$;

-- alternative implementation:
CREATE OR REPLACE PROCEDURE drop_clusterembeddings_fkclusterid()
LANGUAGE plpgsql
AS $$
DECLARE _records CURSOR FOR
  select * 
  from clusterings;
BEGIN
  FOR _record IN _records LOOP
    RAISE NOTICE '%', _record.id;
    RAISE NOTICE 'Dropping clusterembeddings foreign key if exists: clusterembeddings__%', _record.id;
    EXECUTE format('alter table if exists public.clusterembeddings__%s drop CONSTRAINT if exists clusterembeddings__%s_clusterid_fkey', _record.id,_record.id);
  END LOOP;
END
$$;

-- EXAMPLE:
-- call drop_clusterembeddings_fkclusterid();

-- ============================================================================
-- ============================================================================
-- ============================================================================
-- Procedure to remove a single clustering with a certain id
-- in a batched fashion with commits after every table deletion 
-- (an alternative to function delclustering(...) )
CREATE OR REPLACE PROCEDURE delclustering_proc(_clusteringid integer, _batch_size integer DEFAULT 10000)
LANGUAGE plpgsql
AS $$
DECLARE
  rows_deleted int;
BEGIN
  -- Drop clusterembeddings table if exists
  RAISE NOTICE 'Dropping clusterembeddings table if exists: clusterembeddings__%', _clusteringid;
  EXECUTE format('DROP TABLE IF EXISTS clusterembeddings__%s', _clusteringid);
  
  COMMIT;

  -- Drop vectorized frameinstances view/matview if exists
  RAISE NOTICE 'Dropping vectorized frameinstances view if exists: frameinstances_split_vectorized__%', _clusteringid;
  IF EXISTS (
    SELECT 1 
    FROM pg_matviews 
    WHERE schemaname = 'public' AND matviewname = format('frameinstances_split_vectorized__%s', _clusteringid)
  ) THEN
    EXECUTE format('DROP MATERIALIZED VIEW IF EXISTS frameinstances_split_vectorized__%s', _clusteringid);
  ELSEIF EXISTS (
    SELECT 1
    FROM pg_views
    WHERE schemaname = 'public' AND viewname = format('frameinstances_split_vectorized__%s', _clusteringid)
  ) THEN
    EXECUTE format('DROP VIEW IF EXISTS frameinstances_split_vectorized__%s', _clusteringid);
  END IF;
  
  COMMIT;

  -- Delete from clusterassignments in batches
  LOOP
    RAISE NOTICE 'Deleting batch of rows from clusterassignments';
    -- DELETE FROM clusterassignments ca
    -- USING clusters cl
    -- WHERE ca.clusterid = cl.id
    --   AND cl.clusteringid = _clusteringid
    -- LIMIT _batch_size;
    DELETE FROM clusterassignments
    WHERE clusterid IN (
      SELECT ca.clusterid
      FROM clusterassignments ca
      JOIN clusters cl ON ca.clusterid = cl.id
      WHERE cl.clusteringid = _clusteringid
      LIMIT _batch_size
    );

    GET DIAGNOSTICS rows_deleted = ROW_COUNT;

    COMMIT;  -- commit batch
    EXIT WHEN rows_deleted = 0;
  END LOOP;

  -- Delete from clusters in batches
  LOOP
    RAISE NOTICE 'Deleting batch of rows from clusters';
    DELETE FROM clusters
    WHERE id IN (
      SELECT id
      FROM clusters
      WHERE clusteringid = _clusteringid
      LIMIT _batch_size
    );

    GET DIAGNOSTICS rows_deleted = ROW_COUNT;

    COMMIT;  -- commit batch
    EXIT WHEN rows_deleted = 0;
  END LOOP;

  -- Finally, delete the clustering row
  RAISE NOTICE 'Deleting row from clusterings';
  DELETE FROM clusterings 
  WHERE id = _clusteringid;

  COMMIT;

  RAISE NOTICE 'Deletion of clustering % complete', _clusteringid;

END;
$$;

-- EXAMPLE 1:
-- CALL cleanup_clusters(1854);
--
-- EXAMPLE 2: delete multiple clusterings
-- DO 
-- $$
-- DECLARE _records CURSOR FOR
--  select * 
--    from clusterings c 
--    where status != 'finished';
-- BEGIN
--   FOR _record IN _records LOOP
--     RAISE NOTICE 'Dropping clustering with id %', _record.id;
-- 	   CALL delclustering_proc(_record.id);
--     COMMIT;
-- 	 END LOOP;
-- END
-- $$;