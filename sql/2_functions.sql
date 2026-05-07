-- ============================================================================
-- ============================================================================
-- ============================================================================

-- a function to delete a specific clustering
CREATE OR REPLACE FUNCTION delclustering(_clusteringid integer)
  RETURNS SETOF clusterings
  LANGUAGE plpgsql AS
$func$
BEGIN
  --
  RAISE NOTICE 'Dropping clustering %', _clusteringid;
  SET CONSTRAINTS ALL DEFERRED;
	RAISE NOTICE 'Dropping clusterembeddings table if exists: clusterembeddings__%', _clusteringid;
	EXECUTE format('drop table if exists clusterembeddings__%s', _clusteringid);
	--
	RAISE NOTICE 'Dropping vectorized frameinstances view if exists: frameinstances_split_vectorized__%', _clusteringid;
  IF EXISTS (select matviewname from pg_matviews where schemaname = 'public' and matviewname = format('frameinstances_split_vectorized__%s', _clusteringid)) THEN
    EXECUTE format('drop materialized view if exists frameinstances_split_vectorized__%s', _clusteringid);
  ELSEIF EXISTS (select viewname from pg_views where schemaname = 'public' and viewname = format('frameinstances_split_vectorized__%s', _clusteringid)) THEN
    EXECUTE format('drop view if exists frameinstances_split_vectorized__%s', _clusteringid);
  END IF;
  --
	RAISE NOTICE 'Deleting rows from clusterassignments';
	delete from clusterassignments ca 
	using clusters cl
	where ca.clusterid = cl.id
	and cl.clusteringid = _clusteringid;
	RAISE NOTICE 'Deleting rows from clusters';
	delete from clusters 
	where clusteringid = _clusteringid;
	RAISE NOTICE 'Deleting row from clusterings';
	RETURN QUERY
    delete from clusterings
    where id = _clusteringid
    returning *;
END
$func$;

-- EXAMPLE 1:
-- select * from delclustering(1)

-- EXAMPLE 2:
-- select delclustering(c.id) from clusterings c where status != 'finished'

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- handle with care!!
-- a function to delete all clusterings
CREATE OR REPLACE FUNCTION delclustering_all()
  RETURNS SETOF clusterings
  LANGUAGE plpgsql AS
$func$
DECLARE
	_clusterings CURSOR FOR
		SELECT id
		FROM clusterings;
BEGIN
	FOR _clustering IN _clusterings LOOP
		RAISE NOTICE 'Dropping clusterembeddings table if exists: clusterembeddings__%', _clustering.id;
		EXECUTE format('drop table if exists clusterembeddings__%s', _clustering.id);
		RAISE NOTICE 'Dropping vectorized frameinstances view if exists: frameinstances_split_vectorized__%', _clustering.id;
    IF EXISTS (select matviewname from pg_matviews where schemaname = 'public' and matviewname = format('frameinstances_split_vectorized__%s', _clustering.id)) THEN
      EXECUTE format('drop materialized view if exists frameinstances_split_vectorized__%s', _clustering.id);
    ELSEIF EXISTS (select viewname from pg_views where schemaname = 'public' and viewname = format('frameinstances_split_vectorized__%s', _clustering.id)) THEN
      EXECUTE format('drop view if exists frameinstances_split_vectorized__%s', _clustering.id);
    END IF;
	END LOOP;
	RAISE NOTICE 'Deleting rows from clusterassignments';
	delete from clusterassignments ca;
	RAISE NOTICE 'Deleting rows from clusters';
	delete from clusters;
	RAISE NOTICE 'Deleting rows from clusterings';
	RETURN QUERY
    delete from clusterings
    returning *;
END
$func$;

-- EXAMPLE: handle with care!!
-- select * from delclustering_all()

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- a function to delete a specific experiment
CREATE OR REPLACE FUNCTION delexperiment(_experimentid integer)
  RETURNS SETOF experiments
  LANGUAGE plpgsql AS
$func$
BEGIN
	RAISE NOTICE 'Deleting rows from experiment_runs';
	delete from experiment_runs
	where experiment_id = _experimentid;
	RAISE NOTICE 'Deleting row from experiments';
	RETURN QUERY
    delete from experiments
    where id = _experimentid
    returning *;
END
$func$;

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- a function to delete all experiments
CREATE OR REPLACE FUNCTION delexperiment_all()
  RETURNS SETOF experiments
  LANGUAGE plpgsql AS
$func$
BEGIN
	RAISE NOTICE 'Deleting rows from experiment_runs';
	delete from experiment_runs;
	RAISE NOTICE 'Deleting row from experiments';
	RETURN QUERY
    delete from experiments
    returning *;
END
$func$;


-- ============================================================================
-- ============================================================================
-- ============================================================================

-- get all possible labels from a dataset 
CREATE OR REPLACE FUNCTION labels_dataset_all(ds_id integer)
  RETURNS TABLE (
    label frameinstances.frame_label%TYPE,
    numinstances bigint,
    instances integer[]
  )
  LANGUAGE plpgsql AS
$func$
BEGIN
	RETURN QUERY
    select 
      frame_label as label, 
      count(*) as numinstances, 
      array_agg(id) as instances
    from frameinstances 
    where dataset_id = ds_id
    group by frame_label
    order by numinstances desc;
END
$func$;

-- EXAMPLE 1:
-- select * from labels_dataset_all(2)

-- EXAMPLE 2:
-- select * from labels_dataset_all((select id from datasets where name='fn1.7'))

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- get all possible labels from a datasetsplit 
CREATE OR REPLACE FUNCTION labels_split(ds_split_id integer, splits text[])
  RETURNS TABLE (
    label frameinstances.frame_label%TYPE,
    support bigint,
    instances integer[]
  )
  LANGUAGE plpgsql AS
$func$
BEGIN
	RETURN QUERY
    with si_slct as (
      select instance_id 
      from split_instances where datasetsplit_id = ds_split_id and split=ANY(splits)
    ), sifi_slct as (
      select fi.id as instanceid, fi.frame_label as truelabel 
      from si_slct 
      inner join frameinstances fi on si_slct.instance_id = fi.id
    ), known_labels as (
      select truelabel as label, count(truelabel) as support, array_agg(instanceid) as instances 
      from sifi_slct 
      group by truelabel
      order by support desc
    )
    select 
      kn.label,
      kn.support,
      kn.instances
    from known_labels kn;
END
$func$;

-- EXAMPLE:
-- select * from labels_split(
-- 	3,
-- 	array['train']
-- )

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- get all "known" labels i.e. labels from the training instances from a clustering
CREATE OR REPLACE FUNCTION knownlabels(q_clusteringid integer)
  RETURNS TABLE (
    label frameinstances.frame_label%TYPE,
    support bigint,
    instances integer[]
  )
  LANGUAGE plpgsql AS
$func$
DECLARE
  _ds_split_id integer;
	_testsplits text[];
	_trainsplits text[];
BEGIN
  select cl.datasetsplit_id into _ds_split_id from clusterings cl where cl.id = q_clusteringid;
	RAISE NOTICE 'DatasetSplit id : %', _ds_split_id;
	select translate((setting->'data'->'testsplits')::text, '[]','{}')::text[] into _testsplits from clusterings where id = q_clusteringid;
	RAISE NOTICE 'Testsplits : %', _testsplits;
	with arr as (
		select translate((setting->'data'->'splits')::text, '[]','{}')::text[] as ar from clusterings where id = q_clusteringid
	), items as (
		select unnest(ar) as item from arr
	), itemsremoved as (
		select item from items where not item=any(_testsplits)
	), itemsremovedasarr as (
		select array_agg(item) as newarr from itemsremoved
	)
	select newarr into _trainsplits from itemsremovedasarr;
	RAISE NOTICE 'Trainsplits : %', _trainsplits;
	RETURN QUERY
    select *
    from labels_split(
      _ds_split_id,
      _trainsplits
    );
END
$func$;

-- EXAMPLE:
-- select * from knownlabels(
-- 	36
-- )

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- get a full table of true labels and assigned labels of the split_instances for a clustering and the specified datasetsplit
CREATE OR REPLACE FUNCTION evaltable(q_clusteringid integer, q_ds_split_id integer, q_testsplits text[])
  RETURNS TABLE (
    instanceid frameinstances.id%TYPE,
    split split_instances.split%TYPE,
    label_true frameinstances.frame_label%TYPE,
    clabel clusters.label%TYPE,
    lu_lemma frameinstances.lu_lemma%TYPE,
    cid clusters.id%TYPE,
    clusterinfo clusters.extrainfo%TYPE,
    assignmentinfo clusterassignments.extrainfo%TYPE,
    instanceinfo frameinstances.extrainfo%TYPE
  )
  LANGUAGE plpgsql AS
$func$
BEGIN
	RETURN QUERY
    with si_test_slct as (
      select * from split_instances si where si.datasetsplit_id = q_ds_split_id and si.split=ANY(q_testsplits)
    ), cl_slct as (
      select * from clusters where clusteringid = q_clusteringid
    ), ca_cl as (
      select ca.clusterid, ca.instanceid, cl_slct.label as clusterlabel, ca.extrainfo as assignmentinfo, cl_slct.extrainfo as clusterinfo 
      from clusterassignments ca
      inner join cl_slct on cl_slct.id = ca.clusterid
    ), sifi_test as (
      select fi.id as instanceid, fi.frame_label as truelabel, fi.lu_lemma as lu_lemma, si_test_slct.split, si_test_slct.datasetsplit_id, fi.extrainfo as instanceinfo 
      from frameinstances fi 
      inner join si_test_slct on si_test_slct.instance_id = fi.id
    )
    select 
      sifi_test.instanceid,
      sifi_test.split,
      sifi_test.truelabel,
      ca_cl.clusterlabel,
      sifi_test.lu_lemma,
      ca_cl.clusterid,
      ca_cl.clusterinfo,
      ca_cl.assignmentinfo,
      sifi_test.instanceinfo
    from sifi_test
    left join ca_cl on ca_cl.instanceid = sifi_test.instanceid;
END
$func$;

-- EXAMPLE:
-- select * 
-- from evaltable(
--   36, 
--   3, 
--   array['test']
-- ) 
-- where (clusterinfo->'isknown')::boolean is true

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- get a full table of true labels and assigned labels of the split_instances for the test instances of a clustering
CREATE OR REPLACE FUNCTION evaltable(q_clusteringid integer)
  RETURNS TABLE (
    instanceid frameinstances.id%TYPE,
    split split_instances.split%TYPE,
    label_true frameinstances.frame_label%TYPE,
    clabel clusters.label%TYPE,
    lu_lemma frameinstances.lu_lemma%TYPE,
    cid clusters.id%TYPE,
    clusterinfo clusters.extrainfo%TYPE,
    assignmentinfo clusterassignments.extrainfo%TYPE,
    instanceinfo frameinstances.extrainfo%TYPE
  )
  LANGUAGE plpgsql AS
$func$
DECLARE
	_testsplits text[];
	_trainsplits text[];
	_ds_split_id integer;
BEGIN
  select cl.datasetsplit_id into _ds_split_id from clusterings cl where cl.id = q_clusteringid;
	RAISE NOTICE 'DatasetSplit id : %', _ds_split_id;
	select translate((setting->'data'->'testsplits')::text, '[]','{}')::text[] into _testsplits from clusterings where id = q_clusteringid;
	RAISE NOTICE 'Testsplits : %', _testsplits;
	with arr as (
		select translate((setting->'data'->'splits')::text, '[]','{}')::text[] as ar from clusterings where id = q_clusteringid
	), items as (
		select unnest(ar) as item from arr
	), itemsremoved as (
		select item from items where not item=any(_testsplits)
	), itemsremovedasarr as (
		select array_agg(item) as newarr from itemsremoved
	)
	select newarr into _trainsplits from itemsremovedasarr;
	RAISE NOTICE 'Trainsplits : %', _trainsplits;
	RETURN QUERY
    select * from evaltable(
      q_clusteringid,
      _ds_split_id,
      _testsplits
    );
END
$func$;

-- EXAMPLE:
-- select * 
-- from evaltable(36) 
-- where (clusterinfo->'isknown')::boolean is true

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- get instances grouped by lemma
CREATE OR REPLACE FUNCTION get_lemma_groups(datasplit text, splits text[], mincount integer)
  RETURNS TABLE (
    lu_lemma frameinstances.lu_lemma%TYPE,
    numel bigint,
    instances integer[]
  )
  LANGUAGE plpgsql AS
$func$
DECLARE
  _ds_split_id integer;
BEGIN
  select d.id into _ds_split_id from datasetsplits d where d.name = datasplit;
  RETURN QUERY
    select sifi.lu_lemma, count(*) as numel, array_agg(sifi.instance_id) as instances 
    from (
      select si.datasetsplit_id, si.instance_id, si.split, fi.lu_lemma, fi.frame_label 
      from (
        select _s_.datasetsplit_id, _s_.instance_id, _s_.split
        from split_instances _s_
        where datasetsplit_id = _ds_split_id
          and _s_.split = ANY(splits)
      ) si
      join frameinstances fi
        on fi.id = si.instance_id
    ) sifi
    group by sifi.lu_lemma
    having count(*) >= mincount
    order by numel desc;
END
$func$;

-- EXAMPLE:
-- select * 
-- from get_lemma_groups(
--   'fn1.7-default', 
--   array['train','test','dev'], 
--   3
-- )

-- ============================================================================
-- ============================================================================
-- ============================================================================

CREATE OR REPLACE FUNCTION get_filtered_lemmas(_datasplit text, _splits text[], _testsplits text[], _mincount integer, _maxcount bigint)
  RETURNS TABLE (
    lu_lemma frameinstances.lu_lemma%TYPE,
    numel_filtered numeric
  )
  LANGUAGE plpgsql AS
$func$
DECLARE
  _ds_split_id integer;
BEGIN
  select d.id into _ds_split_id from datasetsplits d where d.name = _datasplit;
  RETURN QUERY
  select
    grouped_lemmasplits_filtered.lu_lemma,
    sum(numelsplit) as numel_filtered
  from (
    select 
      sifi.lu_lemma, 
      sifi.split,
      count(*) as numelsplit
    from (
      select si.datasetsplit_id, si.instance_id, si.split, fi.lu_lemma, fi.frame_label 
      from (
        select _s_.datasetsplit_id, _s_.instance_id, _s_.split
        from split_instances _s_
        where datasetsplit_id = _ds_split_id
          and _s_.split = ANY(_splits)
      ) si
      join frameinstances fi
        on fi.id = si.instance_id
    ) sifi
    group by sifi.lu_lemma, sifi.split
    having 
      split != any(_testsplits)
      or ( 
        count(*) >= _mincount
        and 
        count(*) <= _maxcount
      )
    order by numelsplit desc
  ) grouped_lemmasplits_filtered
  group by grouped_lemmasplits_filtered.lu_lemma order by numel_filtered desc;
END
$func$;

-- EXAMPLE:
-- select * 
-- from get_filtered_lemmas(
--   'fn1.7-default', 
--   array['train','dev','test'], 
--   array['test'], 
--   5, 
--   7
-- );

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- search for a sentence that is like query_
-- convert tokens array and try to find a match with the like operator
CREATE OR REPLACE FUNCTION find_frameinstance_sentence_match(query_ text)
  RETURNS TABLE (LIKE frameinstances) 
  LANGUAGE plpgsql AS
$func$
BEGIN
  RETURN QUERY
  select *
  from frameinstances 
  where dataset_id = 2 
  	and array_to_string(array(select lower(jsonb_array_elements_text(extrainfo->'tokens'))), ' ', '*') like query_;
END
$func$;

-- EXAMPLE:
-- select * from find_sentence_match('a%') limit 10

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- show the best scores of a specific datasetsplit 
CREATE OR REPLACE FUNCTION show_scores(q_ds_split_name text, q_testsplits text[])
  RETURNS SETOF clusterings_scored
  LANGUAGE plpgsql AS
$func$
BEGIN
	RETURN QUERY
  select * from clusterings_scored
  where datasetsplit = q_ds_split_name
  and testsplits @> q_testsplits
  and testsplits <@ q_testsplits;
END
$func$;

-- EXAMPLE:
-- select * from show_scores('fn1.7-default', array['test'])

-- ============================================================================
-- ============================================================================
-- ============================================================================