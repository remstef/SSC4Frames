-- ============================================================================
-- ============================================================================
-- ============================================================================

-- a view for joined split_instances with full frame instance information
create view frameinstances_split as
select
  f.dataset_id,
  d.name as dataset_name,
  s.datasetsplit_id, 
  ds.name as datasetsplit_name,
  s.split,
  s.instance_id, 
  f.lu_lemma, 
  f.frame_label, 
  f.global_id, 
  f.extrainfo
from split_instances s
join frameinstances f on s.instance_id = f.id
join datasetsplits ds on s.datasetsplit_id = ds.id
join datasets d on f.dataset_id = d.id;

-- EXAMPLE 1:
-- select * from frameinstances_split 
-- where lu_lemma='geben' 
-- and datasetsplit_id = 19
-- order by random()
-- limit 10

-- EXAMPLE 2:
-- select
-- 	dataset_id,
-- 	split,
-- 	count(*) as numinstances 
-- from frameinstances_split  
-- where lu_lemma='geben' 
-- and datasetsplit_id = 19
-- group by (dataset_id, split);

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- select cluster instances as list of ids per cluster for a specific clustering as view
create view clusterinstances as
	with globalassignment_with_localinfo as (
		select 
			cl.clusteringid,
			ca.clusterid,
			ca.instanceid,
			(ca.extrainfo->'local_cluster_id')::integer as localclusterid,
			cl.label as clusterlabel,
			(cl.extrainfo->>'transitive_label') as transitive_label,
			(cl.extrainfo->>'numelems')::int as numel_clustered
		from 
			clusterassignments ca
		join clusters cl on cl.id = ca.clusterid
	)
	select 
		clusteringid,
		clusterid,
		any_value(distinct clusterlabel) as label_unique,
		any_value(distinct transitive_label) as label_transitive,
		any_value(distinct numel_clustered) as numel_clustered,
		count(distinct localclusterid) as numel_localclusters,
		count(distinct instanceid) as numel_instances,
		array_agg(distinct localclusterid) as localclusterids,
		array_agg(distinct instanceid) as instanceids
	from 
		globalassignment_with_localinfo gca
	group by (clusteringid, clusterid);

-- EXAMPLE:
-- select * from clusterinstances where clusteringid = 996;

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- select cluster instances as list of ids per cluster for a specific clustering as view grouped by transitive label
create view transitiveclusterinstances as
	select 
		ci.clusteringid, 
		count(distinct clusterid) as num_unique_clusters,
		ci.label_transitive,
		count(ci.instanceids_unnested)as numel_instances,
		count(ci.localclusterids_unnested)as numel_localclusters,
		array_agg(distinct ci.clusterid) as uniqueclusterids,
		array_agg(distinct ci.label_unique) as uniqueclusterlabels,
		array_agg(distinct ci.localclusterids_unnested)as localclusterids,
		array_agg(ci.instanceids_unnested)as instanceids
	from (
		select *, 
			unnest(instanceids) as instanceids_unnested,
			unnest(localclusterids) as localclusterids_unnested
		from clusterinstances
	) ci
	group by (ci.clusteringid, ci.label_transitive);

-- EXAMPLE 1:
-- select * from transitiveclusterinstances where clusteringid = 996 order by numel desc;

-- EXAMPLE 2:
-- select * from transitiveclusterinstances where clusteringid = 1016 order by num_unique_clusters desc

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- show best scoring as a view of localglobal clusterings for a specific dataset clustered with specific splits of instances
create view clusterings_scored as
select 
	cl.id as cid, 
    (cl.extrainfo->'evalresults'->'novelty_+_frame'->'micro avg'->>'f1-score')::float as micro_f1__novelty_frame,
	coalesce((cl.setting->'global'->'clusterer'->>'type')::text, (cl.setting->'local'->'clusterer'->>'type')::text) as clusterer,
	cl.type as type,
	(lcl.setting->'local'->'clusterer'->>'type') as local_clusterer,
	cl.datasetsplit_id as datasetsplit_id,
	(select ds.name from datasetsplits ds where ds.id = cl.datasetsplit_id) as datasetsplit,
	cl.splits as splits,
	translate((cl.setting->'data'->'testsplits')::text, '[]','{}')::text[] as testsplits,
	cl.numclusters,
	lcl.numclusters as numclusters_local,
	AGE(cl.finish, cl.start) as duration,
	AGE(lcl.finish, lcl.start) as local_duration,
	lcl.id as local_cid,
	cl.extrainfo as extrainfo,
	cl.setting as setting,
	lcl.extrainfo as localextrainfo,
	lcl.setting as localsetting
from clusterings cl
left join clusterings lcl 
	on lcl.id = (cl.extrainfo->>'clusteringid_local')::int 
	and lcl.type='local';

-- EXAMPLES:
-- select * from clusterings_scored where type = 'localglobal' limit 10

-- select * from clusterings_scored order by micro_f1__novelty_frame desc limit 10

-- with args (ds_split_id, testsplits) as (
--   values (
-- 		(select d.id  from datasetsplits d where d.name = 'fn1.7-default'),
-- 		array['test']
-- 	)
-- )
-- select * from clusterings_scored
-- where datasetsplit_id = (select ds_split_id from args)
-- and testsplits @> (select testsplits from args)
-- and testsplits <@ (select testsplits from args)
-- order by micro_f1__novelty_frame desc;

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- join the clusterings_scored views with experiments so filtering by experiment is possible
create view experiment_scores as
select 
  exr.experiment_id,
  ex.name,
  exr.id as experiment_run_id,
  exr.status,
  exr.clustering_id,
  cs.micro_f1__novelty_frame,
  cs.clusterer,
  cs.type,
  cs.local_clusterer,
  cs.datasetsplit,
  cs.splits,
  cs.testsplits,
  cs.numclusters,
  cs.numclusters_local,
  cs.duration as clustering_duration,
  cs.local_duration as clustering_duration_local,
  cs.local_cid,
  cs.extrainfo as clusteringinfo,
  cs.localextrainfo as clusteringinfo_local,
  exr.setting,
  exr.extrainfo as experimentruninfo,
  ex.extrainfo as experimentinfo
from experiments ex
join experiment_runs exr on ex.id=exr.experiment_id
join clusterings_scored cs on exr.clustering_id=cs.cid;

-- EXAMPLES:

-- select * from experiment_scores where experiment_id = 6

-- select * from experiment_scores where experiment_run_id = 775

-- select * 
-- from experiment_scores 
-- where experiment_id = 6
-- order by micro_f1__novelty_frame desc;

-- select 
--   (clusteringinfo->'evalresults'->'frame_induction_alleval'->>'b^3f1')::float as bcubedf1, 
--   * 
-- from experiment_scores 
-- where experiment_id = 6
-- order by bcubedf1 desc;

-- select 
--   (clusteringinfo->'evalresults'->'frame_induction_alleval'->>'b^3f1')::float as bcubedf1, 
--   (clusteringinfo->'evalresults'->'frame_induction_alleval'->>'fmi')::float as fmi, 
--   * 
-- from experiment_scores 
-- where experiment_id = 6
-- order by fmi desc

-- ============================================================================
-- ============================================================================
-- ============================================================================

-- show complete instance information with cluster assignments  
create view instanceassignments as
select
  f.id as instance_id,
  cl.clusteringid,
  ca.clusterid,
  cls.type as clusteringtype,
  cl.label as clusterlabel,
  cl.extrainfo->>'transitive_label' as tclusterlabel,
  f.dataset_id,
  d.name as dataset_name,
  s.datasetsplit_id, 
  ds.name as datasetsplit_name,
  s.split,
  f.lu_lemma, 
  f.frame_label, 
  f.global_id, 
  f.extrainfo as instanceinfo,
  ca.extrainfo as assignmentinfo,
  cl.extrainfo as clusterinfo,
  cls.extrainfo as clusteringinfo
from split_instances s
join frameinstances f on s.instance_id = f.id
join datasetsplits ds on s.datasetsplit_id = ds.id
join datasets d on f.dataset_id = d.id
join clusterassignments ca on ca.instanceid = f.id
join clusters cl on ca.clusterid = cl.id 
join clusterings cls on cl.clusteringid = cls.id
  and cls.datasetsplit_id = s.datasetsplit_id;

-- select * 
-- from instanceassignments 
-- where clusteringid = 1512
-- and (lu_lemma = 'ruin' or frame_label = 'fn1.7::417::Destroying')

-- NOTE: this does not show unassigned instances. For now, you can find unassigned instances by using the evaltable function, e.g. with this workflow:
-- with cl as (
--   select * from clusterings where id = 17472
-- ), unassigned as (
--   -- 643569
--   -- 124 cid == NULL
--   select * from evaltable((select id from cl), (select datasetsplit_id from cl), (select splits from cl)) where cid is NULL
-- )
-- select * from "bert-base-german-cased-masked" v join unassigned u on v.key = u.instanceid
--
-- or
--
-- with cl as (
--   select * from clusterings where id = 237
-- ), allinstances as (
--   select * from frameinstances_split where datasetsplit_id = (select datasetsplit_id from cl) and split = any((select splits::text from cl)::text[])
-- ), cluster_assigned as (
--   select * from instanceassignments where clusteringid = (select id from cl)
-- )
-- select * from allinstances ia left outer join cluster_assigned ica on ia.instance_id = ica.instance_id
--
-- or ignore the splits
--
-- with cl as (
--   select * from clusterings where id = 237
-- ), allinstances as (
--   select * from frameinstances_split where datasetsplit_id = (select datasetsplit_id from cl)
-- ), cluster_assigned as (
--   select * from instanceassignments where clusteringid = (select id from cl)
-- )
-- select * from allinstances ia left outer join cluster_assigned ica on ia.instance_id = ica.instance_id

-- ============================================================================
-- ============================================================================
-- ============================================================================

