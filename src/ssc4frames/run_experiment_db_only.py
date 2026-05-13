#!/usr/bin/env python
# coding: utf-8

import traceback
from datetime import datetime
import os
import re
import subprocess
import time
from copy import deepcopy
from functools import partial
import json
import numpy as np
import torch
import math
import pprint
import sqlalchemy as sa
import pgvector.sqlalchemy as sapgvec
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.base import ClusterMixin
from ssc4frames.database import DBHandler, DatasetSplit, SplitRelation, Clustering, Cluster, ClusterAssignment
from ssc4frames.evaluation import frame_induction, frame_identification, novelty_detection, known_frame_identification, unknown_frame_induction
from ssc4frames.helpers import get_dburl_from_env, get_obj_hash
from ssc4frames.factoryhelper import create_factory
import ssc4frames.loghelper as loghelper
import logging

### example parameters
### will be overriden if the name of a json-file is passed as argument
default_parameters = {
  # the meta tag will not be used to identify clusterings
  'meta': {
    # Use the specified device for tensors, e.g. cpu or gpu.
    # GPU support is limited as clustering algorithms do not necessarily use other devices than CPU.
    'device': 'cpu',
    'reuse': { 
      # Clusterings will be reused if one with the same (sub-)configuration exists. 
      # Set to False if you want to recompute clustering even if it exists!
      'local': False, 
      'global': False,
    },
    're-evaluate': { 
      # Set to True if you want to re-evaluate the clustering even if it exists and has been evaluated, ie. scores that have been stored will be overwritten
      'local': False, 
      'global': False
    },
    'internaleval': {
      # Internal / unsupervised evaluation is performed using the silhouette score
      'silhouette': {
        'nsamples': 1e3, # how many samples to use
        'ndraws': 3, # how many draws (if using subsampling)
        'randomize_order': True, # for multiple draws, this has to be True
        'random_seed': None, # for multiple draws, this has to be None
      }
    },
    'note': None, # can be a string, an object or a list, anything you like. It will be added to extrainfo, so that it can be used for retrieval. I.e. you can specify if the clustering is used for hyperparameter tuning or if its used for testing, or if it can be deleted 
    # 'save_test_only': False, # only checked in local clustering and transitively applied for global clustering
    # 'parallel_local_clusterings': 128 # not used yet 
  },
  # specify the dataset and the splits to use
  'data': {
    'dataset': 'fn1.7-sample', # 'fn1.7-default',
    'splits': [ 'train', 'dev', 'test' ], # specify the splits which are going to be clustered (possibly in a semi-supervised fashion, labels are not necessarily required), labels are explicitly removed from instances which are in the testsplits list below
    'testsplits': [ 'test' ], # specify the datasplit instances which are used for testing, i.e. during (semi-supervised) clustering, labels are removed from those instances
    'materialize': True, # improves runtime performance if set to true, might increase database storage drasticlally if dataset is too large (> 100K)
  },
  # configure local clustering step
  'local': {
    # which model to use
    'emmodel': 'bert-base-uncased', # 'bert-base-uncased' for English, 'bert-base-german-cased' for German, 'nvidia_NV-Embed-v2' for multilingual
    # specify the dimension of the model, it must match!
    'dim': 768, # 768 for bert-..., 4096 for nvidia_NV-Embed-v2
    # specify the weight of masked vs unmasked embeddings
    # unmasked with weight alpha and masked with weight 1 − alpha 
    # (i.e. alpha=0 -> only masked embeddings are used; alpha=1 -> only unmasked embeddings are used)
    'alpha': '0.3',
    # instances which do not match this filter will be disregarded during clustering, 
    # i.e. they will not be clustered directly, but assigned by the most similar (nearest neighbor) 
    # cluster as a post clustering step. This is mainly useful for large datasets.
    'filter': { 
      # consider only lemmas with at least $min_lemmainstances instances in the test split, use 1 to deactivate filtering
      'min_lemmainstances': 1,
      # consider only lemmas with at most of $max_lemmainstances instances in the test split, use a very large value to deactivate filtering
      'max_lemmainstances': 1e10, 
      # limit the number of instances per lemma to $limit_lemmainstances (use only the first instances as defined by instanceid (default) 
      # or random order if randomize_order is true. Only necessary id dataset is very large (>100K). Use a very large value to deactivate filtering
      'limit_lemmainstances': 1e10,
      'randomize_order': True,
      'random_seed': 0.946684799, # default=None
    },
    # define the clusterer for the local clustering step with options (@see ssc4frames.clusterer)
    'clusterer': {
      'type': 'cw', 
      'options': {
        'random_state': 946684799, # default=None
        'criterion': 'minw_0.6', # set the minimum weight an edge in the similarity graph should have 
      }
    },
    # as of now, there is no other option than to average embeddings of a cluster, 
    # i.e. this option is disregarded
    'emaggregation': 'avg' 
  },
  # define the global clustering step
  'global': {
    # define the local clustering, on which the global clusterer should operate
    # use ##local@latest to refer to the local clustering as defined in this file, 
    # ##local@latest will be replaced to identifier@id internally
    # use identifier@id to refer to any local clustering identifier + id that has to exist in the database independent of the local setting in this file
    'localclustering': '##local@latest',
    # cluster instances which do not match this filter will be disregarded during clustering, 
    # i.e. instances within filtered clusters will be assigned by the most similar (nearest neighbor) 
    # cluster as a post clustering step. This is mainly useful for large datasets.
    'filter': {
      # consider only clusters with at least $min_clusterinstances instances, use 1 to deactivate filtering
      'min_clusterinstances': 1, 
      # consider only clusters with at most $max_clusterinstances instances, use a large value to deactivate clustering
      'max_clusterinstances': 1e10, 
      'randomize_order': True,
      'random_seed': 0.946684799, # default=None
    },
    # define the clusterer for the global clustering step with options (@see ssc4frames.clusterer)
    'clusterer': {
      'type': 'cw', 
      'options': {
        'random_state': 946684799, # default=None
        'criterion': 'minw_0.9', # set the minimum weight an edge in the similarity graph should have 
      }
    },
    # define the strategy how clusters with known labels will be merged across the two clustering stages
    # merge_knowns=after_local => creates a new clustering with merged local clusters where local cluster embeddings are based on merged clusters
    # merge_knowns=before_global => use merge label (transitive label) as yinput for global clustering, i.e. the local cluster embeddings are different and have multiple instances, but their (known) label is the same
    # merge_knowns=after_global => no merging while clustering but transtive labels are used to create merged global clusters where global cluster embeddings are based on merged clusters
    # merge_knowns=never => nothing is never ever merged (evaluation is still based on the transitive labels)
    'merge_knowns' : 'before_global' 
  }
}
__default_db_url__ = get_dburl_from_env()
__base_logger_name__ = os.path.basename(__file__)
__base_logger = loghelper.setup_logger(__base_logger_name__)
machinename = os.uname()[1]
try:
  git_revision_short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__))).decode('ascii').strip()
except:
  from importlib.metadata import version
  git_revision_short_hash = f'ssc4frames.v{version("ssc4frames")}'

fmtstr__create_vectorized_split_instances_view = '''
  create {materialized:s} view if not exists frameinstances_split_vectorized__{clusteringid:d} as
  select
    fs.dataset_id,
    fs.dataset_name,
    fs.datasetsplit_id, 
    fs.datasetsplit_name,
    fs.split,
    fs.instance_id, 
    fs.lu_lemma, 
    fs.frame_label, 
    fs.global_id, 
    fs.extrainfo,
    l2_normalize((vu.embedding * sv.w1) + (vm.embedding * sv.w2)) as vector
  from frameinstances_split fs
  join "{emmodel}" vu on vu.key = fs.instance_id
  join "{emmodel}-masked" vm on vm.key = fs.instance_id
  cross join (
    select w1,w2 from (values(array_fill({alphaval}, '{{{dim}}}')::vector,array_fill({minusalphaval}, '{{{dim}}}')::vector)) sv(w1,w2)
  ) sv
  where fs.datasetsplit_name = :datasetsplit
    and fs.split = any(:splits);
  '''
fmtstr__vector_instances_cte_query = '''
  with vectorinstances as (
    select 
      fsv.dataset_id,
      fsv.datasetsplit_id,
      fsv.instance_id,
      fsv.split,
      fsv.lu_lemma,
      fsv.frame_label,
      fsv.global_id,
      fsv.extrainfo,
      fsv.vector
      from {setseed_str} frameinstances_split_vectorized__{clusteringid:d} fsv
      where fsv.lu_lemma = :lemmaquery
    order by fsv.frame_label='<unk>', {order_by}
    limit :limitinstances
  )
'''
fmtstr__vector_clusters_cte_query = '''
  with embeddedclusters as (
    select 
      cl.id as clusterid, 
      cl.clusteringid as clusteringid,
      cl.label as clusterlabelu,
      cl.extrainfo->>'transitive_label' as clusterlabel,
      (cl.extrainfo->'numelems')::integer as numel,
      (cl.extrainfo->'isknown')::boolean as isknown,
      cle.aggregationtype as aggregationtype,
      cle.embedding as clusterembedding
    from {setseed_str} clusters cl 
    inner join clusterembeddings__{clusteringid:d} cle
    on cl.id = cle.clusterid
    where cle.aggregationtype = :aggtype
    and ( 
      (cl.extrainfo->>'isknown')::boolean is true
      or
      (
        (cl.extrainfo->'numelems')::integer >= :minelems
        and 
        (cl.extrainfo->'numelems')::integer <= :maxelems
      )
    )
    order by {order_by}
  )
'''
fmtstr__create_cluster_embedding_table = '''
  CREATE TABLE IF NOT EXISTS clusterembeddings__{clustering_id} (
    clusterid integer NOT NULL,
    embedding vector({vectordim}),
    aggregationtype character varying(32),
    extrainfo jsonb,
    CONSTRAINT clusterembeddings__{clustering_id}_pkey PRIMARY KEY (clusterid),
    CONSTRAINT clusterembeddings__{clustering_id}_clusterid_fkey FOREIGN KEY (clusterid) REFERENCES clusters(id)
  )
'''


class DBLogHandler(logging.Handler):
    '''
    Customized logging handler that puts logs to the database.
    '''
    def __init__(self, dbh:DBHandler, clusteringid):
      logging.Handler.__init__(self)
      self.dbh = dbh
      self.clusteringid = clusteringid

    def add_logentry(self, clusteringid, logmessage):
      # UPDATE table SET array_field = array_append(array_field,'new item') WHERE
      with self.dbh.sessionmaker() as session:
        session.execute(
        sa.update(Clustering)
            .where(Clustering.id==clusteringid)
            # .where(Clustering.status=='running')
            .values(logs=sa.text(f'array_append({Clustering.logs.name}, :newlogentry)')),
          {'newlogentry': logmessage}
        )
      return

    def emit(self, record):
      msg = self.format(record)
      self.add_logentry(self.clusteringid, msg)
  

def get_logger_name_for_clustering(clusteringid:int):
  return f'{__base_logger.name}.clustering{clusteringid}'


def setup_database_handler(dbconnectionstring:str) -> DBHandler:
  dbh:DBHandler = DBHandler(dbconnectionstring)
  return dbh


def setup_logger_for_clustering(dbh:DBHandler, clusteringid:int) -> logging.Logger:
  db_log_handler = DBLogHandler(dbh, clusteringid)
  db_log_handler.setFormatter(__base_logger.handlers[0].formatter)
  # 2 options: 
  # 1) add the handler to the current logger
  # 2) create a new separate logger for the clustering
  # prefer option 2) because of possible parallism issues with option 1)
  # base_logger.addHandler(db_log_handler) # <- this would be option 1)
  # this is option 2) ->:
  cl_logger = logging.getLogger(get_logger_name_for_clustering(clusteringid))
  cl_logger.addHandler(db_log_handler)
  cl_logger.setLevel(__base_logger.getEffectiveLevel())
  return cl_logger


def cleanup_logger_for_clustering(cl_logger:logging.Logger, clusteringid:int):
  assert get_logger_name_for_clustering(clusteringid) == cl_logger.name
  dbloghandlers = filter(lambda h: isinstance(h, DBLogHandler), cl_logger.handlers)
  for dbloghandler in dbloghandlers:
    cl_logger.removeHandler(dbloghandler)
  # not sure if this is necessary, but better be safe than sorry
  del logging.Logger.manager.loggerDict[cl_logger.name]


 # merge recursively
def merge_params(params1:dict, params2:dict, skip_merge_dict_keys=['clusterer']) -> dict:
  # merge
  for key, val in params1.items():
    if type(val) == dict:
      if key in params2 and type(params2[key]) == dict:
        # if the key is 'clusterer' simply use the one from params2, nothing will be merged here
        if key in skip_merge_dict_keys:
          params1[key] = params2[key]
        else:
          merge_params(params1[key], params2[key], skip_merge_dict_keys)
      else:
        if key in params2:
          params1[key] = params2[key]        
    else:
      if key in params2:
        params1[key] = params2[key]
  # add rest
  for key, val in params2.items():
    if not key in params1:
      params1[key] = val
  return params1


def merge_with_default_params(params_override:dict) -> dict:
  # copy, merge and return
  return merge_params(deepcopy(default_parameters), params_override)


def get_retrieve_instance_similarity_matrix_dense(dbh:DBHandler, localclusteringid:int, data:dict, lulemma:str, expect_n:int, alphaval:float, emmodel:str, vectordim:int, filter:dict, device:torch.DeviceObjType|str, cl_logger:logging.Logger):
  order_by = 'random()' if filter.get('randomize_order', False) else 'instance_id'
  rseed = filter.get('random_seed', None)
  setseed_str = f'setseed({rseed}),' if rseed is not None and filter.get('randomize_order', False) else ''
  stmt = sa.text(fmtstr__vector_instances_cte_query.format(clusteringid=localclusteringid, setseed_str=setseed_str, order_by=order_by) + '''
    select 
      a.instance_id as id_a, 
      b.instance_id as id_b,
      a.split as split_a,
      b.split as split_b,
      b.frame_label as label_b,
      1 - (a.vector <=> b.vector) as cosim,
      b.extrainfo->'tokens' as tokens_b
    from vectorinstances a
    cross join vectorinstances b 
    order by a.instance_id, a.split, b.instance_id, b.split;
  ''')
  with dbh.sessionmaker() as session:
    # get the adjacency matrix of wheighted instances and their labels
    res = session.execute(stmt, {
      'lemmaquery': lulemma,
      'limitinstances': filter.get('limit_lemmainstances', 1e10)
    })
    rows_t = dict(zip(res.keys(), zip(*res)))
  if len(rows_t) <= 0:
    cl_logger.error(f'Retrieved 0 similarity values for {expect_n} instances.')
    # TODO: return empty lists if resultset is empty, for now, throw an error
    raise ValueError('Nothing to cluster, similarity matrix is empty.')
  # prepare similarity matrix
  cosims = torch.tensor(rows_t['cosim'], device=device, dtype=torch.float32)
  # since we have a square matrix the number of elements should be the squareroot of the result length % TODO: this could be optimzed to return results already filered by cosim
  num_elems = int(math.sqrt(cosims.size(0)))
  if expect_n != num_elems:
    cl_logger.warning(f'Expected {expect_n} instances for "{lulemma}" in {data["dataset"]} ({data["splits"]}), but got {num_elems}. Please check that frame instances are uniquely distributed across train/test splits and {emmodel} embeddings exist for all instances. Ignore this warning if you used a filter')
  # reshape matrix
  A_L = cosims.view(num_elems, -1)
  # get all other information from rows
  ids = rows_t['id_b'][:num_elems]
  src_split = rows_t['split_b'][:num_elems]
  y_true = rows_t['label_b'][:num_elems]
  # RETURN
  return A_L, ids, src_split, y_true


def get_retrieve_instance_feature_matrix(dbh:DBHandler, localclusteringid:int, data:dict, lulemma:str, expect_n:int, alphaval:float, emmodel:str, vectordim:int, filter:dict, device:torch.DeviceObjType|str, cl_logger:logging.Logger):
  order_by = 'random()' if filter.get('randomize_order', False) else 'instance_id'
  rseed = filter.get('random_seed', None)
  setseed_str = f'setseed({rseed}),' if rseed is not None and filter.get('randomize_order', False) else ''
  stmt = sa.text(fmtstr__vector_instances_cte_query.format(clusteringid=localclusteringid, setseed_str=setseed_str, order_by=order_by) + '''
    select 
      vi.instance_id as id, 
      vi.split as split,
      vi.frame_label as label,
      vi.vector as vec,
      vi.extrainfo->'tokens' as tokens
    from vectorinstances vi;
  ''').columns(vec=sapgvec.Vector)
  with dbh.sessionmaker() as session:
    # get the adjacency matrix of wheighted instances and their labels
    res = session.execute(stmt, {
      'lemmaquery': lulemma,
      'limitinstances': filter.get('limit_lemmainstances', 1e10)
    })
    rows_t = dict(zip(res.keys(), zip(*res)))
  # prepare feature matrix
  M = np.array(rows_t['vec'])
  M = torch.tensor(M, device=device, dtype=torch.float32)
  num_elems = M.size(0)
  if expect_n != num_elems:
    cl_logger.warning(f'Expected {expect_n} instances for "{lulemma}" in {data["dataset"]} ({data["splits"]}), but got {num_elems}. Please check that frame instances are uniquely distributed across train/test splits and {emmodel} embeddings exist for all instances. Ignore this warning if you used a filter')
  # get all other information from rows
  ids = rows_t['id']
  src_split = rows_t['split']
  y_true = rows_t['label']
  # RETURN
  return M, ids, src_split, y_true


def get_retrieve_instances_plain(dbh:DBHandler, localclusteringid:int, data:dict, lulemma:str, expect_n:int, alphaval:float, emmodel:str, vectordim:int, filter:dict, device:torch.DeviceObjType|str, cl_logger:logging.Logger):
  order_by = 'random()' if filter.get('randomize_order', False) else 'instance_id'
  rseed = filter.get('random_seed', None)
  setseed_str = f'setseed({rseed}),' if rseed is not None and filter.get('randomize_order', False) else ''
  stmt = sa.text(fmtstr__vector_instances_cte_query.format(clusteringid=localclusteringid, setseed_str=setseed_str, order_by=order_by) + '''
    select * from vectorinstances;
  ''').columns(vec=sapgvec.Vector)
  with dbh.sessionmaker() as session:
    res = session.execute(stmt, {
      'lemmaquery': lulemma,
      'limitinstances': filter.get('limit_lemmainstances', 1e10)
    })
    df = pd.DataFrame.from_records(res.fetchall(), columns=res.keys())
  num_elems = df.shape[0]
  if expect_n != num_elems:
    cl_logger.warning(f'Expected {expect_n} instances for "{lulemma}" in {data["dataset"]} ({data["splits"]}), but got {num_elems}. Please check that frame instances are uniquely distributed across train/test splits and {emmodel} embeddings exist for all instances. Ignore this warning if you used a filter')
  return df, df.instance_id.tolist(), df.split.tolist(), df.frame_label.tolist()


def get_retrieve_labels_from_clusterassignments_for_local_clustering(dbh:DBHandler, data:dict, clustering_obj:Clustering, source_clustering_config:dict, lulemma:str, expect_n:int, filter:dict, device:torch.DeviceObjType|str, cl_logger:logging.Logger):

  ## check that the given clustering fits the settings
  if source_clustering_config is not None:

      localidentifier_settings = {'data': data, 'local': {**source_clustering_config, **{'filter': filter}}}
      localidentifier = Clustering.get_identifier_from_settings(localidentifier_settings)
      assert clustering_obj.identifier == localidentifier

  else:
      assert get_obj_hash(clustering_obj.setting['data']) == get_obj_hash(data), f'Expected data dict for clustering {clustering_obj.identifier}: {pprint.pformat(clustering_obj.setting['data'])} \n but got {pprint.pformat(data)}.'
      assert get_obj_hash(clustering_obj.setting['local'].get('filter', {})) == get_obj_hash(filter), f'Expected filter dict for clustering {clustering_obj.identifier}: {pprint.pformat(clustering_obj.setting['local'].get('filter'))} \n but got {pprint.pformat(filter)}.'
  order_by = 'random()' if filter.get('randomize_order', False) else 'instance_id'
  rseed = filter.get('random_seed', None)
  setseed_str = f'setseed({rseed}),' if rseed is not None and filter.get('randomize_order', False) else ''
  stmt = sa.text(f'''
    with instances as (
      select 
        dataset_id, 
        datasetsplit_id, 
        instance_id, 
        split, 
        lu_lemma, 
        frame_label, 
        global_id, 
        extrainfo
      from {setseed_str} frameinstances_split 
      where 
        datasetsplit_name = :datasetsplit
        and split = ANY(:splits)
        and lu_lemma = :lemmaquery
      order by frame_label='<unk>', {order_by}
      limit :limitinstances
    ), clusters_slct as (
      select * 
      from clusters 
      where clusteringid = :clusteringid
    ), assignments as (
      select
        ca.instanceid as instanceid,
        ca.clusterid as cid,
        cl.label as uclabel,
        cl.extrainfo->>'transitive_label' as clabel,
        (cl.extrainfo->>'isknown')::bool as isknown
      from clusterassignments ca 
      join clusters_slct cl on cl.id = ca.clusterid
    )
    select 
      i.instance_id as id,
      a.cid,
      a.uclabel,
      a.clabel,
      a.isknown,
      i.split as split,
      i.frame_label as true_label
    from instances i
    join assignments a on i.instance_id = a.instanceid
  ''')
  with dbh.sessionmaker() as session:
    # get the adjacency matrix of wheighted instances and their labels
    res = session.execute(stmt, { 
      'clusteringid':clustering_obj.id, 
      'datasetsplit':data['dataset'],
      'splits': data['splits'],  #array['train','test','dev']
      'lemmaquery': lulemma,
      'limitinstances': filter.get('limit_lemmainstances', 1e10)
    } )
    rows_t = dict(zip(res.keys(), zip(*res)))
  # prepare feature matrix
  clabels = rows_t['clabel']
  num_elems = len(clabels)
  if expect_n != num_elems:
    cl_logger.warning(f'Expected {expect_n} instances for "{lulemma}" in clusterassignments {clustering_obj.id}, but got {num_elems}.')
  # get all other information from rows
  ids = rows_t['id']
  src_split = rows_t['split']
  y_true = rows_t['true_label']
  # RETURN
  return clabels, ids, src_split, y_true


def get_retrieve_cluster_similarity_matrix_dense(dbh:DBHandler, currentclusteringid:int, embeddingclusteringid:int, emaggregation:str, expect_n:int, filter:dict, device:torch.DeviceObjType|str, cl_logger:logging.Logger):
  order_by = 'random()' if filter.get('randomize_order', False) else 'clusterid'
  rseed = filter.get('random_seed', None)
  setseed_str = f'setseed({rseed}),' if rseed is not None and filter.get('randomize_order', False) else ''
  stmt = sa.text(fmtstr__vector_clusters_cte_query.format(clusteringid=embeddingclusteringid, setseed_str=setseed_str, order_by=order_by) + '''
    select 
      a.clusterid as id_a, 
      b.clusterid as id_b,
      b.clusterlabel as label_b,
      b.clusterlabelu as labelu_b,
      b.numel as numel_b,
      b.isknown as hasknownlabel_b,
      1 - (a.clusterembedding <=> b.clusterembedding) as cosim
    from embeddedclusters a
    cross join embeddedclusters b 
    order by id_a, id_b;
  ''')
  cl_logger.info(f'Retrieving {expect_n}x{expect_n} similarity matrix for {expect_n} local cluster embeddings ({expect_n*expect_n} values)...')
  # get local cluster embeddings
  with dbh.sessionmaker() as session:
    res = session.execute(stmt, {
      'aggtype': emaggregation,
      'minelems': filter.get('min_clusterinstances', 1),
      'maxelems': filter.get('max_clusterinstances', 1e10)
    })
    rows_t = dict(zip(res.keys(), zip(*res)))
  if len(rows_t) <= 0:
    cl_logger.error(f'Retrieved 0 similarity values for {expect_n} local cluster embeddings.')
    # TODO: return empty lists if resultset is empty, for now, throw an error
    raise ValueError('Nothing to cluster, similarity matrix is empty.')
  cl_logger.info(f'Retrieved {len(rows_t['cosim'])} similarity values for {expect_n} local cluster embeddings. Reshaping...')
  # prepare similarity matrix
  cosims = torch.tensor(rows_t['cosim'], device=device, dtype=torch.float32)
  # since we have a square matrix the number of elements should be the squareroot of the result length % TODO: this could be optimzed to return results already filered by cosim
  num_elems = int(math.sqrt(cosims.size(0)))
  if expect_n != num_elems:
    cl_logger.warning(f'Expected {expect_n} local cluster instances (clustering id {embeddingclusteringid}), but got {num_elems}. Ignore this warning if you used a filter')
  # reshape matrix
  A_G = cosims.view(num_elems, -1)
  # get all other information from rows
  ids = rows_t['id_b'][:num_elems]
  labels = rows_t['label_b'][:num_elems]
  ulabels = rows_t['labelu_b'][:num_elems]
  label_is_known = rows_t['hasknownlabel_b'][:num_elems]
  # RETURN
  return A_G, ids, labels, ulabels, label_is_known


def get_retrieve_cluster_feature_matrix(dbh:DBHandler, currentclusteringid:int, embeddingclusteringid:int, emaggregation:str, expect_n:int, filter:dict, device:torch.DeviceObjType|str, cl_logger:logging.Logger):
  order_by = 'random()' if filter.get('randomize_order', False) else 'clusterid'
  rseed = filter.get('random_seed', None)
  setseed_str = f'setseed({rseed}),' if rseed is not None and filter.get('randomize_order', False) else ''
  stmt = sa.text(fmtstr__vector_clusters_cte_query.format(clusteringid=embeddingclusteringid, setseed_str=setseed_str, order_by=order_by) + '''
    select 
      emc.clusterid as id, 
      emc.clusterlabel as label,
      emc.clusterlabelu as ulabel,
      emc.numel as numel,
      emc.isknown as hasknownlabel,
      emc.clusterembedding as vec
    from embeddedclusters emc;
  ''').columns(vec=sapgvec.Vector)
  cl_logger.info(f'Retrieving {expect_n}xD feature matrix for {expect_n} local cluster embeddings...')
  # get local cluster embeddings
  with dbh.sessionmaker() as session:
    res = session.execute(stmt, {
      'aggtype': emaggregation,
      'minelems': filter.get('min_clusterinstances', 1),
      'maxelems': filter.get('max_clusterinstances', 1e10)
    })
    rows_t = dict(zip(res.keys(), zip(*res)))
  if len(rows_t) <= 0:
    cl_logger.error(f'Retrieved 0 clusterembeddings for {expect_n} local cluster embeddings.')
    # TODO: return empty lists if resultset is empty, for now, throw an error
    raise ValueError('Nothing to cluster, feature matrix is empty.')
  cl_logger.info(f'Retrieved {len(rows_t['id'])} clusterembeddings for {expect_n} local cluster embeddings.')
  # prepare feature matrix
  M = np.array(rows_t['vec'])
  M = torch.tensor(M, device=device, dtype=torch.float32)
  num_elems = M.size(0)
  if expect_n != num_elems:
    cl_logger.warning(f'Expected {expect_n} local cluster instances (clustering id {embeddingclusteringid}), but got {num_elems}. Ignore this warning if you used a filter')
  # get all other information from rows
  ids = rows_t['id'][:num_elems]
  labels = rows_t['label'][:num_elems]
  ulabels = rows_t['ulabel'][:num_elems]
  label_is_known = rows_t['hasknownlabel'][:num_elems]
  # RETURN
  return M, ids, labels, ulabels, label_is_known


def add_aggregated_weighted_local_cluster_embeddings(dbh:DBHandler, src_local_clusteringid:int, local_clusteringid:int, emmodel:str, alpha_val:float, vectordim:int, emaggregation:str):
  # make sure that an embedding table exists for the current clustering
  with dbh.sessionmaker() as session: # create table
    res = session.execute(sa.text(fmtstr__create_cluster_embedding_table.format(clustering_id=local_clusteringid, vectordim=vectordim)))

  stmt = sa.text(f'''
    DO
    $$
    DECLARE _cntr integer := 0;
    DECLARE _records CURSOR FOR
      select * 
        from clusterinstances 
        where clusteringid = :clusteringid 
        order by numel_instances desc;
    BEGIN
      _cntr := 0;
      FOR _record IN _records LOOP
        _cntr := _cntr+1;
        -- RAISE NOTICE 'Adding cluster % with id % having instances: %', _cntr, _record.clusterid, _record.instanceids;
        with avg as (
          select 
            count(*) as numel, 
            array_agg(fsv.instance_id) as instances, 
            avg(fsv.vector) as emaveraged
          from frameinstances_split_vectorized__{src_local_clusteringid:d} fsv
          where fsv.instance_id = ANY(_record.instanceids)
        ), gen_embeddings as (
          select
            avg.emaveraged as emaveraged,
            format('{{ "numaveraged": "%s/%s" }}', avg.numel, array_length(_record.instanceids, 1))::jsonb as info
          from avg
        ), gen_embeddings_norm as (
          select l2_normalize(g.emaveraged) as normalizedemavg, g.info 
            from gen_embeddings g
        )
        insert into clusterembeddings__{local_clusteringid:d} (clusterid, embedding, aggregationtype, extrainfo)
          select _record.clusterid, gn.normalizedemavg, :agg, gn.info from gen_embeddings_norm gn;
      END LOOP;
      RAISE NOTICE 'Added % clusters embeddings', _cntr;
    END
    $$;
  ''')
  with dbh.sessionmaker.begin() as session:
    res = session.execute(stmt, {
      'clusteringid': local_clusteringid,
      'agg': emaggregation
    })
  return


def add_aggregated_global_cluster_embeddings(dbh:DBHandler, global_clustering_id:int, local_clustering_id:int, emaggregation:str, vectordim:int):
  # make sure that an embedding table exists for the current clustering
  # CurrentClusterEmbedding = create_new_clusterembedding_table_class(clustering_id=global_clustering_id, vectordimension=vectordim)
  # # add dynamic CurrentClusterEmbedding table by re-initiating prepare_db
  # dbh.prepare_db()
  with dbh.sessionmaker() as session:
    stmt = sa.text(fmtstr__create_cluster_embedding_table.format(clustering_id=global_clustering_id, vectordim=vectordim))
    res = session.execute(stmt)

  stmt = sa.text(f'''
    DO 
    $$
    DECLARE _cntr integer := 0;
    DECLARE _records CURSOR FOR
      select * 
        from clusterinstances 
        where clusteringid = :globalclusteringid 
        order by numel_localclusters desc;
    BEGIN
      _cntr := 0;
      FOR _record IN _records LOOP
        _cntr := _cntr+1;
        -- RAISE NOTICE 'Adding cluster % with id % having local clusters: %', _cntr, _record.clusterid, _record.localclusterids;
        with emavg as (
          select 
            count(*) as numel, 
            array_agg(cv.clusterid) as localclusterids, 
            avg(cv.embedding) as emaveraged
          from clusterembeddings__{local_clustering_id} cv
          where cv.clusterid = ANY(_record.localclusterids)
        ), emavg_norm as (
          select 
            l2_normalize(emavg.emaveraged) as emavgnorm, 
            format('{{ "numaveraged": "%s/%s" }}', emavg.numel, array_length(_record.localclusterids, 1))::jsonb as info
          from emavg
        )
        insert into clusterembeddings__{global_clustering_id} (clusterid, embedding, aggregationtype, extrainfo)
          select _record.clusterid, emavg_norm.emavgnorm, :agg, emavg_norm.info from emavg_norm;
      END LOOP;
      RAISE NOTICE 'Added % cluster embeddings', _cntr;
    END
    $$;
  ''')
  with dbh.sessionmaker.begin() as session:
    res = session.execute(stmt, {
      'globalclusteringid': global_clustering_id,
      'agg': emaggregation
    })
  return


def create_new_clustering_obj(dbh:DBHandler, settings:dict, clusteringtype:str, identifier: str, note:str|list|dict) -> Clustering:
  with dbh.sessionmaker() as session:
    datasetsplit_obj: DatasetSplit = session.execute(sa.select(DatasetSplit).where(DatasetSplit.name == settings['data']['dataset'])).scalar()
    numinstances: int = session.execute(sa.select(sa.func.count()).select_from(SplitRelation).where(SplitRelation.datasetsplit_id == datasetsplit_obj.id).where(SplitRelation.split.in_(settings['data']['splits']))).scalar()
    # numinstances: int = session.execute(sa.select(sa.func.count(SplitRelation.instance_id)).where(SplitRelation.datasetsplit_id == datasetsplit_obj.id).where(SplitRelation.split.in_(splits))).scalar()
    clustering_obj = Clustering(
      identifier = identifier,
      datasetsplit_id = datasetsplit_obj.id,
      splits = settings['data']['splits'],
      numinstances = numinstances,
      type = clusteringtype,
      start=datetime.now(),
      status=f'initialized',
      setting = settings,
      extrainfo = { 'commit': git_revision_short_hash, 'host': machinename, 'note': note }
    )
    session.add(clustering_obj)
    # flush to fill in id
    session.flush()
    # refresh to also load relationships
    clustering_obj = session.get(Clustering, clustering_obj.id, options=(sa.orm.undefer(Clustering.extrainfo),), populate_existing=True)
  return clustering_obj


def get_or_create_db_clustering(dbh: DBHandler, clusteringtype:str, identifier: str, settings: dict, get_if_exist:bool, note:str|list|dict, logger=None, lock=None, tries=10, wait_for_seconds=300):
  '''
  prepare a new clustering
  '''
  # overwrite defaults from .env
  tries = int(os.getenv('concurrentclusterings.wait.numretry', f'{tries}'))
  wait_for_seconds = int(os.getenv('concurrentclusterings.wait.numseconds', f'{wait_for_seconds}'))
  # wait for running clusterings to finish first before creating a new one
  total_tries = tries
  while tries > 0:
    ## make sure that only one process acquires a clustering at a time
    if lock is not None:
      lock.acquire()
    try:
      # create a new clustering if that is desired from the start
      if not get_if_exist:
        clustering_obj = create_new_clustering_obj(dbh, settings, clusteringtype, identifier, note)
        if logger is not None:
            logger.info(f'created new clustering object: \n{clustering_obj}')
        return clustering_obj

      # otherwise check if it exists and take the last successful one, or create one if none exists
      with dbh.sessionmaker() as session:
        clustering_objects = session.execute(
          sa.select(Clustering)
          .where(Clustering.identifier == identifier)
          .where(Clustering.type == clusteringtype)
          .order_by(Clustering.finish.desc())
          .options(sa.orm.undefer(Clustering.extrainfo))
        ).scalars().all()
      # search for successfully finished clusterings
      for clustering_obj in clustering_objects:
        if clustering_obj.success:
          # finished clustering exists - use this
          if logger is not None:
              logger.info(f'Reusing clustering: \n{clustering_obj}')
          return clustering_obj
      # else no succsessfully finished clustering exists, check if one is running
      clustering_objects_running = [c.id for c in clustering_objects if c.status.lower() == 'running' or c.status.lower() == 'initialized']
      if len(clustering_objects_running) > 0:
        if logger is not None:
            logger.info(f'Clustering(s) {clustering_objects_running} is/are currently runnig. Waiting for result.')
      else:
        # no successfully finished clustering or running clustering exists
        # no clustering exists
        clustering_obj = create_new_clustering_obj(dbh, settings, clusteringtype, identifier, note)
        if logger is not None:
            logger.info(f'created new clustering object: \n{clustering_obj}')
        return clustering_obj
    finally:
      if lock is not None:
        lock.release()

    ## clustering exists but is not finished
    if logger is not None:
        logger.info(f'Waiting for clustering {clustering_obj.id} to finish.')
    ## wait before retrying
    time.sleep(wait_for_seconds)
    tries -= 1

  ## was not able to acquire a finished clustering in given number of tries
  raise TimeoutError(f'running clustering did not finish in {total_tries}*{wait_for_seconds}')


def get_clustering_obj_from_identifier_string(dbh:DBHandler, clustering_ident_string:str, require_merged_clustering:bool=False, internal_eval_opts:dict={}, logger:logging.Logger=None):
  clustering_identifier, clustering_id = clustering_ident_string.split('@')
  with dbh.sessionmaker() as session:
    clustering_obj = session.get(Clustering, clustering_id, options=(sa.orm.undefer(Clustering.extrainfo),))
  if clustering_obj == None:
    raise KeyError(f'Local clustering {clustering_ident_string} does not exist!')
  if require_merged_clustering and not 'merged' in clustering_obj.identifier:
    clustering_obj = get_merged_local_clustering(dbh, clustering_obj, clustering_obj.setting['local'], internal_eval_opts, clustering_obj.extrainfo['note'], logger)
  elif not require_merged_clustering and 'merged' in clustering_obj.identifier:
    # don't stay agnostic of require_merged_clustering=False but the clustering_identifier contains ':merged', i.e. the requested clustering is merged although it is not required?
    match = re.compile(r'^(?P<srcident>[a-zA-Z0-9]+)\[(?P<srcid>[0-9]+)\]:merged$')\
      .search(clustering_identifier) # fefb3ed011e1595c1146852ab0ff5d04[1693]:merged
    if match is None:
      raise KeyError(f"Malformed merged local clustering identifier. Expected srcident[srcid]:merged but got '{clustering_identifier}.'")
    src_clustering_identifier = match.group('srcident')
    src_clustering_id = match.group('srcid')
    with dbh.sessionmaker() as session:
      clustering_obj = session.get(Clustering, src_clustering_id, options=(sa.orm.undefer(Clustering.extrainfo),))
    clustering_identifier = src_clustering_identifier
  if clustering_obj.identifier != clustering_identifier:
    raise KeyError(f'Local clustering id ({clustering_obj.id}) does not match identifier ({clustering_obj.identifier}), expected: {clustering_ident_string}!')
  return clustering_obj


def update_status(dbh:DBHandler, clusteringid:int, message:str):
  with dbh.sessionmaker() as session:
    session.execute(
      sa.update(Clustering)
        .where(Clustering.id == clusteringid)
        .values(status=message)
    )


def update_clustering_obj_values(dbh:DBHandler, clusteringid:int, **values):
  with dbh.sessionmaker() as session:
    session.execute(
      sa.update(Clustering)
        .where(Clustering.id == clusteringid)
        .values(**values)
    )
    res = session.get(Clustering, clusteringid, options=(sa.orm.undefer(Clustering.extrainfo),))
  return res


def create_instances_view_if_not_exists(dbh:DBHandler, data:dict, clusteringid:int, emmodel:str, vectordim:int,  alphaval:float):
  # create a view with the vectorized instances for the current clustering
  stmt_create_view = sa.text(fmtstr__create_vectorized_split_instances_view.format(
    materialized='materialized' if data.get('materialize', True) else '',
    clusteringid=clusteringid,
    alphaval=alphaval,
    minusalphaval=1-alphaval,
    emmodel=emmodel,
    dim=vectordim,
  ))
  with dbh.sessionmaker() as session:
    res = session.execute(stmt_create_view, {
      'datasetsplit':data['dataset'],
      'splits': data['splits']
    })


def run_local_clustering_experiment(dbh:DBHandler, factory, data:dict, clusteringid:int, emmodel:str, vectordim:int,  alphaval:float, emaggregation:str, filter_:dict, clusterer:ClusterMixin, device:torch.DeviceObjType|str, save_test_only:bool, cl_logger:logging.Logger, lock=None, meta_note=None):
  update_status(dbh, clusteringid, 'running')
  create_instances_view_if_not_exists(dbh, data, clusteringid, emmodel, vectordim,  alphaval)
  stmt_retrieve_lemmas = sa.text('''
    select lu_lemma, numel_filtered
    from get_filtered_lemmas(
      :datasetsplit, 
      :splits,
      :testsplits,
      :mincount, 
      :maxcount
    )
  ''')
  # get lemmas ordered by count
  with dbh.sessionmaker() as session:
    res = session.execute(stmt_retrieve_lemmas, {
      'datasetsplit': data['dataset'],
      'splits': data['splits'],
      'testsplits': data['testsplits'],
      'mincount': int(filter_.get('min_lemmainstances', 1)),
      'maxcount': int(filter_.get('max_lemmainstances', 1e10))
    })
    lemma_rows = res.all()
  cl_logger.info(f'found {len(lemma_rows)} lemmas.')
  # HELPER for individual lemmas; TODO parallelize
  def cluster_lemma(lulemma:str, expect_n:int, i:int, src_clustering_obj:Clustering):
    # helper method for internal clustering evaluation
    # will be overwritten depending on the clusterer's requirements
    def silhouette_samples_fun(*args, **kwargs): pass
    # retrieve feature matrix depending on the needs of the clusterer
    if clusterer.requires == 'similarity':
      A_L, ids, src_split, y_true = get_retrieve_instance_similarity_matrix_dense(dbh, clusteringid, data, lulemma, expect_n, alphaval, emmodel, vectordim, filter_, device, cl_logger)
      silhouette_samples_fun = partial(silhouette_samples, X=1-A_L, metric='precomputed')
    elif clusterer.requires == 'feature':
      A_L, ids, src_split, y_true = get_retrieve_instance_feature_matrix(dbh, clusteringid, data, lulemma, expect_n, alphaval, emmodel, vectordim, filter_, device, cl_logger)
      silhouette_samples_fun = partial(silhouette_samples, X=A_L, metric='cosine')
    elif clusterer.requires == 'instances':
      # clusterer requires only the list of instances
      A_L, ids, src_split, y_true = get_retrieve_instances_plain(dbh, clusteringid, data, lulemma, expect_n, alphaval, emmodel, vectordim, filter_, device, cl_logger)
    elif clusterer.requires == 'clustering':
      # A_L is the list of cluster labels!!!
      A_L, ids, src_split, y_true = get_retrieve_labels_from_clusterassignments_for_local_clustering(dbh, data, src_clustering_obj, clusterer.source_config, lulemma, expect_n, filter_, device, cl_logger)
    else:
      raise KeyError(f'Expected clusterer to expect either "feature", "similarity", "instances", or "clustering" but got "{clusterer.requires}".')
    # prepare fixed_labels input to CW
    y_input = [ None if split in data['testsplits'] or label=='<unk>' else label for split, label in zip(src_split, y_true) ]
    y_input_labels = set(y_input) - set([None]) # i.e. the fixed_labels explicitly removing the None object
    # prepare a local dataframe for ease of access
    local_df = pd.DataFrame(list(zip(ids, src_split, y_input, y_true)), columns=['id', 'split', 'yinput', 'ytrue'])
    # RUN CLUSTERING
    label_to_ids, list_of_labels = clusterer.fit_predict(A_L, y_input)
    local_df['ypred'] = list_of_labels
    silhouette_sample_scores = np.zeros((len(list_of_labels), )) if not 1 < len(label_to_ids) < len(list_of_labels) else silhouette_samples_fun(labels=list_of_labels)
    silhouette_sample_scores = np.zeros((len(list_of_labels), )) if silhouette_sample_scores is None else silhouette_sample_scores ## happens when silhouette_samples_fun is not overwritten
    local_df['silhouette_by_lemma'] = silhouette_sample_scores.astype(float)
    # Prepare records to save in a database (will be aggregated and saved once all local clusterings are finished)
    lemma_cluster_assignments_as_dicts = [ ]
    for cid, (clabel, citems) in enumerate(label_to_ids.items()):
      ids = local_df.loc[citems].id.tolist()
      cname = clabel if clabel in y_input_labels else f'{lulemma}::{clabel}'
      ucname = f'_v_{lulemma}__{cname}'
      # save clusters (ie. label, isknown, and number of elements) to get a unique id
      with dbh.sessionmaker.begin() as session:
        local_cluster: Cluster = Cluster(
          clusteringid = clusteringid,
          label = ucname,
          extrainfo = { 
            'transitive_label': cname, 
            'lu_lemma': lulemma,
            'isknown': cname in y_input_labels,
            'numelems': len(citems)
          }
        )
        session.add(local_cluster)
        session.flush() # get id of new cluster
        local_cluster_id = local_cluster.id
      # collect cluster assignments
      lemma_cluster_assignments_as_dicts += [ {
          'instanceid': int(dfrow.id),
          'clusterid': local_cluster_id,
          'extrainfo': {
            'fixed_label_input': dfrow.yinput,
            'assigned_by': 'localstep',
            'silhouette_by_lemma': dfrow.silhouette_by_lemma
          }
        } for idx, dfrow in local_df.loc[citems].iterrows() ]
    return len(label_to_ids), lemma_cluster_assignments_as_dicts

  # get src_clustering for clusterer with requirement clustering
  src_clustering_obj = None
  if clusterer.requires == 'clustering':
      if clusterer.source is not None:
        src_clustering_obj = get_clustering_obj_from_identifier_string(dbh, clusterer.source)
      else:
        src_clustering_obj = run_or_retrieve_local_clustering(dbh, factory, data, {**clusterer.source_config, **{'filter': filter_}}, device, save_test_only, True, meta_note, cl_logger, lock)
      ## write src id into extrainfo->"src_local_clustering"
      with dbh.sessionmaker() as session:
          update_clustering_obj_values(dbh, clusteringid,
             extrainfo=sa.func.jsonb_set(
                 Clustering.extrainfo,
                 '{src_local_clustering}',
                 f'{src_clustering_obj.id}'
             )
          )
  # ITERATE OVER ALL LEMMAS
  n_clusters = 0
  all_cluster_assignments_as_dicts = [ ]
  for i, lemma_row in enumerate(lemma_rows):
    cl_logger.info(f'...using {clusterer.name} for local clustering {i+1}/{len(lemma_rows)}: {lemma_row.numel_filtered} instances for "{lemma_row.lu_lemma}".')
    n_clusters_i, cluster_assignments_as_dicts_i = cluster_lemma(lulemma=lemma_row.lu_lemma, expect_n=lemma_row.numel_filtered, i=i, src_clustering_obj=src_clustering_obj)
    n_clusters += n_clusters_i
    all_cluster_assignments_as_dicts += cluster_assignments_as_dicts_i
  # save clusterassignments in a bulk
  cl_logger.info(f'Saving local clusterassignments...')
  with dbh.sessionmaker.begin() as session:
    session.execute(sa.insert(ClusterAssignment), all_cluster_assignments_as_dicts)
  cl_logger.info(f'Start generating and adding aggregated cluster embeddings...')
  add_aggregated_weighted_local_cluster_embeddings(dbh, clusteringid, clusteringid, emmodel, alphaval, vectordim, emaggregation)
  cl_logger.info(f'Finshed generating and adding aggregated cluster embeddings.')
  mean_silhouette_score_by_lemma = np.mean(list(map(lambda ca: ca['extrainfo']['silhouette_by_lemma'], all_cluster_assignments_as_dicts)))
  mean_silhouette_score_by_lemma_unlabelled_samples_only = np.mean(list(map(lambda ca: ca['extrainfo']['silhouette_by_lemma'], filter(lambda ca: ca['extrainfo']['fixed_label_input'] is None,all_cluster_assignments_as_dicts))))
  # UPDATE clustering object in DB
  with dbh.sessionmaker() as session:
    update_clustering_obj_values(dbh, clusteringid,
      status='finished',
      success=True,
      finish=datetime.now(),
      numclusters=n_clusters,
      extrainfo=sa.func.jsonb_set( # chain func.jsonb_set to set multiple values in one go
        sa.func.jsonb_set(
          Clustering.extrainfo, 
          '{mean_silhouette_score_by_lemma}', 
          f'{mean_silhouette_score_by_lemma}'), 
        '{mean_silhouette_score_by_lemma_unlabelled_samples_only}', 
        f'{mean_silhouette_score_by_lemma_unlabelled_samples_only}'
      )                          
    )
  # RETURN
  return n_clusters


def get_merged_local_clustering(dbh:DBHandler, src_local_clusteringobj:Clustering, localparams:dict, internal_eval_opts, meta_note:str, logger:logging.Logger, lock=None) -> Clustering:
  # first, create new clustering from source identifier + postfix
  merged_localidentifier = f'{src_local_clusteringobj.identifier}[{src_local_clusteringobj.id}]:merged'
  merged_local_clustering_obj = get_or_create_db_clustering(dbh=dbh, clusteringtype='local', identifier=merged_localidentifier, settings=src_local_clusteringobj.setting, get_if_exist=True, note=meta_note, logger=logger, lock=lock)
  if merged_local_clustering_obj.numclusters <= 0 and not merged_local_clustering_obj.success:
    # create if not prepared before
    logger_merged_clustering_local = setup_logger_for_clustering(dbh, merged_local_clustering_obj.id)
    try:
      merged_local_clustering_obj = create_new_clusters_and_assignments_from_transitive_label(dbh, src_local_clusteringobj.id, merged_local_clustering_obj.id, localparams['emmodel'], float(localparams['alpha']), localparams['dim'], localparams['emaggregation'], logger_merged_clustering_local)
      # # update merged local clustering object
      # with dbh.sessionmaker() as session:
      #   merged_local_clustering_obj = session.get(Clustering, merged_local_clustering_obj.id, options=(sa.orm.undefer(Clustering.extrainfo), ))
    except BaseException as e:
      # update run merged_local_clustering_obj
      merged_local_clustering_obj = update_clustering_obj_values(dbh, merged_local_clustering_obj.id, 
        status='failed',
        success=False,
        finish=datetime.now(),
        extrainfo=( merged_local_clustering_obj.extrainfo if merged_local_clustering_obj.extrainfo is not None else { } ) | { 
          'exception': {
            'name': e.__class__.__name__,
            'message': str(e),
            'traceback': traceback.format_exc()
        }}
      )
      logger_merged_clustering_local.exception(e)
      logger_merged_clustering_local.error(f'Local clustering {merged_local_clustering_obj.id} failed.')
      raise # re-raise, don't let this exception go away
    finally:
      # cleanup db clustering logger
      cleanup_logger_for_clustering(logger_merged_clustering_local, merged_local_clustering_obj.id)
    internal_eval(dbh=dbh, src_local_clusteringid=src_local_clusteringobj.id, clusteringid=merged_local_clustering_obj.id, emmodel=localparams['emmodel'], vectordim=localparams['dim'], alphaval=float(localparams['alpha']), opts=internal_eval_opts, logger=logger)
  return merged_local_clustering_obj


def create_new_clusters_and_assignments_from_transitive_label(dbh:DBHandler, source_localclusteringid:int, merged_localclusteringid:int, emmodel:str, alphaval:float, vectordim:int, emaggregation:str, cl_logger:logging.Logger) -> Clustering | None:
  '''
  create new clustering with merged_knowns, i.e. transitivively applied known labels
  '''
  # create new clusters
  stmt_create_clusters = '''
    insert into clusters (clusteringid, label, extrainfo)
    select 
      :mergedclusteringid as clusteringid,
      extrainfo->>'transitive_label' as label,
      format('{ "transitive_label": %s, "isknown": %s, "lu_lemma": %s, "numelems": %s, "numelems_src": %s, "srcids": %s }', 
        to_jsonb((extrainfo->>'transitive_label')::text),
        to_jsonb(any_value((extrainfo->>'isknown')::bool)),
        array_to_json(array_agg(extrainfo->'lu_lemma')),
        sum((extrainfo->>'numelems')::integer),
        array_to_json(array_agg(extrainfo->'numelems')),
        array_to_json(array_agg(id))
      )::jsonb as extrainfo
    from clusters 
    where clusteringid = :sourceclusteringid
    group by extrainfo->>'transitive_label'
  '''
  # then create new cluster assignments
  stmt_create_assignments = '''
    insert into clusterassignments (instanceid, clusterid, extrainfo)
    select 
      ca.instanceid,
      cl.id,
      jsonb_set(
        jsonb_set(
          ca.extrainfo, 
          '{srcclusterid}',
          to_jsonb(ca.clusterid)
        ),
        '{assigned_by}',
        format('"%s__merged"', ca.extrainfo->>'assigned_by')::jsonb
      ) as new_assignmentinfo
    from clusterassignments ca
    join ( 
      select 
        jsonb_array_elements(extrainfo->'srcids')::int as srcid, 
        id, clusteringid, label, extrainfo from clusters
        where clusteringid = :mergedclusteringid
      ) cl on cl.srcid = ca.clusterid
  '''
  # apply everything within a single transaction block
  with dbh.sessionmaker() as session:
    with session.begin():
      session.execute(sa.text(stmt_create_clusters), {
        'sourceclusteringid': source_localclusteringid,
        'mergedclusteringid': merged_localclusteringid
      })
      session.execute(sa.text(stmt_create_assignments), {
        'mergedclusteringid': merged_localclusteringid
      })
    n_clusters_merged = session.execute(sa.select(sa.func.count()).select_from(Cluster).where(Cluster.clusteringid == merged_localclusteringid)).scalar()
  
  # generate new clusterembeddings
  cl_logger.info(f'Start generating and adding aggregated cluster embeddings...')
  add_aggregated_weighted_local_cluster_embeddings(dbh, source_localclusteringid, merged_localclusteringid, emmodel, alphaval, vectordim, emaggregation)
  cl_logger.info(f'Finshed generating and adding aggregated cluster embeddings.')
  
  # finally update clustering object values
  with dbh.sessionmaker() as session:
    clustering_obj = update_clustering_obj_values(dbh, merged_localclusteringid,
      status='finished',
      success=True,
      finish=datetime.now(),
      numclusters=n_clusters_merged,
      extrainfo=sa.func.jsonb_set( # chain func.jsonb_set to set multiple values in one go
        sa.func.jsonb_set(
          Clustering.extrainfo, 
          '{note}', 
          '"automatically merged from transitive label of instances with known labels across different lemmas from local clustering id @see src_local_clustering."'), 
        '{src_local_clustering}', 
        f'{source_localclusteringid}'
      )                          
    )
  return clustering_obj


def run_global_clustering_experiment(dbh:DBHandler, clusteringid:int, local_clusteringid:int, emaggregation:str, n_expect:int, filter:dict, merge_knowns:str, clusterer:ClusterMixin, device:torch.DeviceObjType|str, save_test_only:bool, cl_logger:logging.Logger):
  update_status(dbh, clusteringid, 'running')
  # helper method for internal clustering evaluation
  # will be overwritten depending on the clusterer's requirements
  def silhouette_samples_fun(*args, **kwargs): pass
  # retrieve feature matrix depending on the needs of the clusterer
  if clusterer.requires == 'similarity':
    A_G, ids, labels, ulabels, label_is_known = get_retrieve_cluster_similarity_matrix_dense(dbh, clusteringid, local_clusteringid, emaggregation, n_expect, filter, device, cl_logger)
    silhouette_samples_fun = partial(silhouette_samples, X=1-A_G, metric='precomputed')
  elif clusterer.requires == 'feature':
    A_G, ids, labels, ulabels, label_is_known = get_retrieve_cluster_feature_matrix(dbh, clusteringid, local_clusteringid, emaggregation, n_expect, filter, device, cl_logger)
    silhouette_samples_fun = partial(silhouette_samples, X=A_G, metric='cosine')
  elif clusterer.requires == 'instances':
    # TODO clusterer requires only the list of instances, this can be a simplified query, for now use feature matrix, but change this for the future!!!
    A_G, ids, labels, ulabels, label_is_known = get_retrieve_cluster_feature_matrix(dbh, clusteringid, local_clusteringid, emaggregation, n_expect, filter, device, cl_logger)
  elif clusterer.requires == 'clustering':
    raise TypeError(f"Clusterer {clusterer.__class__.__name__} is not applicable for global clustering. You are probaly looking for the 'ident' clusterer.")
  else:
    raise KeyError(f"Expected clusterer to expect either 'feature', 'similarity' or 'instance' but got '{clusterer.requires}'.")
  # label to transitive_label map, used if merge_knowns is either never or after_global
  labelu_to_transitivelabel_map = None
  # prepare fixed_labels input to CW
  if merge_knowns == 'after_local': # ulabels == labels, a transitive label map is not needed
    y_input = [ None if not known else label for known,label in zip(label_is_known, ulabels) ]
  elif merge_knowns == 'before_global': # formerly transitivelabel_as_yinput=true:
    y_input = [ None if not known else label for known,label in zip(label_is_known, labels) ]
  elif merge_knowns == 'after_global' or merge_knowns=='never': # 'never' == formerly transitivelabel_as_yinput=false
    y_input = [ None if not known else label for known,label in zip(label_is_known, ulabels) ]
    labelu_to_transitivelabel_map = { labelu: labels[i] for i, labelu in enumerate(ulabels) }
  else:
    raise ValueError(f"Parameter 'merge_knowns' must be one of {{'after_local', 'before_global', 'after_global', 'never'}} but is '{merge_knowns}'.")
  y_input_labels = set(y_input) - set([None]) # i.e. the fixed_labels explicitly removing the None object
  # prepare a local dataframe for ease of access
  local_df = pd.DataFrame(list(zip(ids, labels, ulabels, y_input)), columns=['inputcid', 'clabel', 'clabelu', 'yinput'])
  # RUN CLUSTERING
  cl_logger.info(f'Using {clusterer.name} for globally clustering {local_df.shape[0]} cluster instances with {sum(label_is_known)} fixed labels.')
  label_to_ids, list_of_labels = clusterer.fit_predict(A_G, y_input)
  # merge
  if merge_knowns == 'after_global':
    list_of_labels = [ labelu_to_transitivelabel_map.get(label, f'g::{label}') for label in list_of_labels ]
    label_to_ids_ = { }
    for label, ids in label_to_ids.items():
      label_ = labelu_to_transitivelabel_map.get(label, f'g::{label}') ##
      label_to_ids_[label_] = label_to_ids_.get(label_, []) + ids
    label_to_ids = label_to_ids_
  else:
    list_of_labels = [ clabel if clabel in y_input_labels else f'g::{clabel}' for clabel in list_of_labels ]
    label_to_ids = { clabel if clabel in y_input_labels else f'g::{clabel}':ids for clabel, ids in label_to_ids.items() }
  # compute silhouettes
  local_df['ypred'] = label_to_ids
  silhouette_sample_scores = np.zeros((len(list_of_labels), )) if not 1 < len(label_to_ids) < len(list_of_labels) else silhouette_samples_fun(labels=list_of_labels)
  silhouette_sample_scores = np.zeros((len(list_of_labels), )) if silhouette_sample_scores is None else silhouette_sample_scores ## happens when silhouette_samples_fun is not overwritten
  local_df['silhouette_by_local_cluster'] = silhouette_sample_scores.astype(float)
  cl_logger.info(f'Finished clustering {local_df.shape[0]} cluster instances with {sum(label_is_known)} fixed labels, got {len(label_to_ids.items())} clusters. Saving results...')
  # SAVE RESULTS IN DATABASE
  # first, get instance assigments to local clusters, call this transitive_clusterassignments
  with dbh.sessionmaker() as session:
    stmt = sa.select(ClusterAssignment).join(Cluster).where(Cluster.clusteringid == local_clusteringid).options(sa.orm.undefer(ClusterAssignment.extrainfo))
    res = session.scalars(stmt)
    transitive_clusterassignments = res.all()
  # get a mapping of clusterid to instances
  clusterid_to_transitiveassignments = { }
  for clusterassignment in transitive_clusterassignments:
    items = clusterid_to_transitiveassignments[clusterassignment.clusterid] = clusterid_to_transitiveassignments.get(clusterassignment.clusterid, list())
    items.append(clusterassignment)
  # create a new cluster obj in the database for each cluster, and
  # add cluster assigments transitively for each local cluster item
  new_clusters_as_dicts = [ ]
  new_clusterassignments_as_dicts = [ ]
  for cid, (clabel, cids) in enumerate(label_to_ids.items()):
    # cname = clabel if clabel in y_input_labels else f'g::{clabel}'
    new_clusters_as_dicts.append({
      'clusteringid': clusteringid, 
      'label': clabel, 
      'extrainfo' : {
        'transitive_label': clabel if labelu_to_transitivelabel_map is None else labelu_to_transitivelabel_map.get(clabel, clabel),
        'isknown': not clabel.startswith('g::'),
        'numelems': len(cids)
      }})
    # collect transitive items, ie. cluster assignments from the local cluster
    for item in cids:
      dfrow = local_df.loc[item]
      if dfrow.inputcid in clusterid_to_transitiveassignments.keys():
        new_clusterassignments_as_dicts += [ {
            'instanceid': transitive_cluster_item.instanceid,
            'clabel': clabel, # save unique clabel as reference for now as it will be replaced by clusterid
            'extrainfo': {
              'local_cluster_id': int(dfrow.inputcid),
              'assigned_by': 'globalstep',
              'fixed_label_input': dfrow.yinput,
              'silhouette_by_local_cluster': dfrow.silhouette_by_local_cluster
            }
          } for transitive_cluster_item in clusterid_to_transitiveassignments[dfrow.inputcid]
        ]
  # bulkinsert speeds things up
  if len(new_clusters_as_dicts):
    with dbh.sessionmaker() as session:
      res = session.execute(sa.insert(Cluster).returning(Cluster), new_clusters_as_dicts)
      clusters = res.scalars().all()
    clabel_to_cid = {cl.label: cl.id for cl in clusters}
    for cla in new_clusterassignments_as_dicts:
      cla['clusterid'] = clabel_to_cid[cla['clabel']]
      del cla['clabel']
    with dbh.sessionmaker() as session:
      session.execute(sa.insert(ClusterAssignment), new_clusterassignments_as_dicts)
  # compute mean silhouette scores
  transitive_sample_mean_silhouette_score = np.mean(list(map(lambda ca: ca['extrainfo']['silhouette_by_local_cluster'], new_clusterassignments_as_dicts)))
  local_cluster_mean_silhouette_score = local_df.silhouette_by_local_cluster.mean()
  # UPDATE clustering obj in database
  n_clusters = len(label_to_ids)
  with dbh.sessionmaker() as session:
    session.execute(sa.update(Clustering).where(Clustering.id == clusteringid).values(
      status='finished',
      success=True,
      finish=datetime.now(),
      numclusters=n_clusters,
      extrainfo=sa.func.jsonb_set( # chain func.jsonb_set to set multiple values in one go
        sa.func.jsonb_set(
          Clustering.extrainfo, 
          '{mean_silhouette_score_by_local_cluster}',
          f'{local_cluster_mean_silhouette_score}'), 
        '{mean_silhouette_score_by_local_cluster_ta}', 
        f'{transitive_sample_mean_silhouette_score}'
      )
    ))
  return n_clusters


def run_nn_assignment_single_step(dbh:DBHandler, src_clusteringid:int, clusteringid:int, emmodel:str, nn_type, cl_logger:logging.Logger) -> None:
  cl_logger.info(f'Start filling missing instances...')
  # check if there are actually unassigned instances that have to be assigned via knn
  stmt = sa.text(f'''
    select 
      count(instanceid)
    from evaltable(:clusteringid) 
    where cid is null 
  ''')
  with dbh.sessionmaker() as session:
    res = session.execute(stmt, {'clusteringid': clusteringid })
    num_unassigned = res.scalar_one()
  cl_logger.info(f'Assigning nearest neighboring cluster to {num_unassigned} unassigned instances.')
  if num_unassigned <= 0:
    cl_logger.info(f'Finished filling missing instances.')
    return
  # find the unassigned test instances and add them directly to the clusterassignments table
  stmt = sa.text(f'''
    with unassigned_instances as (
      select 
        instanceid
      from evaltable(:clusteringid) 
      where cid is null
    ), vectorinstances as (
      select 
        ui.instanceid, 
        fsv.vector as embedding
      from unassigned_instances ui
      join frameinstances_split_vectorized__{src_clusteringid} fsv
      on fsv.instance_id = ui.instanceid
    ), nearestneighbors as (
      select 
        vi.instanceid,
        nn.*
      from vectorinstances vi
      cross join lateral (
        select 
          *, 
          row_number() over (order by _nn.cosim desc) as k
        from (
          select 
            clusterid, 
            1 - (vc.embedding <=> vi.embedding) as cosim 
          from clusterembeddings__{clusteringid} vc
          order by cosim desc
          limit 1
        ) _nn
      ) nn
    ), nearestneighborassignments as (
      select 
        nn.instanceid,
        nn.clusterid, 
        format('{{ "assigned_by": "nn", "nn": {{ "type": "{nn_type}", "emmodel": "{emmodel}", "clusterembeddings": "clusterembeddings__{clusteringid}", "sim": "cosim", "rank": %s, "val": "%s" }} }}', nn.k, to_char(nn.cosim, '0D9999'))::jsonb as extrainfo
      from nearestneighbors nn
    )
    insert into clusterassignments (instanceid, clusterid, extrainfo)
      select 
        nna.instanceid,
        nna.clusterid, 
        nna.extrainfo
      from nearestneighborassignments nna;
  ''')
  with dbh.sessionmaker() as session:
    res = session.execute(stmt, {'clusteringid': clusteringid})
  cl_logger.info(f'Finished filling missing instances.')
  return


def internal_eval(dbh:DBHandler, src_local_clusteringid:int, clusteringid:int, emmodel:str, vectordim:int, alphaval:float, opts:dict, logger=None) -> None:
  n_silhouette_sample_draws = int(opts['silhouette']['ndraws'])
  for i in range(n_silhouette_sample_draws):
    __internal_eval_silhouette__(dbh=dbh, src_local_clusteringid=src_local_clusteringid, clusteringid=clusteringid, emmodel=emmodel, vectordim=vectordim, alphaval=alphaval, opts=opts['silhouette'], logger=logger)
  return


def __internal_eval_silhouette__(dbh:DBHandler, src_local_clusteringid:int, clusteringid:int, emmodel:str, vectordim:int, alphaval:float, opts:dict, logger=None):
  nsamples = int(opts.get('nsamples', 1e3))
  order_by = 'random()' if opts.get('randomize_order', False) else 'inst.instanceid'
  rseed = opts.get('random_seed', None)
  setseed_str = f'setseed({rseed}),' if rseed is not None and opts.get('randomize_order', False) else ''
  # NOTE TODO: this selects only the test instances, if all instances are wanted here use 
  #   with cls as (
  # 	  select * from clusterings where id = 17472
  #   )
  #   select * from evaltable((select id from cls), (select datasetsplit_id from cls), (select splits from cls))
  stmt = sa.text('''
    with instances as (
      select 
        inst.instanceid, 
        inst.split,
        inst.label_true, 
        inst.clabel,
        inst.cid, 
        inst.clusterinfo, 
        inst.assignmentinfo, 
        inst.instanceinfo
      from {setseed_str} evaltable(:clusteringid) inst
      where inst.cid is not null
      order by {order_by}
      limit :nsamples
    ), vectorinstances as (
      select 
        inst.instanceid, 
        inst.split,
        inst.label_true, 
        inst.clabel,
        inst.cid, 
        inst.clusterinfo, 
        inst.assignmentinfo, 
        inst.instanceinfo,
        fsv.vector
      from instances inst
      join frameinstances_split_vectorized__{src_local_clusteringid} fsv
      on fsv.instance_id = inst.instanceid
    )
    select 
      a.instanceid as id_a, 
      b.instanceid as id_b,
      a.split as split_a,
      b.split as split_b,
      b.cid as cid_b,
      b.label_true as truelabel_b,
      b.clabel as clabel_b,
      b.clusterinfo->>'transitive_label' as tlabel_b,
      (a.vector <=> b.vector) as cosinedist
    from vectorinstances a
    cross join vectorinstances b 
    order by a.instanceid, a.split, b.instanceid, b.split;
  '''.format(src_local_clusteringid=src_local_clusteringid, setseed_str=setseed_str, order_by=order_by)) # query similarity matrix of sampled assignments
  logger.info(f'Retrieving similiraity values for {nsamples} random test samples in to compute a silhouette score.')
  with dbh.sessionmaker() as session:
    # get the adjacency matrix of wheighted instances and their labels
    res = session.execute(stmt, {
      'clusteringid': clusteringid,
      'nsamples': nsamples
    })
    rows_t = dict(zip(res.keys(), zip(*res)))
  if len(rows_t) <= 0:
    logger.error(f'Retrieved 0 similarity values.')
    raise ValueError('Nothing to compute silhouette scores from, similarity matrix is empty.')
  logger.info(f"Retrieved {len(rows_t['cosinedist'])} similarity values.")
  # prepare similarity matrix
  dists_cos = torch.tensor(rows_t['cosinedist'], dtype=torch.float32)
  # since we have a square matrix the number of elements should be the squareroot of the result length % TODO: this could be optimzed to return results already filered by cosim
  num_elems = int(math.sqrt(dists_cos.size(0)))
  logger.info(f'Creating {num_elems} x {num_elems} matrix.')
  # reshape matrix
  D = dists_cos.view(num_elems, -1)
  # get all other information from rows
  sampled_ids = rows_t['id_b'][:num_elems]
  sampled_cids = rows_t['cid_b'][:num_elems]
  sampled_nclusters = len(set(sampled_cids))
  sampled_labels_t = rows_t['tlabel_b'][:num_elems]
  sampled_nclusters_t = len(set(sampled_labels_t))
  # compute silhouette scores on the cluster id / unique cluster label and the transitive cluster label
  logger.info(f'Computing silhouette scores...')
  silhouette_sample_scores = np.zeros((len(sampled_ids), )) if not 1 < sampled_nclusters < len(sampled_ids) else silhouette_samples(X=D, labels=sampled_cids, metric='precomputed')
  mean_silhouette_sample_score = np.mean(silhouette_sample_scores, dtype=float)
  silhouette_sample_scores_t = np.zeros((len(sampled_ids), )) if not 1 < sampled_nclusters_t < len(sampled_ids) else silhouette_samples(X=D, labels=sampled_labels_t, metric='precomputed')
  mean_silhouette_sample_score_t = np.mean(silhouette_sample_scores_t, dtype=float)
  logger.info(f'Mean silhouette score {mean_silhouette_sample_score:.4f}')
  logger.info(f'Mean silhouette score (transitive) {mean_silhouette_sample_score_t:.4f}')
  logger.info(f'Saving results...')
  # SAVE results
  with dbh.sessionmaker() as session:
    with session.begin():
      clustering_obj = session.get(Clustering, clusteringid, options=(sa.orm.undefer(Clustering.extrainfo),))
      if clustering_obj.extrainfo is None:
        clustering_obj.extrainfo = { }
      if not 'internaleval' in clustering_obj.extrainfo:
        clustering_obj.extrainfo['internaleval'] = { }
      if not 'silhouette' in clustering_obj.extrainfo['internaleval']:
        clustering_obj.extrainfo['internaleval']['silhouette'] = [ ]
      if not 'silhouette_transitive' in clustering_obj.extrainfo['internaleval']:
        clustering_obj.extrainfo['internaleval']['silhouette_transitive'] = [ ]
      clustering_obj.extrainfo['internaleval']['silhouette'].append(mean_silhouette_sample_score)
      clustering_obj.extrainfo['internaleval']['silhouette_transitive'].append(mean_silhouette_sample_score_t)
    # commit is implicit on exit
  logger.info(f'...done.')
  return
  
  
def evaluate_clustering(dbh:DBHandler, clusteringid:int, re_evaluate=False, logger=None):
  with dbh.sessionmaker() as session:
    clustering_obj = session.get(Clustering, clusteringid, options=(sa.orm.undefer(Clustering.extrainfo),))
  if not re_evaluate and clustering_obj.extrainfo is not None and 'evalresults' in clustering_obj.extrainfo:
    return clustering_obj.extrainfo['evalresults']
  splits = clustering_obj.splits
  testsplits = clustering_obj.setting['data']['testsplits'] # data['testsplits']
  trainsplits = list(set(splits)-set(testsplits))
  # query label counts from train split instances
  stmt = sa.text('''
    select label, support, instances from knownlabels(:clusteringid) order by support desc
  ''')
  logger.info(f'Evaluating clustering {clusteringid} (test={testsplits} of known={trainsplits})')
  with dbh.sessionmaker() as session:
    res = session.execute(stmt, {'clusteringid': clusteringid})
    rows_t = dict(zip(res.keys(), zip(*res)))
  known_labels_train, known_labels_train_support, instances = (rows_t['label'], rows_t['support'], rows_t['instances']) if len(rows_t) else ([],[], [])
  known_train_labelset = set(known_labels_train)
  assert len(known_labels_train) == len(known_train_labelset)
  #
  stmt = sa.text('''
    select instanceid, split,
      label_true, clabel,
      lu_lemma, cid, 
      clusterinfo, assignmentinfo, instanceinfo
    from evaltable(:clusteringid)
    where cid is not null
		and label_true is not null 
		and not label_true = '<unk>';
  ''')
  with dbh.sessionmaker() as session:
    res = session.execute(stmt, {'clusteringid': clusteringid})
    rows_t = dict(zip(res.keys(), zip(*res)))
  if len(rows_t) <= 0:
    return {'error': 'No test instances with a gold label.'}
  instanceids, instancessplit, labels_true, uniqueclusterlabels, lu_lemma, clusterids, clusterinfo, assignmentinfo, instanceinfo = rows_t['instanceid'], rows_t['split'], rows_t['label_true'], rows_t['clabel'], rows_t['lu_lemma'], rows_t['cid'], rows_t['clusterinfo'], rows_t['assignmentinfo'], rows_t['instanceinfo'] # consistency check: fixed_label_inputs  should all be none for a local clustering and only local clusterlabels with known classes or None for global clusterings
  #
  label_true_is_known = [(l in known_train_labelset) for l in labels_true]
  labels_pred, label_pred_is_known = zip(*map(lambda info: (info['transitive_label'], info['isknown']), clusterinfo))
  #
  test_labelset = set(labels_true) # contains classes and novel classes from the test gold instances
  predicted_labelset = set(labels_pred) # contains classes and clusters from the test predictions
  #
  any_known_label = known_train_labelset | test_labelset # contains classes from training and! testing
  any_known_label_list = list(any_known_label)
  any_label = any_known_label | predicted_labelset # contains classes from training and testing and all cluster labels
  #
  predicted_known_labels = predicted_labelset & known_train_labelset # intersection, contains only those known classes from test predictions, not the gold instances
  known_labels_test = known_train_labelset & test_labelset # 
  clusters_labelset = predicted_labelset - known_train_labelset # set minus, contains only clusters
  all_possible_labels = predicted_labelset | known_train_labelset # union contains classes and clusters
  expected_novel_labelset = set([l for l in labels_true if l not in known_train_labelset])
  #
  majority_class_label, majority_class_label_support, i = (known_labels_train[0], known_labels_train_support[0], 0) if len(known_labels_train) > 0 else ('None', -1, -1)
  if len(predicted_known_labels) > 0:
    while majority_class_label not in predicted_known_labels and i < len(known_labels_train)-1:
      i+=1
      majority_class_label, majority_class_label_support = known_labels_train[i], known_labels_train_support[i]
    if i == len(known_labels_train):
      logger.warning('...')
  logger.info(f'Majority class: {majority_class_label} (support {majority_class_label_support}, label {i+1})')
  #
  labels_true_with_outlier = [l if l_known else "outlier" for l, l_known in zip(labels_true, label_true_is_known)]
  labels_pred_with_outlier = [l if l_known else "outlier" for l, l_known in zip(labels_pred, label_pred_is_known)]
  #
  labels_pred_with_majority = [l if l_known else majority_class_label for l, l_known in zip(labels_pred, label_pred_is_known)]
  #
  results = {
    'class_labels': {
      'all (clusters + known classes + novel classes)': len(any_label),
      'all possible (clusters + known classes)': len(all_possible_labels), 
      'predicted labels (known + clusters)': len(predicted_labelset), 
      'expected predicted (partially-known w/o novel)': len(known_labels_test),
      'expected predicted (partially-known+novel)': len(test_labelset),
      'expected predicted any (known+novel)': len(any_known_label),
      'known': len(known_labels_train), 
      'predicted known (w/o clusters)': len(predicted_known_labels), 
      'predicted clusters (w/o known classes)': len(clusters_labelset), 
      'novel expected (w/o known classes)': len(expected_novel_labelset)
    },
    'novelty_detection': novelty_detection(predicted=label_pred_is_known, gold=label_true_is_known, known_labels=known_labels_train, legacy_edits=False),
    'frame_identification': known_frame_identification(predicted=labels_pred, gold=labels_true, known_labels=known_labels_train),
    'frame_identification_alleval': frame_identification(predicted=labels_pred, gold=labels_true, labels=any_known_label_list),
    'frame_induction': unknown_frame_induction(predicted=labels_pred, gold=labels_true,known_labels=known_labels_train),
    'frame_induction_alleval': frame_induction(predicted=labels_pred, gold=labels_true),    
    'novelty_+_frame': frame_identification(predicted=labels_pred_with_outlier, gold=labels_true_with_outlier, labels=any_known_label_list+['outlier']), # novelty_and_frame_identification(predicted=labels_pred_with_outlier, gold=labels_true_with_outlier, known_labels=known_labels_train, legacy_edits=False)
    'frame_identification_majority_fb': frame_identification(predicted=labels_pred_with_majority, gold=labels_true, labels=any_known_label_list), # novelty_and_frame_identification(predicted=labels_pred_with_outlier, gold=labels_true_with_outlier, known_labels=known_labels_train, legacy_edits=False)
  }
  # SAVE results
  with dbh.sessionmaker() as session:
    with session.begin():
      clustering_obj = session.get(Clustering, clusteringid)
      if clustering_obj.extrainfo is None:
        clustering_obj.extrainfo = {'evalresults': results}
      else:
        clustering_obj.extrainfo['evalresults'] = results
  # RETURN
  return results


def run_or_retrieve_local_clustering(dbh:DBHandler, factory, dataparams:dict, localparams:dict, device:torch.DeviceObjType|str, save_test_only:bool, reuse_local:bool, meta_note:str|list|dict, logger=None, lock=None):

  # define local identifier
  localidentifier_settings = {'data': dataparams, 'local': localparams}
  localidentifier = Clustering.get_identifier_from_settings(localidentifier_settings)

  logger.info(f'Initializing local clustering with identifier {localidentifier} from commit {git_revision_short_hash}...')
  local_clustering_obj = get_or_create_db_clustering(dbh=dbh, clusteringtype='local', identifier=localidentifier, settings=localidentifier_settings, get_if_exist=reuse_local, note=meta_note, logger=logger, lock=lock)
  if local_clustering_obj.numclusters <= 0 and not local_clustering_obj.success:
    logger_clustering_local = setup_logger_for_clustering(dbh, local_clustering_obj.id)
    logger_clustering_local.info(f'Local clustering {local_clustering_obj.id} ({localidentifier}) initialized.')
    try:
      # instantiate clusterer via factory
      local_clusterer = factory.create_from_name('dbclusterer', localparams['clusterer'])
      local_clusterer.setLogger(logger_clustering_local)
      run_local_clustering_experiment(dbh, factory, dataparams, local_clustering_obj.id, localparams['emmodel'], localparams['dim'], float(localparams['alpha']), localparams['emaggregation'], localparams.get('filter',{}), local_clusterer, device, save_test_only, logger_clustering_local, lock, meta_note)
      logger_clustering_local.info(f'Local clustering {local_clustering_obj.id} ({localidentifier}) finished.')
      # cleanup local clusterer
      local_clusterer.finalize()
      del local_clusterer
    except BaseException as e:
      # update run local_clustering_obj
      local_clustering_obj = update_clustering_obj_values(dbh, local_clustering_obj.id, 
        status='failed',
        success=False,
        finish=datetime.now(),
        extrainfo=( local_clustering_obj.extrainfo if local_clustering_obj.extrainfo is not None else { } ) | { 
          'exception': {
            'name': e.__class__.__name__,
            'message': str(e),
            'traceback': traceback.format_exc()
        }}
      )
      logger_clustering_local.exception(e)
      logger_clustering_local.error(f'Local clustering {local_clustering_obj.id} failed.')
      raise # re-raise, don't let this exception go
    finally:
      # cleanup db clustering logger
      cleanup_logger_for_clustering(logger_clustering_local, local_clustering_obj.id)
    # update local clustering object
    with dbh.sessionmaker() as session:
      local_clustering_obj = session.get(Clustering, local_clustering_obj.id, options=(sa.orm.undefer(Clustering.extrainfo),))
  else:
    create_instances_view_if_not_exists(dbh, dataparams, local_clustering_obj.id, localparams['emmodel'], localparams['dim'], float(localparams['alpha']))
    logger.info(f'Reusing local clustering {local_clustering_obj.id} ({localidentifier}).')

  return local_clustering_obj

# run local global clustering experiment
def run_experiment(dbh:DBHandler, dataparams:dict, localparams:dict, globalparams:dict, device:torch.DeviceObjType|str, save_test_only:bool, reuse_local:bool, reuse_global:bool, re_evaluate_local:bool, re_evaluate_global:bool, meta_note:str|list|dict, internal_eval_opts:dict, logger=None, lock=None):
  if not len(set(dataparams['testsplits']) | set(dataparams['splits'])) == len(dataparams['splits']):
    raise RuntimeError('All names in testsplits must exist in splits!')

  # setup clusterer factory
  factory = create_factory()

  ### LOCAL
  local_clustering_obj = run_or_retrieve_local_clustering(dbh, factory, dataparams, localparams, device, save_test_only, reuse_local, meta_note, logger, lock)

  # get (or create and get) merged local clustering
  # TODO: think about it: create this thing always or only on demand if merge_knowns==after_local?
  # local_merged_clustering_obj = get_merged_local_clustering(dbh, local_clustering_obj, localparams, internal_eval_opts, meta_note, logger)

  ### EVAL LOCAL
  if re_evaluate_local:
    # internal eval
    internal_eval(dbh=dbh, src_local_clusteringid=local_clustering_obj.id, clusteringid=local_clustering_obj.id, emmodel=localparams['emmodel'], vectordim=localparams['dim'], alphaval=float(localparams['alpha']), opts=internal_eval_opts, logger=logger)
  # external eval (merged clustering and unmerged clustering should have same evaluation scores since the transitive label is used in evaluation)
  eval_results_local = evaluate_clustering(dbh, local_clustering_obj.id, re_evaluate_local, logger)
  logger.info(f'\n=== Local Clustering Eval Results {local_clustering_obj.identifier}@{local_clustering_obj.id} ===\n'+pprint.pformat(eval_results_local.get('micro avg', eval_results_local), depth=3)+'...\n')
  
  ### GLOBAL 
  # check which local clustering to use for global clustering
  local_clustering_obj_for_g = local_clustering_obj
  if globalparams['localclustering'] == '##local@latest':
    # get the merged local clustering on demand, i.e. only if it is required
    if globalparams.get('merge_knowns', 'after_global') == 'after_local':
      # get (or create and get) merged local clustering
      # use this as localclustering obj for the global clustering (only if globalparams.merge_knowns == 'after_local')
      local_clustering_obj_for_g = get_merged_local_clustering(dbh, local_clustering_obj_for_g, local_clustering_obj_for_g.setting['local'], internal_eval_opts, local_clustering_obj_for_g.extrainfo['note'], logger, lock)
    # update global parameters for saving purposes
    globalparams['localclustering'] = f'{local_clustering_obj_for_g.identifier}@{local_clustering_obj_for_g.id}'
  else: # get the referenced local clustering and keep the params entry
    local_clustering_obj_for_g = get_clustering_obj_from_identifier_string(dbh, globalparams['localclustering'], globalparams.get('merge_knowns', 'after_global')=='after_local', internal_eval_opts, logger)
  # prepare global clustering
  globalidentifier_settings = {'data': dataparams, 'global': globalparams}
  globalidentifier = Clustering.get_identifier_from_settings(globalidentifier_settings)
  # init
  logger.info(f'Initializing global clustering with identifier {globalidentifier} from commit {git_revision_short_hash}...')
  global_clustering_obj = get_or_create_db_clustering(dbh=dbh, clusteringtype='localglobal', identifier=globalidentifier, settings=globalidentifier_settings, get_if_exist=reuse_global, note=meta_note, logger=logger, lock=lock)
  if global_clustering_obj.numclusters <= 0 and not global_clustering_obj.success:
    logger_clustering_global = setup_logger_for_clustering(dbh, global_clustering_obj.id)
    logger_clustering_global.info(f'Global clustering {global_clustering_obj.id} ({globalidentifier}) initialized.')
    try:
      logger_clustering_global.info(f'Using local clustering {local_clustering_obj_for_g.id} for global clustering {global_clustering_obj.id}.')
      # update obj
      global_clustering_obj = update_clustering_obj_values(
        dbh, global_clustering_obj.id, 
        extrainfo=( global_clustering_obj.extrainfo if global_clustering_obj.extrainfo is not None else { } ) | { 'clusteringid_local': local_clustering_obj_for_g.id }
      )
      emaggregation = local_clustering_obj_for_g.setting['local']['emaggregation']
      vectordim = local_clustering_obj_for_g.setting['local']['dim']
      emmodel = local_clustering_obj_for_g.setting['local']['emmodel']
      # instantiate clusterer via factory
      global_clusterer = factory.create_from_name('dbclusterer', globalparams['clusterer'])
      global_clusterer.setLogger(logger_clustering_global)
      run_global_clustering_experiment(dbh, global_clustering_obj.id, local_clustering_obj_for_g.id, emaggregation, local_clustering_obj_for_g.numclusters, globalparams.get('filter',{}), globalparams.get('merge_knowns', 'after_global'), global_clusterer, device, save_test_only, logger_clustering_global)
      logger_clustering_global.info(f'Global clustering {global_clustering_obj.id} ({globalidentifier}) finished.')
      # cleanup global clusterer
      global_clusterer.finalize()
      del global_clusterer
      # generate global cluster embeddings
      logger_clustering_global.info(f'Start generating and adding aggregated cluster embeddings...')
      add_aggregated_global_cluster_embeddings(dbh, global_clustering_id=global_clustering_obj.id, local_clustering_id=local_clustering_obj_for_g.id, emaggregation=emaggregation, vectordim=vectordim)
      logger_clustering_global.info(f'Finished generating and adding aggregated cluster embeddings.')
      ### FILL MISSING (filtered test instances)
      src_local_clusteringid = local_clustering_obj_for_g.extrainfo.get('src_local_clustering', local_clustering_obj_for_g.id)
      run_nn_assignment_single_step(dbh=dbh, src_clusteringid=src_local_clusteringid, clusteringid=global_clustering_obj.id, emmodel=emmodel, nn_type='global', cl_logger=logger_clustering_global)
    except BaseException as e:
      # update global_clustering_obj
      global_clustering_obj = update_clustering_obj_values(dbh, global_clustering_obj.id, 
        status='failed',
        success=False,
        finish=datetime.now(),
        extrainfo=( global_clustering_obj.extrainfo if global_clustering_obj.extrainfo is not None else { } ) | { 
          'exception': {
            'name': e.__class__.__name__,
            'message': str(e),
            'traceback': traceback.format_exc()
        }}
      )
      logger_clustering_global.exception(e)
      logger_clustering_global.error(f'Local clustering {global_clustering_obj.id} failed.')
      raise # re-raise, don't let this exception go
    finally:
      # cleanup db clustering logger
      cleanup_logger_for_clustering(logger_clustering_global, global_clustering_obj.id)
    # update global clustering object
    with dbh.sessionmaker() as session:
      global_clustering_obj = session.get(Clustering, global_clustering_obj.id)
  else:
    logger.info(f'Reusing global clustering {global_clustering_obj.id} ({globalidentifier}).')

  ### EVAL GLOBAL
  if re_evaluate_global:
    # internal
    internal_eval(dbh=dbh, src_local_clusteringid=local_clustering_obj_for_g.extrainfo.get('src_local_clustering', local_clustering_obj_for_g.id), clusteringid=global_clustering_obj.id, emmodel=localparams['emmodel'], vectordim=localparams['dim'], alphaval=float(localparams['alpha']), opts=internal_eval_opts, logger=logger)
  # external
  eval_results_global = evaluate_clustering(dbh, global_clustering_obj.id, re_evaluate_global, logger)
  logger.info(f'\n=== Global Clustering Results {global_clustering_obj.identifier}@{global_clustering_obj.id} ===\n'+pprint.pformat(eval_results_global, depth=3)+'...\n')
  return {
    'local': {'clusteringid': local_clustering_obj.id, 'numclusters': local_clustering_obj.numclusters},
    'global': {'clusteringid': global_clustering_obj.id, 'numclusters': global_clustering_obj.numclusters}
  }


def run_with_params_DB(dbh:DBHandler, params__:dict, await_key_confirmation=True, logger=__base_logger, lock=None):
  params = deepcopy(params__) # make sure that shared parameters are never overwritten from within the application
  logger.info('\n\n'+pprint.pformat(params)+'\n')
  params_meta = params['meta']
  params_data = params['data']
  params_local = params['local']
  params_global = params['global']
  # get the device
  device = torch.device(params_meta['device'])
  # check if clusterings should be reused or not
  reuse_local = params_meta['reuse']['local']
  reuse_global = params_meta['reuse']['global']
  # check if clusterings should be re-evaluated if they exist
  re_evaluate_local = params_meta['re-evaluate']['local']
  re_evaluate_global = params_meta['re-evaluate']['global']
  # save all clustererd items, i.e. train too? or only test?
  save_test_only = params_meta.get('save_test_only', False)
  # note to add to extrainfo, can be an object or a string or list
  meta_note = params_meta['note']
  # options for internal evaluation
  internal_eval_opts = params_meta['internaleval']
  # wait for user input?
  if await_key_confirmation:
    input("Press Key to continue...")
  # START experiment
  ret_val = run_experiment(dbh, params_data, params_local, params_global, device, save_test_only, reuse_local, reuse_global, re_evaluate_local, re_evaluate_global, meta_note, internal_eval_opts, logger, lock)
  logger.info('\n\n'+pprint.pformat(ret_val)+'\n')
  return ret_val
  

def run_with_params(params__:dict, await_key_confirmation=True, logger=__base_logger, dburl=__default_db_url__):
  # setup db connection
  dbh: DBHandler = setup_database_handler(dburl)
  ret_val = run_with_params_DB(dbh, params__, await_key_confirmation, logger)
  # forcibly cleanup dbhandler (avoid connection leaks)
  dbh.dispose()
  return ret_val


if __name__ == '__main__':
  import argparse
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('filename', nargs='?')
  args = arg_parser.parse_args()
  ## load parameters from json file if given as argument
  if args.filename is not None:
    with open(args.filename, 'r') as jsonfile:
      params = merge_with_default_params(json.load(jsonfile))
  else:
    params = default_parameters
  # stop execution and wait for confirmation if executed from this file
  ret_val = run_with_params(params, await_key_confirmation=True, logger=__base_logger)
