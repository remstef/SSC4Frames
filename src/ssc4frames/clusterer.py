import abc
from collections import Counter
import csv
from gzip import open as gzopen
from io import open
import numbers
import os
import re
from functools import partial
from collections.abc import Callable
from pathlib import Path
from itertools import product as it_product, combinations as it_combinations
import typing

import numpy as np
import pandas as pd
import torch

from ssc4frames.helpers import get_obj_hash

__executed_import_funcs = set()

def import_required_on_init(import_func):
  if import_func.__name__ not in __executed_import_funcs:
    import_func()
    __executed_import_funcs.add(import_func.__name__)

def import_conll():
  global parse_incr, literal_eval
  from conllu import parse_incr
  from ast import literal_eval

def import_cw():
  global ptgcl
  import pt_graph_cluster_lib as ptgcl
  
def import_sklearn():
  global AgglomerativeClustering, sklearn
  from sklearn.cluster import AgglomerativeClustering
  import sklearn.linear_model
  import sklearn.metrics
  import sklearn.preprocessing

def import_pyclustering_xmeans():
  global xmeans, kmeans_plusplus_initializer
  from pyclustering.cluster.xmeans import xmeans
  from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
  # needed for older version of numpy in pyclustering
  import warnings
  np.warnings = warnings


class DBClusterer(metaclass=abc.ABCMeta):

  def __init__(self, min_support=None, min_accuracy=None):
    self.min_support = min_support
    self.min_accuracy = min_accuracy

  def setLogger(self, logger):
    self.logger = logger

  def finalize(self):
    pass

  def _get_labels_for_clustering(self, label_vector, y, constraints_violated=False):

    if not constraints_violated:
      ## get labels for clusters with labelled instances
      cluster_label_2_frame_label = {}
      for cluster_label, frame_label in zip(label_vector, y):
        if frame_label is not None:
          cluster_label_2_frame_label[cluster_label] = frame_label

      return [cluster_label_2_frame_label[k] if k in cluster_label_2_frame_label else str(k)
              for k in label_vector]
    else:

      if (self.min_support is None) or (self.min_accuracy is None):
        raise ValueError('min_support and min_accuracy need to be set in order to get labels for clusterings with violated constraints')

      # get a dict of cluster label to list of items - with original labels
      label_to_items = {}
      for i, l in enumerate(label_vector):
        items = label_to_items[l] = label_to_items.get(l, list())
        items.append(i)

      # get the most frequent label, check min_support and min_accuracy
      label_vector_str = [str(k) for k in label_vector]
      for numeric_label, cluster_items in label_to_items.items():

        labels_for_cluster = [y[i] for i in cluster_items if y[i] is not None]
        if not labels_for_cluster:
          ## no known label in the given cluster
          string_label = None
        else:
          string_label, label_freq = Counter(labels_for_cluster).most_common(1)[0]

          if label_freq < self.min_support:
            ## not enough support for label
            string_label = None
          elif label_freq/len(labels_for_cluster) < self.min_accuracy:
            ## known labels too ambiguous
            string_label = None

        ## set labels in label_vector_str
        if string_label is not None:
          for i in cluster_items:
            label_vector_str[i] = string_label

        return label_vector_str

  def _get_label_to_items_from_label_vector(self, label_vector):

    # get a dict of cluster label to list of items
    label_to_items = {}
    for i, l in enumerate(label_vector):
      items = label_to_items[l] = label_to_items.get(l, list())
      items.append(i)

    return label_to_items

  def _assert_constraints(self, label_vector, y):

    for i, j in it_product(range(len(y)), range(len(y))):
      if y[i] is None or y[j] is None:
        continue
      if y[i] == y[j]:
        if not (label_vector[i] == label_vector[j]):
          raise ValueError("Not all labelled examples are clustered together.")
        continue
      if y[i] != y[j]:
        if not (label_vector[i] != label_vector[j]):
          raise ValueError("Labelled examples with different labels are clustered together.")

  @abc.abstractmethod
  def fit_predict(self, X, y=None):
    pass


# base class for Clusterers that create multiple clusterings,
# e.g. hierarchical clustering
class MultiClusterer(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def get_clusterings(self, X, y=None, min_clusters=None):
    pass

  @abc.abstractmethod
  def get_requirement(self):
    pass

  def setLogger(self, logger):
    self.logger = logger

class OrderedMultiClusterer(MultiClusterer):
  pass



class CW(DBClusterer):

    name = 'cw'
    requires = 'similarity'

    __cw_default_args__ = {
      'binarize': True, 
      'max_iter': 100, 
      'label_by': 'max_label', 
      'weighting': 'top', 
      'optimize_space': False, 
      'verbose': False, 
      'quiet': True
    }

    @staticmethod
    def __criterion_fun_mean_local_clustering_coefficient_is_smaller_or_equal_to_val__(val):
      return lambda M: ptgcl.cl.mean_local_clustering_coefficient(M, sample=1000) <= val 
    
    @staticmethod
    def __parse_criterion_from_str__(criterion_str)-> float|Callable:
      if criterion_str.startswith('minw_'):
        return float(criterion_str[5:])
      elif criterion_str.startswith('lmc<='):
        thresh = float(criterion_str[5:])
        criterion_fun = CW.__criterion_fun_mean_local_clustering_coefficient_is_smaller_or_equal_to_val__(thresh)
        return partial(ptgcl.cl.keep_topk_edges_by_criterium, stepsize=0.05, criterium=criterion_fun, add_edges=True)
      elif criterion_str.startswith('umap_'):
        regex = r'^umap_(?P<optk>[a-z]+)(?P<k>\d+)(_sym:(?P<sym>[a-z]+))?$'
        pattern = re.compile(regex)
        match = pattern.search(criterion_str)
        if not match:
          raise ValueError(f"Pattern '{regex}' not found.")
        param_optk = {
          'optkv': 'values',
          'optkw': 'weights',
          'k': None 
        }.get(match.group('optk'), None)
        param_k = int(match.group('k'))
        param_symmetrize = match.group('sym')
        cosim_to_dist = lambda M: 2-(M+1) # cosine is in [-1,1] => add one to be in [0,2] then make it a distance by subtracting it from 2 such that 0 is most similar (nearest) and 2 is farthest (most dissimilar)
        umap_fun = lambda A, *args, **kwargs: ptgcl.cl.umap_style_graph(A=cosim_to_dist(A), *args, **kwargs)
        return partial(umap_fun, topk=param_k, apply_optimal_k_heuristic=param_optk, symmetrize=param_symmetrize)
      else:
        raise RuntimeError(f"Unknown criterion '{criterion_str}'.")

    def __init__(self, criterion:str, random_state:int = None, cwargs:dict = {}):
      import_required_on_init(import_cw)
      super().__init__()
      self.criterion_str:str = criterion
      self.random_state = random_state
      cwargs['random_seed'] = 0 if self.random_state is None else self.random_state
      self.cw_params = CW.__cw_default_args__ | cwargs
      self.criterion_obj:float|Callable = CW.__parse_criterion_from_str__(criterion)

    def fit_predict(self, A, y=None):
      # run pruning and clustering
      if isinstance(self.criterion_obj, numbers.Number):
        _A = A
        _minweight = self.criterion_obj
      else:
        # prune A explicitly, criterion_obj is a partial function
        _A = self.criterion_obj(A)
        _minweight = float('-inf') # explicitly set minweight to minus infinity to avoid internal pruning

      # pruning is done internally by setting minweight 
      clustered = ptgcl.run_cw(
        _A, 
        fixed_labels=y, 
        minweight=_minweight, 
        out=['fixed_labelsmap', 'labelvector'],
        **self.cw_params
      )
      
      # prepare output
      fixed_labelsmap = clustered['fixed_labelsmap'] if 'fixed_labelsmap' in clustered.keys() else {}
      fixed_labelsmap_rev = {v: k for k,v in fixed_labelsmap.items()}

      label_to_ids = clustered['labels']
      label_vector = clustered['labelvector']

      resolved_label_to_items = { str(fixed_labelsmap_rev.get(cid, cid)): ids for cid, ids in label_to_ids.items() }
      resolved_list_of_labels = [ str(fixed_labelsmap_rev.get(cid.item(), cid.item())) for cid in label_vector ]
      
      # RETURN 
      # a) dictionary of clusters, i.e. {label1: [id1, id3, id8, ....], label2: [id2, id4, ...], ...}
      # b) list of labels, same order as the input features, ie, [ label1, label2, label1, label2, label3, label4, ... ]
      # label names are resolved from input label names if any        
      return resolved_label_to_items, resolved_list_of_labels



# Identity map: every item into a single cluster
class Identity(DBClusterer):
  '''
  use in factory with default parameters:

    'clusterer': {
      'type': 'ident',
      'options': { }
    }

  '''
  name = 'ident'
  requires = 'instances'
  def fit_predict(self, X, y=None):
    ## forward one cluster per input sample
    label_vector_str = [str(i) if y is None or y[i] is None else y[i] for i in range(X.shape[0])]
    return self._get_label_to_items_from_label_vector(label_vector_str), label_vector_str
  

# Constant map: every item into the same cluster
class Constant(DBClusterer):
  '''
  use in factory with default parameters:

    'clusterer': {
      'type': 'const',
      'options': { }
    }

  '''
  name = 'const'
  requires = 'instances'
  def fit_predict(self, X, y=None):
    ## forward all unlabelled input samples in one cluster
    label_str = '0'
    if y is not None:
      assert label_str not in set(y)

    label_vector_str = [label_str if y is None or y[i] is None else y[i] for i in range(X.shape[0])]
    return self._get_label_to_items_from_label_vector(label_vector_str), label_vector_str


# Hierarchical Agglomerative Clustering
class HAClustering(DBClusterer, OrderedMultiClusterer):
  '''
  use in factory with default parameters:

    'clusterer': {
      'type': 'ha',
      'options': {
        'linkage': 'ward', # or 'average'
        'metric': 'euclidean', # or 'cosine'
        'n_clusters': 2,
        'compute_full_tree': 'auto',
        'distance_threshold': None,
        'compute_distances': False
      }
    }

  '''
  name = 'ha'
  requires = 'feature'

  #
  # according to [1], HA resolves ties by the ordering of elements, the outcome is thus deterministic, but is still random.
  # so a random state is not strictly necessary, but might be benficial. For pre-computed similarites, this is easy to resolve, 
  # but for feature matrices, a random theta would need to be added to the internal distances. Skipping random_state for HA.
  # 
  # [1] https://github.com/scikit-learn/scikit-learn/issues/7689#issuecomment-260533718
  #
  def __init__(self, n_clusters=2, metric='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
    import_required_on_init(import_sklearn)
    super().__init__()
    self.clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, memory=memory, connectivity=connectivity, compute_full_tree=compute_full_tree, linkage=linkage, distance_threshold=distance_threshold, compute_distances=compute_distances)


  def fit(self, X, y=None):
    ## fully unsupervised - y is ignored
    # X is a feature matrix
    if X.size(0) == 1:
      ## only one element, HA can't be fitted
      ## simply set clusterer.labels_ - any problems?
      self.clusterer.labels_ = [0]
      return self
    M = X
    if torch.is_tensor(M):
      M = X.numpy()
    self.clusterer.fit(M, y)
    return self

  def fit_predict(self, X, y=None):
    ## return labels and ids to make it compatible with our other clusterers
    self.fit(X=X, y=y)
    label_vector = self.clusterer.labels_
    label_vector_str = [str(k) for k in label_vector]
    return self._get_label_to_items_from_label_vector(label_vector_str), label_vector_str


  def get_requirement(self):
    return self.requires

  def get_clusterings(self, X, y=None, min_clusters=1):

    self.fit(X, y)
    n_samples = X.size(dim=0)

    # dict with cluster_id -> elements
    clusters = {i: {i} for i in range(n_samples)}
    label_vector = list(range(n_samples))

    ## iterate over children_ of clusterer and collect clusters:
    # children_ : array-like of shape (n_samples-1, 2)
    # at the i-th iteration, children[i][0] and children[i][1] are merged to form node `n_samples + i`
    for i, merged_clusters in enumerate(self.clusterer.children_):

      n_clusters = n_samples - i - 1
      ## stop if number of clusters is smaller than the given number of clusters or 2
      if n_clusters < max(min_clusters, 2):
        break

      clusters[n_samples + i] = clusters[merged_clusters[0]] | clusters[merged_clusters[1]]
      clusters.pop(merged_clusters[0])
      clusters.pop(merged_clusters[1])

      for element in clusters[n_samples + i]:
        label_vector[element] = n_samples + i

      assert len(set(label_vector)) == n_clusters

      yield (
        {str(cluster_label): list(elements) for cluster_label, elements in clusters.items()},
        label_vector.copy()
      )


# forward labels from another clustering
class ForwardLabels(DBClusterer):
  '''
  use in factory with default parameters:

    'clusterer': {
      'type': 'forward',
      'options': {
        'source': identifier@id
      }
    }

  '''
  name = 'forward'
  requires = 'clustering'

  def __init__(self, source=None, source_config: dict | None = None):

    ## at least one of source or source_config must be given
    assert source is not None or source_config is not None

    super().__init__()
    self.source = source
    self.source_config = source_config
  
  def fit_predict(self, X, y=None):
    '''
    X is the list of cluster labels which will be forwarded
    '''
    # simply forward y as output predictions
    label_vector_str = [str(k) for k in X]
    return self._get_label_to_items_from_label_vector(label_vector_str), label_vector_str


# Use some externalized information
class ConllFilePseudoClusterer(DBClusterer):
  '''

  echo "select * from labels_dataset_all((select id from datasets where name='fn1.7'))" | docker exec -i german-frame-clustering-paradedb-1 psql postgresql://root:root@pdb/gfncdata -tAF$'\t'

  echo "select label, numinstances from labels_dataset_all((select id from datasets where name='fn1.7'))" | docker exec -i german-frame-clustering-paradedb-1 psql postgresql://root:root@pdb/gfncdata -tAF$'\t' | perl -lanE '$edt = $F[0]; $edt =~ s/^.*:://g; print "$edt","\t","$_"' > fn1.7_labelmap__conll_to_db.tsv
  
  echo "select label, numinstances from labels_dataset_all((select id from datasets where name='fn1.7'))" | docker exec -i german-frame-clustering-paradedb-1 psql postgresql://reader:reads@ltdemos.informatik.uni-hamburg.de:8099/gfncdata -tAF$'\t' | perl -lanE '$edt = $F[0]; $edt =~ s/^.*:://g; print "$edt","\t","$_"' > fn1.7_labelmap__conll_to_db_all.tsv

  use in factory with, e.g.:

    'clusterer': {
      'type': 'conll',
      'options': {
        'fname_conll': './file-name.conll' | './file-name.conll.gz',
        'conll_fields': None | 15 | ['id', 'form', '_u1', 'lemma', 'upos', 'xpos', 'docid',  '_u2', '_u3', '_u4', '_u5', '_u6', 'lu', 'frame', 'B-I-O'],
        'default_label': 'unk',
        'fname_labelmap': None | './file-name.tsv.gz' | './file-name.tsv'
        'save_conll_df_to_file': None | './file-name.tsv.gz'
      }
    }

  '''

  name = 'conll'
  requires = 'instances'

  def get_data_rows(fname, fields):
    import_required_on_init(import_conll)
    is_byte_pattern = re.compile(r'^b([\'"])(.*)(\1)$')
    readfun = gzopen if fname.endswith('.gz') else open
    with readfun(fname, mode='rt', encoding='utf-8') as fh:
      for di, cdoc in enumerate(parse_incr(fh, fields=fields)): # 'ID FORM LEMMA PLEMMA POS PPOS FEAT PFEAT HEAD PHEAD DEPREL PDEPREL FILLPRED_LU PRED_FRAME APREDS'.lower().split()
        tokens, lemmas, annotations = zip(*[(literal_eval(t['form']).decode('utf-8') if is_byte_pattern.fullmatch(t['form']) is not None else t['form'], t['lemma'], t['b-i-o']) for t in cdoc])
        lu_idx = [i for i, t in enumerate(cdoc) if t['frame'] != '_']
        lu_idx_str = '_'.join(map(str, lu_idx))
        unk_idx = [i for i, t in enumerate(cdoc) if t['form'].upper() == 'UNK']
        lu_lemma = [lemmas[i] for i in lu_idx]
        lu_fn = cdoc[lu_idx[0]]['lu']
        frame_label = cdoc[lu_idx[0]]['frame']
        hashid = get_obj_hash([
          list(map(str.lower, tokens)), 
          lu_idx_str,
        ])
        yield {
          'hashid': hashid,
          'i': di,
          'tokens': tokens,
          'lemmas': lemmas,
          'annotations': annotations,
          'lu_idx': lu_idx,
          'lu_idx_str': lu_idx_str,
          'unk_idx': unk_idx,
          'lu_lemma': lu_lemma,
          'lu': lu_fn,
          'frame_label': frame_label,
          'matches': []
        }

  def read_conll_as_dataframe(fname, fields):
    dataframe = pd.DataFrame.from_dict(ConllFilePseudoClusterer.get_data_rows(fname=fname, fields=fields), orient='columns')
    dataframe = dataframe.set_index('hashid')
    return dataframe

  def read_labelmap_as_dataframe(fname):
    dataframe = pd.read_csv(
      fname,
      sep='\t',
      quoting=csv.QUOTE_MINIMAL,
      header=None,
      names=['conll', 'db', 'dataset_support'],
      skip_blank_lines=True,
      encoding='utf-8'
    )
    dataframe = dataframe.set_index('conll')
    return dataframe

  def __init__(self, fname_conll, conll_fields, fname_labelmap, default_label='<unk>', save_conll_df_to_file=None):
    import_required_on_init(import_conll)
    super().__init__()
    self.conll_filename_abs_path = str(Path(fname_conll).expanduser())
    self.save_conll_df_to_file_overwrite = False
    self.save_conll_df_to_file = None
    if save_conll_df_to_file is not None:
      if save_conll_df_to_file.endswith(':o'):
        self.save_conll_df_to_file_overwrite = True
        self.save_conll_df_to_file = str(Path(save_conll_df_to_file[:-2]).expanduser())
      else:
        self.save_conll_df_to_file = str(Path(save_conll_df_to_file).expanduser())
    # check if file already exists
    if self.save_conll_df_to_file is not None and os.path.exists(self.save_conll_df_to_file) and not self.save_conll_df_to_file_overwrite:
      raise FileExistsError(self.save_conll_df_to_file)
    self.fields = conll_fields
    self.completed_fields = [f'f{i}' for i in range(self.fields)] if isinstance(self.fields, int) else list(map(str.lower, self.fields))
    self.labelmap_filename = fname_labelmap
    self.conll_df = None
    self.labeldf_conll_db = None
    self.default_label = default_label
    self.no_match_found_collection = []
    self.no_label_match_found_collection = {}
    self.multi_data_instance_matches = []

  def prepare_dataframe(self):
    try:
      dataframe = ConllFilePseudoClusterer.read_conll_as_dataframe(
        fname=self.conll_filename_abs_path, 
        fields=self.completed_fields)
    except Exception as e:
      self.logger.error(e)
      raise e
    return dataframe

  def prepare_labeldf(self):
    try:
      dataframe = ConllFilePseudoClusterer.read_labelmap_as_dataframe(self.labelmap_filename)
    except Exception as e:
      self.logger.error(e)
      raise e
    return dataframe

  def find_match_for_row(self, row):
    lu = row.extrainfo['LU']
    lu_idx = row.extrainfo['LU_INDEX'] + row.extrainfo['LU_INDEX_PART']
    luidx_str = '_'.join(map(str, lu_idx))
    tokens = row.extrainfo['tokens']
    hashid_query = get_obj_hash([
      list(map(str.lower, tokens)), 
      luidx_str,
    ])
    try:
      res = self.conll_df.loc[hashid_query]
      if len(res.shape) == 1:
        assert res.lu == lu
      else:
        uniq_lu = res.lu.unique()
        assert uniq_lu[0] == lu and len(uniq_lu) == 1
      # found an easy match, note: this might still be more than one row
      return res
    except KeyError as e:
      pass
    # try to find a more complicated match which contains UNK tokens
    # first get a subset where the lu and the lu index matches
    subdf = self.conll_df[self.conll_df.lu == lu]
    subdf = subdf[subdf.lu_idx_str == luidx_str]
    # consider 3 cases
    # 1) the result contains exactly one match -> trivial: it's a match
    if subdf.shape[0] == 1:
      return subdf.iloc[0]
    # 2) the result contains multiple rows. Iterate over the rows, replace UNK tokens and try to find a match
    for idx_hashid, crow in subdf.iterrows():
      tokens_with_unks = tokens.copy()
      for unk_i in crow.unk_idx:
        # skip if unk_index is not within the sentence boundaries
        if unk_i < 0 or unk_i >= len(tokens_with_unks):
          continue
        tokens_with_unks[unk_i] = 'UNK'
      unk_hashid_query = get_obj_hash([
        list(map(str.lower, tokens_with_unks)), 
        luidx_str,
      ])
      if unk_hashid_query == idx_hashid: # we found our match
        return crow
    # 3) the result is empty or we didn't find a match in 2) -> we certainly don't have a match
    return None

  def find_label_match(self, conll_label_candidate, data_instance_row):
    try:
      db_label = self.labeldf_conll_db.loc[conll_label_candidate].db
    except KeyError as e:
      self.logger.warning(f"did not find a DB label match for CONLL label '{conll_label_candidate}'.")
      no_label_match_ids = self.no_label_match_found_collection[conll_label_candidate] = self.no_label_match_found_collection.get(conll_label_candidate, list())
      no_label_match_ids.append(data_instance_row['instance_id'])
      db_label = f'conll::{conll_label_candidate}'
    return db_label

  def fit_predict(self, X, y=None):
    # lazy load
    if self.conll_df is None:
      self.conll_df = self.prepare_dataframe()
    if self.labeldf_conll_db is None:
      self.labeldf_conll_db = self.prepare_labeldf()
    # X is a dataframe with instance information
    # y contains train labels or none, prepare y_pred by copying y's labels
    label_vector_str = y.copy() if y is not None else [ None ]*X.shape[0]
    # TASK: match dataframe X with conll dataframe
    # for each instance in X:
    #   assign label from y that are known, i.e. train instances --> trivial: skip
    #   for other test instances: try to find a match
    for (i, row), y_pred__i in zip(X.iterrows(), label_vector_str):
      if y_pred__i is not None:
        continue # skip this row because we already have a label in y_pred
      matched_conll_row = self.find_match_for_row(row)
      if matched_conll_row is None:
        self.logger.warning(f'Did not find a matching instance. Assigning default label {self.default_label}.')
        label_vector_str[i] = self.default_label
        self.no_match_found_collection.append(row['instance_id'])
        continue
      # else, double check that we have really only one row, otherwise there might be duplicates!
      if len(matched_conll_row.shape) > 1 and matched_conll_row.shape[0] > 0:
        # attention: a dragon has been found, we have duplicates! -> make sure we have only one unique label and log that shit
        unique_labels = matched_conll_row.frame_label.unique()
        assert len(unique_labels) == 1 # <- this make sure that we don't have ambiguous matches 
        conll_label = unique_labels[0]
        self.logger.warning(f'Found duplicate conll matches {matched_conll_row.i.tolist()} for instance {row.instance_id} ')
        # keep track of what has been matched to what
        self.multi_data_instance_matches.append((row.instance_id, matched_conll_row.index[0], matched_conll_row.shape[0], matched_conll_row.i.tolist()))
        for _, matched_conll_row_ in matched_conll_row.iterrows():
          if len(matched_conll_row_.matches) > 0:
            self.logger.warning(f'Conll row {matched_conll_row_.i} has been matched to at least one other instance before.')
          matched_conll_row_.matches.append(row['instance_id'])
      else:
        conll_label = matched_conll_row.frame_label
        # keep track of what has been matched to what
        if len(matched_conll_row.matches) > 0:
          self.logger.warning(f'Conll row {matched_conll_row.i} has been matched to at least one other instance before.')
        matched_conll_row.matches.append(row['instance_id'])
      # assign the label
      label_vector_str[i] = self.find_label_match(conll_label, row)

    # last step: create a label dictionary: resolve label_vector_str so that out matches dict, list
    return self._get_label_to_items_from_label_vector(label_vector_str), label_vector_str
  
  def finalize(self):
    # log some more info, e.g. the number of unmatched conll instances
    self.conll_df['num_matches'] = self.conll_df.matches.apply(len)
    df_matched_multiple = self.conll_df[self.conll_df.num_matches > 1]
    df_unmatched = self.conll_df[self.conll_df.num_matches == 0]
    num_matched_multiple = df_matched_multiple.shape[0]
    num_matched_multiple_total = df_matched_multiple.num_matches.sum() - num_matched_multiple
    num_unmatched = df_unmatched.shape[0]
    num_nomatch = len(self.no_match_found_collection)
    num_nomatch_label = len(self.no_label_match_found_collection)
    num_nomatch_label_total = sum(map(len, self.no_label_match_found_collection.values()))
    nomatch_label_df = pd.DataFrame.from_records(list(self.no_label_match_found_collection.items()), columns=['label', 'instances'])
    df_multi_instance_matches = pd.DataFrame.from_records(self.multi_data_instance_matches, columns=['instance_id', 'hash', 'num_matches', 'conll_i'])
    num_multi_instance_matches = df_multi_instance_matches.shape[0]
    num_multi_instance_matches_total = df_multi_instance_matches.num_matches.sum()
    # log nomatch, unmatched and multimatched instances
    self.logger.info(f'''{num_unmatched} conll instances not matched to any data instance: \n===\n{df_unmatched[['i', 'lu', 'lu_idx_str', 'matches' ]].to_string(index=False)}\n===\n''')
    self.logger.info(f'''{num_matched_multiple} conll instances matched to more that one data instance for {num_matched_multiple_total} times: \n===\n{df_matched_multiple[['i', 'lu', 'lu_idx_str', 'matches' ]].to_string(index=False)}\n===\n''')
    self.logger.info(f'''{num_multi_instance_matches} data instances matched to more that one conll instance for {num_multi_instance_matches_total} times: \n===\n{df_multi_instance_matches.to_string(index=False)}\n===\n''')
    self.logger.info(f'''{num_nomatch} data instances did not match any conll instance: \n===\ninstance_id\n{'\n'.join(map(str, self.no_match_found_collection))}\n===\n''')
    self.logger.info(f'''{num_nomatch_label} conll labels did not match any known DB label for {num_nomatch_label_total} times: \n===\n{nomatch_label_df.to_string(index=False)}\n===\n''')
    if self.save_conll_df_to_file is not None and not os.path.exists(self.save_conll_df_to_file):
      self.conll_df['sent'] = self.conll_df.tokens.apply(' '.join)
      self.conll_df[['num_matches', 'matches', 'i', 'lu', 'lu_lemma', 'lu_idx_str', 'frame_label', 'sent']].to_csv(
        self.save_conll_df_to_file,
        header=True,
        sep='\t',
        index=True,
        quoting=csv.QUOTE_MINIMAL,
        encoding = 'utf-8')


# Hierarchical Agglomerative Clustering
class ConstrainedHAClustering(DBClusterer):
  '''
  HAClustering with must-link and cannot-link constraints extracted from given labels.
  The constraints are enforced by altering the distance matrix and using complete linkage:
  - nodes that must link get a distance of 0
  - nodes that cannot link get a distance of 3
  - all other distances are scaled between 1 and 2

  use in factory with default parameters:

    'clusterer': {
      'type': 'ha-constrained',
      'options': {
        # number of clusters that is expected for new classes
        # total number of clusters is n_labels + n_clusters
        # must be lower than the number of unlabelled instances
        'n_extra_clusters': 2,
        ### or:
        'distance_threshold': 0.8,
        'random_state': None,
        ha_kwargs: {
          'memory': None
        }
      }
    }

  '''
  
  name = 'ha-constrained'
  
  requires = 'similarity'

  __ha_default_kwargs__ = {'memory': None}

  def __init__(self, n_extra_clusters=None, distance_threshold=None, random_state=None, ha_kwargs={}):
    import_required_on_init(import_sklearn)
    super().__init__()
    self.n_extra_clusters = n_extra_clusters
    self.ha_kwargs = ConstrainedHAClustering.__ha_default_kwargs__ | ha_kwargs
    self.distance_threshold = distance_threshold
    self.random_state = random_state
    self.random_generator = torch.Generator()
    self.random_generator.manual_seed(0 if random_state == None else random_state)
    # distance threshold needs to be in [0,+inf[
    if self.distance_threshold is not None:
      assert 0 < self.distance_threshold < 3
    

  def plot_dendrogram(model, **kwargs):
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
      current_count = 0
      for child_idx in merge:
        if child_idx < n_samples:
          current_count += 1  # leaf node
        else:
          current_count += counts[child_idx - n_samples]
      counts[i] = current_count

    linkage_matrix = np.column_stack(
      [model.children_, model.distances_, counts]
    ).astype(float)

    plt.title("Hierarchical Clustering Dendrogram")
    # Plot the corresponding dendrogram
    # plot the top three levels of the dendrogram
    dendrogram(linkage_matrix, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


  def fit_predict(self, X, y):
    # y contains labels
    assert y is not None and len(y) == X.shape[0] and X.shape[0] == X.shape[1]
    # get some information of the labeled and unlabeled examples
    n_samples = len(y)
    n_unlabelled = y.count(None)
    n_labelled = n_samples - n_unlabelled
    n_classes = len(set(y))-1 # -1 for the None object
    self.logger.debug(f'#:{n_samples}; #labelled:{n_labelled}; #classes:{n_classes}')
    # for some trivial cases, clustering is not necessary:
    #   1. only labelled samples -> return the input vector
    #   2. only one sample -> return a single cluster (if the example was labelled, it was already covered by trivial case 1.)
    if n_unlabelled == 0: # 1.
      return self._get_label_to_items_from_label_vector(y), y
    if n_samples == 1: # 2.
      return {'0': [0]}, ['0']
    # make sure that self.n_clusters is in [0, #unlabelled examples]
    if self.n_extra_clusters is not None:
      assert 0 <= self.n_extra_clusters <= n_unlabelled
    # make sure that similarities are in [0,1]
    assert X.min() >= 0 and X.max() <= 1
    # convert similarity matrix X to distance matrix D
    D = 1-X
    # add some minimal random noise, so that ties are broken randomly
    D = D+(torch.randn(D.shape, generator=self.random_generator)*1e-4)
    # scale within [0,1]
    D = D-D.min()
    D = D/D.max()
    # make sure that distances are still in [0,1]
    assert D.min() >= 0 and D.max() <= 1

    # apply constraints to the distance matrix:
    # - nodes that must link get a distance of 0
    # - nodes that cannot link get a distance of 3
    # - all other distances are scaled between 1 and 2
    D = D+1
    for i, j in it_product(range(X.shape[0]), range(X.shape[0])):
      if i == j: # TODO: add self loops or not? 
        D[i,j] = 0 # add self loop
        continue
      if (y[i] is None) or (y[j] is None):
        continue
      if y[i] == y[j]: # same class, let the distance be the lowest possible value
        D[i,j] = 0
      else: # different class, let the distance be maximal
        D[i,j] = 3 # max_distance+1 # 1 # float('inf') # 1 or maximum distance
    
    if self.n_extra_clusters is not None:
      n_target_clusters = n_classes + self.n_extra_clusters
      if n_target_clusters == 0: # this might happen if we have only unlabelled examples and n_extra_clusters was set to 0, imitating a classification scenario
        n_target_clusters = 1 # simply set to 1
    else:
      n_target_clusters = None
    
    skl_clusterer = AgglomerativeClustering(
      metric='precomputed',
      compute_distances=True,
      n_clusters=n_target_clusters,
      distance_threshold=self.distance_threshold, 
      compute_full_tree=True,
      linkage='complete',
      **self.ha_kwargs
    )
    
    label_vector = skl_clusterer.fit_predict(D)
    # ConstrainedHAClustering.plot_dendrogram(skl_clusterer)

    ## check that constraints are not violated
    self._assert_constraints(label_vector, y)

    ## get labels for clusters with labelled instances
    label_vector_str = self._get_labels_for_clustering(label_vector, y)
    ## return labels and ids to make it compatible with our other clusterers
    items = self._get_label_to_items_from_label_vector(label_vector_str)
    return items, label_vector_str


# XMeans using pyclustering
class PyclusteringXMeans(DBClusterer):
  '''
  use in factory with default parameters:

    'clusterer': {
      'type': 'pycl-xmeans',
      'options': {
        'amount_initial_centers': 2,
        'max_clusters': 1e10
      }
    }

  '''
  name = 'pycl-xmeans'
  requires = 'feature'

  def __init__(self, amount_initial_centers=2, max_clusters=1e10, random_state=None):
    import_required_on_init(import_pyclustering_xmeans)
    super().__init__()
    self.amount_initial_centers = amount_initial_centers
    self.max_clusters = max_clusters
    self.random_state = random_state

  def fit_predict(self, X, y=None):
    ## fully unsupervised - y is ignored
    # X is a feature matrix
    if X.size(0) == 1:
      ## only one element, no need for clustering
      return { '0': [ 0 ] }, ['0']
    M = X
    if torch.is_tensor(X):
      M = X.numpy()
    initial_centers = kmeans_plusplus_initializer(M, self.amount_initial_centers, random_state=self.random_state).initialize()
    # set maximum to the number of instances to cluster if larger than this
    _max_clusters = int(np.min((self.max_clusters, M.shape[0])))
    xmeans_instance = xmeans(M, initial_centers, _max_clusters, random_state=self.random_state)
    xmeans_instance.process()
    ## list of the clusters, each cluster is represented by the list of its instances (int)
    clusters = xmeans_instance.get_clusters()
    # resolve so that out matches dict, list
    label_to_items = {}
    label_vector_str = [None]*X.size(0)
    for i, items in enumerate(clusters):
      label_to_items[f'{i}'] = items
      for item in items:
        label_vector_str[item] = f'{i}'
    # finally return results
    return label_to_items, label_vector_str


class SoftmaxClassification(DBClusterer):

  name = 'softmax'
  requires = 'feature'

  def __init__(self, cls_args: dict=None, standardization=False, random_state=None):
    import_required_on_init(import_sklearn)
    super().__init__()
    self.cls_args = {}
    if cls_args is not None:
      self.cls_args = cls_args.copy()

    # from sklearn-docs:
    # Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data
    self.cls_args['random_state'] = random_state

    self.standardization = standardization

  def fit_predict(self, X, y=None):

    ## does not work unsupervised
    assert y is not None

    ## separate labelled and unlabelled instances
    y_unlabelled_indices = [i for i in range(len(y)) if y[i] is None]
    y_labelled = [label for label in y if label is not None]

    if not y_unlabelled_indices:
      # everything is labelled - return ground truth
      return self._get_label_to_items_from_label_vector(y), y

    ## labelled data is needed to train the classifier
    assert y_labelled

    if torch.is_tensor(X):
      X = X.numpy()
    X_labelled = np.delete(X,y_unlabelled_indices, axis=0)
    X_unlabelled = X[y_unlabelled_indices,]

    if self.standardization:
      scaler = sklearn.preprocessing.StandardScaler().fit(X_labelled)
      X_labelled = scaler.transform(X_labelled)
      X_unlabelled = scaler.transform(X_unlabelled)

    ## train and apply classifier
    classifier = sklearn.linear_model.LogisticRegression(**self.cls_args)
    classifier.fit(X_labelled, y_labelled)
    y_prediction = classifier.predict(X_unlabelled)

    ## merge predictions with y
    label_vector = y.copy()
    for i, predicted_label in zip(y_unlabelled_indices, y_prediction):
      label_vector[i] = predicted_label

    return self._get_label_to_items_from_label_vector(label_vector), label_vector


class DBClustererUnsupervised(DBClusterer):

  name = 'unsupervised'

  def __init__(self, clusterer: DBClusterer, discard_labelled_instances=False):
    super().__init__()
    self.clusterer = clusterer
    self.requires = clusterer.requires
    self.discard_labelled_instances = discard_labelled_instances

  def fit_predict(self, X, y=None):

    if self.discard_labelled_instances:
      if y is None:
        y = [None]*X.size(0)

      y_unlabelled_indices = [i for i in range(len(y)) if y[i] is None]

      X_unlabelled = X[y_unlabelled_indices,]

      _, y_prediction = self.clusterer.fit_predict(X_unlabelled, y=None)

      label_vector = y.copy()
      for i, predicted_label in zip(y_unlabelled_indices, y_prediction):
        label_vector[i] = predicted_label

      return self._get_label_to_items_from_label_vector(label_vector), label_vector

    else:
      return self.clusterer.fit_predict(X, y=None)


class ClustererList(MultiClusterer):

  name='clusterer-list'

  def __init__(self, clusterers: typing.Sequence[DBClusterer]):

    import_required_on_init(import_sklearn)

    self.clusterers = clusterers

  def setLogger(self, logger):
    super().setLogger(logger)
    for clusterer in self.clusterers:
      clusterer.setLogger(logger)


  def get_requirement(self):

    return 'similarity' if all(
      [clusterer.requires == 'similarity' for clusterer in self.clusterers]
    ) else 'feature'

  def get_clusterings(self, X, y=None, min_clusters=None):
    ## min_clusters is ignored

    if self.get_requirement == 'similarity':
      ## all clusterers need similarity
      X_sim = X
    else:
      X_sim = sklearn.metrics.pairwise.cosine_similarity(X)

    for clusterer in self.clusterers:
      if clusterer.requires == 'similarity':
        yield clusterer.fit_predict(X_sim, y)
      elif clusterer.requires == 'feature':
        yield clusterer.fit_predict(X, y)
      else:
        raise ValueErrror('Only similarity or feature is allowed')


class Silhouette(DBClusterer):

  name = 'silhouette'
  requires = 'feature'

  def __init__(self, clusterings: MultiClusterer, metric='euclidean'):

    import_required_on_init(import_sklearn)
    super().__init__()

    self.clusterings = clusterings
    self.metric = metric

  def setLogger(self, logger):
    super().setLogger(logger)
    self.clusterings.setLogger(logger)

  def fit_predict(self, X, y=None):

    n_samples = X.size(dim=0)

    best_clustering = {
      'label_vector': list(range(n_samples)),
      'label_to_items': {str(i): [i] for i in range(n_samples)}
    }
    best_silhouette_score = -1

    # get distances
    if callable(self.metric):
      X_dist = self.metric(X)
    elif self.metric == 'precomputed':
      self.logger.warn('Precomputed distances are not supported for silhouette score yet. Using cosine distance.')
      X_dist = sklearn.metrics.pairwise.cosine_distances(X)
    else:
      X_dist = sklearn.metrics.pairwise.distance_metrics()[
        self.metric
      ](X)

    multiclusterer_requires = self.clusterings.get_requirement()
    if multiclusterer_requires == 'feature':
      X_cluster = X
    elif multiclusterer_requires == 'similarity':
      X_cluster = X_dist
    else:
      raise ValueError(f'Clustering needs to require either "feature" or "similarity", not "{multiclusterer_requires}"')

    min_clusters = len(set(y))
    for label_to_items, label_vector in self.clusterings.get_clusterings(X_cluster, y, min_clusters=min_clusters):

      self.logger.info(f'Evaluating {len(label_to_items.keys())} clusters with silhouette score')

      curr_silhouette_score = sklearn.metrics.silhouette_score(X_dist, label_vector, metric='precomputed')

      ## upate best clustering
      if curr_silhouette_score > best_silhouette_score:
        best_clustering['label_vector'] = label_vector
        best_clustering['label_to_items'] = label_to_items
        best_silhouette_score = curr_silhouette_score
        self.logger.info(f'New best silhouette: {best_silhouette_score}')

      ## stop if optimal silhoutte score is reached
      if best_silhouette_score == 1:
        break

    return best_clustering['label_to_items'], best_clustering['label_vector']


# Yamada stopping criterion
## From Yamada: 
#  Specifically, the clustering is terminated when the ratio of pLU pairs
# belonging to the same cluster pF1=F2 is greater than
# or equal to the ratio of LU pairs belonging to the
# same frame in the development set pC1=C2 . Here,
# pF1=F2 is calculated as:
# pF1=F2 = # of pLU pairs in the same cluster
# # of all pLU pairs . (2
class PairsInClusterRatio(DBClusterer):

  name = 'pairs-in-cluster'
  requires = 'feature'

  def __init__(self, clusterings: OrderedMultiClusterer, stopping_ratio=0.5):

    super().__init__()

    self.clusterings = clusterings
    self.stopping_ratio = stopping_ratio
    assert self.clusterings.get_requirement() == self.requires

  def setLogger(self, logger):
    super().setLogger(logger)
    self.clusterings.setLogger(logger)


  def fit_predict(self, X, y=None):

    n_samples = X.size(dim=0)
    pairs_in_same_cluster = (n_samples * (n_samples-1))/2 * self.stopping_ratio

    for label_to_items, label_vector in self.clusterings.get_clusterings(X, y):

      if sum([(len(cluster)*(len(cluster)-1))/2 for cluster in label_to_items.values()]) >= pairs_in_same_cluster:
        return label_to_items, label_vector

