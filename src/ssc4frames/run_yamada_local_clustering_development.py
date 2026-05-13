import os
from copy import deepcopy

import sqlalchemy as sa

from ssc4frames.database import DBHandler, DatasetSplit, Clustering
from ssc4frames.run_experiment_db_only import get_or_create_db_clustering, run_local_clustering_experiment, merge_with_default_params, run_with_params, merge_params, default_parameters, setup_logger_for_clustering
from ssc4frames.helpers import get_dburl_from_env
from ssc4frames.factoryhelper import create_factory

import ssc4frames.loghelper as loghelper

__base_logger_name__ = os.path.basename(__file__)
base_logger = loghelper.setup_logger(__base_logger_name__)


# group average clustering based on a Euclidean distance
def get_local_clustering_parameters(distance_threshold, emmodel, dim, local_alpha):
    return merge_params(deepcopy(default_parameters['local']), {
        'emmodel': emmodel,
        'dim': dim,
        'alpha': local_alpha,
        'emaggregation': 'avg',
        'clusterer': {
            'type': 'ha',
            'options': {
                'metric': 'euclidean',
                'linkage': 'average',
                'n_clusters': None,
                'distance_threshold': distance_threshold
            }
        }
    })

def get_local_xmeans_parameters(max_clusters, emmodel, dim, local_alpha):
    return merge_params(deepcopy(default_parameters['local']), {
        'emmodel': emmodel,
        'dim': dim,
        'alpha': local_alpha,
        'emaggregation': 'avg',
        'clusterer': {
            'type': 'pycl-xmeans',
            'options': {
                'amount_initial_centers': 2, ## minimal amount of clusters
                'max_clusters': max_clusters ## maximal amount of clusters
            }
        }
    })

def test_local(dbh, dataparams, localparams, reuse_local=True, meta_note=None, save_test_only=False):


  # define local identifier
  localidentifier_settings = {'data': dataparams, 'local': localparams}
  localidentifier = Clustering.get_identifier_from_settings(localidentifier_settings)
  # setup clusterer factory
  factory = create_factory()

  local_clustering_obj = get_or_create_db_clustering(dbh=dbh, clusteringtype='local', identifier=localidentifier, settings=localidentifier_settings, get_if_exist=reuse_local, note=meta_note)
  logger_clustering_local = setup_logger_for_clustering(dbh, local_clustering_obj.id)
  if local_clustering_obj.numclusters <= 0 and not local_clustering_obj.success:
    logger_clustering_local.info(f'Local clustering {local_clustering_obj.id} ({localidentifier}) initialized.')
    # instantiate clusterer via factory
    local_clusterer = factory.create_from_name('dbclusterer', localparams['clusterer'])
    local_clusterer.setLogger(logger_clustering_local)
    run_local_clustering_experiment(dbh, factory, dataparams, local_clustering_obj.id, localparams['emmodel'], localparams['dim'], float(localparams['alpha']), localparams['emaggregation'], localparams.get('filter',{}), local_clusterer, device, save_test_only, logger_clustering_local)
    logger_clustering_local.info(f'Local clustering {local_clustering_obj.id} ({localidentifier}) finished.')

  return local_clustering_obj


def get_development_stats(dbh, data):

  with dbh.sessionmaker() as session:

    datasetsplit_obj: DatasetSplit = session.execute(sa.select(DatasetSplit).where(DatasetSplit.name == data['dataset'])).scalar()
    datasetsplit_df = datasetsplit_obj.get_instance_df()
    train_devel_data = datasetsplit_df[datasetsplit_df.split.isin(data['splits'])]

    return (
        # number of instances
        train_devel_data.shape[0],
        # number of lus
        train_devel_data.groupby(['lu_lemma', 'frame_label']).ngroups,
        # number of lu_pairs in same frame
        sum(train_devel_data.groupby(['frame_label'])['lu_lemma'].nunique().map(lambda n: (n*(n-1))/2))
    )


if __name__ == '__main__':

  import argparse
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('distance_threshold', type=float)
  arg_parser.add_argument('dataset', nargs='?', default='fn1.7-default')
  arg_parser.add_argument('--device', default='cpu')
  arg_parser.add_argument('--run_global', action='store_true')
  args = arg_parser.parse_args()

  dburl = get_dburl_from_env()
  data = {
    'dataset': args.dataset,
    'splits': ['train', 'dev'],
    'testsplits': ['dev']
  }
  embeddings_for_dataset = {
     'fn1.7-default': {
        'modelname': 'bert-base-uncased',
        'dim': 768
     },
     'salsa-default': {
        'modelname': 'bert-base-german-cased',
        'dim': 768
     },
  }
  emmodel = embeddings_for_dataset[args.dataset]['modelname']
  dim = embeddings_for_dataset[args.dataset]['dim']
  device = args.device

  local_alpha = 0 # use only masked

  print(f"Using dataset '{args.dataset}' with embeddings model '{emmodel}'")

  dbh: DBHandler = DBHandler(dburl)

  number_of_instances, number_of_lus, number_of_lu_pairs_in_same_frame = get_development_stats(dbh, data)

  number_of_lu_pairs = (number_of_lus*(number_of_lus - 1))/2
  lus_in_same_frame_ratio = number_of_lu_pairs_in_same_frame/number_of_lu_pairs

  local_clusterings = []

  # run X-Means for local clustering
  clustering_obj = test_local(dbh, data, get_local_xmeans_parameters(number_of_instances, emmodel, dim, local_alpha))
  with dbh.sessionmaker() as session:
      clustering_obj = session.get(Clustering, clustering_obj.id)
      local_clusterings.append(f'{clustering_obj.identifier}@{clustering_obj.id}')

  ## run Agglomerative clustering for local clustering
  clustering_obj = test_local(dbh, data, get_local_clustering_parameters(args.distance_threshold, emmodel, dim, local_alpha))
  with dbh.sessionmaker() as session:
      clustering_obj = session.get(Clustering, clustering_obj.id)
      local_clusterings.append(f'{clustering_obj.identifier}@{clustering_obj.id}')

      print(f'Distance threshold for local agglomerative clustering: {args.distance_threshold}')
      print(f'(Generated clusters: {str(clustering_obj.numclusters)} / Actual clusters: {str(number_of_lus)})')
      assert clustering_obj.numclusters == number_of_lus

      print(f'Stopping ratio for global clustering: {lus_in_same_frame_ratio}')

  if args.run_global:
    ## run average linking and ward for global clustering with different alphas and local clusterings
    for local_clustering in local_clusterings:
        for alpha in [x/10.0 for x in range(11)]:
            for linkage in ['average', 'ward']:

                params_override = {
                    'meta': {
                        'database': dburl,
                        'note': 'Yamda-Development test run'
                    },
                    'data': data,
                    'local': {
                        'emmodel': emmodel,
                        'dim': dim,
                        'alpha': alpha,
                        'emaggregation': 'avg',
                        'clusterer': {
                            'type': 'forward',
                            'options': {
                                'source': local_clustering
                            }
                        }
                    },
                    'global': {
                        'clusterer': {
                            'type': 'ha-stopping',
                            'options': {
                                'stopping_ratio': lus_in_same_frame_ratio,
                                'cluster_options': {
                                    'metric': 'euclidean',
                                    'linkage': linkage
                                }
                            }
                        }
                    }
                }

                params = merge_with_default_params(params_override)
                run_with_params(params, await_key_confirmation=False)

