from copy import deepcopy
import json

import numpy as np
import torch

import sqlalchemy as sa
import pgvector.sqlalchemy as sapgvec

from ssc4frames.database import DBHandler
from ssc4frames.factoryhelper import create_factory
from ssc4frames.helpers import get_dburl_from_env
from ssc4frames.run_yamada_local_clustering_development import get_development_stats
from ssc4frames.run_experiment_db_only import merge_params, default_parameters

data = {
  'dataset': 'fn1.7-default',
  'splits': ['train', 'dev'],
  'testsplits': ['dev']
}

local_filter = {
  'min_lemmainstances': 1,
  'max_lemmainstances': 1e10,
  'limit_lemmainstances': 1e10,
  'randomize_order': True,
  'random_seed': 0.946684799,
}
device = 'cpu'

def main():

  import argparse
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('dataset', nargs='?', default=data['dataset'])
  arg_parser.add_argument("--dev_file_ha")
  arg_parser.add_argument("--dev_file_xmeans")
  args = arg_parser.parse_args()

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

  local_clustering_params = merge_params(deepcopy(default_parameters['local']), {
    'emmodel': emmodel,
    'dim': dim,
    'alpha': 0, # use only masked embeddings
    'emaggregation': 'avg',
    'clusterer': {
      'type': 'ha',
      'options': {
        'metric': 'euclidean',
        'linkage': 'average',
        'n_clusters': 1,
        'distance_threshold': None,
        'compute_full_tree': True,
        'compute_distances': True
      }
    }
  })


  dbh: DBHandler = DBHandler(get_dburl_from_env())

  number_of_instances, number_of_lus, number_of_lu_pairs_in_same_frame = get_development_stats(dbh, {**data, **{'dataset': args.dataset}})

  number_of_lu_pairs = (number_of_lus*(number_of_lus - 1))/2
  lus_in_same_frame_ratio = number_of_lu_pairs_in_same_frame/number_of_lu_pairs


  ## from run_experiment_db_only::run_local_clustering_experiment
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

  ## from run_experiment_db_only::fmtstr__create_vectorized_split_instances_view
  stmt_retrieve_embeddings_for_lemma = sa.text('''
  select
    fs.lu_lemma,
    fs.frame_label,
    l2_normalize(vm.embedding) as vector
  from frameinstances_split fs
  join "{emmodel}-masked" vm on vm.key = fs.instance_id
  where fs.datasetsplit_name = :datasetsplit
    and fs.split = any(:splits)
    and fs.lu_lemma = :lemmaquery;
  '''.format(emmodel=emmodel)
  ).columns(vector=sapgvec.Vector)


  factory = create_factory()
  local_clusterer = factory.create_from_name('dbclusterer', local_clustering_params['clusterer'])


  with dbh.sessionmaker() as session:
    res = session.execute(stmt_retrieve_lemmas, {
      'datasetsplit': args.dataset,
      'splits': data['splits'],
      'testsplits': data['testsplits'],
      'mincount': int(local_filter.get('min_lemmainstances', 1)),
      'maxcount': int(local_filter.get('max_lemmainstances', 1e10))
    })
    lemma_rows = res.all()

    ## run local clustering with ha and collect merge distances
    distances = []

    for lemma, l_frequency in lemma_rows:

      if l_frequency > 1:

        print(f"Collecting embeddings for {lemma}")

        res = session.execute(stmt_retrieve_embeddings_for_lemma, {
          'datasetsplit': args.dataset,
          'splits': data['splits'],
          'lemmaquery': lemma,
        })
        rows_t = dict(zip(res.keys(), zip(*res)))

        # prepare feature matrix
        M = np.array(rows_t['vector'])
        M = torch.tensor(M, device=device, dtype=torch.float32)
        num_elems = M.size(0)
        if l_frequency != num_elems:
          raise ValueError('Number of vectors differs from number of instances')

        print(f"Clustering embeddings for {lemma}")
        local_clusterer.fit(M)

        ## collect distances for merges
        distances.extend(local_clusterer.clusterer.distances_)

    ## using the merge distance at position k, the number of local clusters is exactly the number of lus in dev
    k = number_of_lus - len(lemma_rows) + 1
    distance_threshold = np.sort(np.array(distances))[::-1][:k][-1] + 1e-6

  print(str(['global', 'clusterer', 'options', 'stopping_ratio']))
  print(lus_in_same_frame_ratio)

  print('X-Means:')
  print(str(['local', 'clusterer', 'options', 'source_config', 'clusterer', 'options', 'max_clusters']))
  print(number_of_instances)

  print('HA:')
  print(str(['local', 'clusterer', 'options', 'source_config', 'clusterer', 'distance_threshold']))
  print(distance_threshold)

  # output json file for dev experiments - if filenames are given

  yamada_base_dev_settings = {
    "name": "To be set",
    "extrainfo": {"note": "unsupervised development"},
    "hyperparameters": [
      {
        "key": ["local", "alpha"],
        "values": [
          0,
          0.1,
          0.2,
          0.3,
          0.4,
          0.5,
          0.6,
          0.7,
          0.8,
          0.9,
          1
        ]
      },
      {
        "key": ["global", "clusterer", "options", "cluster_options", "linkage"],
        "values": [
          "average",
          "ward"
        ]
      },
      {
        "key": ["global", "merge_knowns"],
        "values": [
          "never"
        ]
      }
    ],
    "base_run_settings": {
      "meta": {
        "device": "cpu",
        "reuse": {
          "local": True,
          "global": True
        }
      },
      "data": {
        "dataset": args.dataset,
        "splits": data['splits'],
        "testsplits": data['testsplits'],
        "materialize": True
      },
      "local": {
        "emmodel": emmodel,
        "dim": dim,
        "filter": local_filter,
        "clusterer": {
          "type": "forward",
          "options": {
            "source_config": {
              "emmodel": emmodel,
              "dim": dim,
              "alpha": 0,
              "emaggregation": "avg"
            }
          }
        },
        "emaggregation": "avg"
      },
      "global": {
        "localclustering": "##local@latest",
        "filter": {
          "min_clusterinstances": 1,
          "max_clusterinstances": 1e10,
          "randomize_order": True,
          "random_seed": 0.946684799
        },
        "clusterer": {
          "type": "ha-stopping",
          "options": {
            "stopping_ratio": lus_in_same_frame_ratio,
            "cluster_options": {
              "metric": "euclidean"
            }
          }
        }
      }
    }
  }

  if args.dev_file_ha:

    local_ha_dev_settings = {
      "type": "ha",
      "options": {
	    "metric": "euclidean",
        "linkage": "average",
        "n_clusters": None,
        "distance_threshold": distance_threshold
      }
    }

    yamada_local_ha_dev_settings = yamada_base_dev_settings
    yamada_local_ha_dev_settings['name'] = f'{args.dataset}-split_yamada-ha_hyperparameter-tuning'
    yamada_local_ha_dev_settings['base_run_settings']['local']['clusterer']['options']['source_config']['clusterer'] = local_ha_dev_settings

    with open(args.dev_file_ha, 'w', encoding='utf-8') as f:
      json.dump(yamada_local_ha_dev_settings, f, ensure_ascii=False, indent=4)


  if args.dev_file_xmeans:

    local_xmeans_dev_settings = {
      "type": "pycl-xmeans",
      "options": {
        "amount_initial_centers": 2,
        "max_clusters": number_of_instances
      }
    }

    yamada_local_xmeans_dev_settings = yamada_base_dev_settings
    yamada_local_xmeans_dev_settings['name'] = f'{args.dataset}-split_yamada-xmeans_hyperparameter-tuning'
    yamada_local_xmeans_dev_settings['base_run_settings']['local']['clusterer']['options']['source_config']['clusterer'] = local_xmeans_dev_settings

    with open(args.dev_file_xmeans, 'w', encoding='utf-8') as f:
      json.dump(yamada_local_xmeans_dev_settings, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
  main()
