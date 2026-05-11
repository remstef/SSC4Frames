import os
import gzip
import hashlib
import json

import torch

from ssc4frames.embeddings import EmbeddingsExtractor

import ssc4frames.loghelper as loghelper; logger = loghelper.setup_logger(os.path.basename(__file__))

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
dot_env_variables_loaded = load_dotenv(dotenv_path=dotenv_path, verbose=True)
logger.info(f"Environment variables loaded from '.env'? => {f'yes ({dotenv_path})' if dot_env_variables_loaded else 'no'}")

def get_dburl_from_env(application_name='ssc_experiments'):

    default_url = f'postgresql+psycopg2://root:root@localhost:54322/ssc4frames'

    if application_name is not None:
        default_url = f'{default_url}?application_name={application_name}'

    return os.getenv('DB', default_url)


def get_datadir_from_env(default_dir='./data'):
    data_dir = os.getenv('DATADIR', default_dir)
    if dot_env_variables_loaded:
        data_dir = os.path.join(os.path.dirname(dotenv_path), data_dir)
    return data_dir
    
    
def get_obj_hash(d:object) -> str:
  json_str = json.dumps(d, sort_keys=True, indent=None, separators=(',', ':'))
  hash_str = hashlib.md5(json_str.encode('utf-8')).hexdigest()
  return hash_str


def mean_pooling(mat):
    return mat.mean(dim=0)

def first_subword(mat):
    return mat[0]

pooling_strategies = {
    'mean': mean_pooling,
    'first': first_subword
}

def load_embeddings(df, modelname, masking, pooling,
                    device, device_store,
                    filename, LOAD, SAVE, OVERWRITE, batch_size=100):

    ## compute embeddings if necessary
    if os.path.exists(filename) and not OVERWRITE:
        logger.warn(f"Embeddings file '{filename}' already exists and overwriting is not desired. \nSkipping this step.")
    else:
        luembeddings = EmbeddingsExtractor(masking=masking,
                                           batch_size=batch_size,
                                           modelname=modelname,
                                           device=device,
                                           device_store=device_store,
                                           pooling=pooling).transform(df)["luembeddings"]
        logger.info(f'Size: {luembeddings.size()}')

    # save the lu embeddings matrix
    saved = False
    if os.path.exists(filename):
        logger.info(f"Embeddings file '{filename}' already exists.")
        if SAVE and OVERWRITE:
            logger.info(f"Overwriting '{filename}'.")
            torch.save(luembeddings, filename)
            saved = True
        else:
            logger.info(f"Skip saving.")
    else:
        if SAVE:
            logger.info(f"Saving lu embeddings to '{filename}'.")
            torch.save(luembeddings, filename)
            saved = True

    # load the lu embeddings matrix, if loading is desired
    if LOAD and os.path.exists(filename) and not saved:
        logger.info(f"Loading embeddings from '{filename}'.")
        luembeddings = torch.load(filename).to(device_store)
        emb_type = "masked" if masking else "unmasked"
        logger.info(f'M - {emb_type}: {luembeddings.shape}, {luembeddings.device}')

    return luembeddings


def read_clusterings(filename):

    # ## content of clusterlabels dict:
    # {
    #     'local': {
    #         name: [clusterlabels],
    #         ...
    #     },
    #     'global': {
    #         name: [clusterlabels],
    #         ...
    #     },
    # }
    clusterlabels = {
        'local': {},
        'global': {},
    }
    ## each line contains one clustering:
    ## format: {'type': 'local'|'global', 'settings': '', 'clusterlabels': []}
    if os.path.exists(filename):
        with gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename, 'r') as clustering_file:
            for line in clustering_file:
                clustering = json.loads(line)
                clusterlabels[clustering['type']][clustering['name']] = clustering['clusterlabels']
    
    return clusterlabels


def update_value(dict_, key_tuple, value):
    d = dict_
    for key in key_tuple[:-1]:
        d = d.setdefault(key, {})
    d[key_tuple[-1]] = value
    return dict_
