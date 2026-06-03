#!/usr/bin/env python
# coding: utf-8

import click
from click_option_group import optgroup
import json

import os, sys
import pathlib
import json
import enum

from ssc4frames.helpers import get_dburl_from_env, dotenv_path, pooling_strategies
import ssc4frames.loghelper as loghelper; logger = loghelper.setup_logger(os.path.basename(__file__))
from ssc4frames.run_experiment_db_only import merge_with_default_params

## conditional import
def require_experiment_imports():
    global Counter, deepcopy, hashlib, itertools, pprint, pd, sa, update_value, runexp, ExperimentManager, ExperimentRun, Experiment, Clustering
    
    from collections import Counter
    from copy import deepcopy
    import hashlib
    import itertools
    import pprint

    import pandas as pd
    import sqlalchemy as sa

    from ssc4frames.helpers import update_value
    import ssc4frames.run_experiment_db_only as runexp
    from ssc4frames.ExperimentManager import ExperimentManager
    from ssc4frames.database import Clustering, Experiment, ExperimentRun

## define some helper variables, functions and click types

# to be merged with @see run_experiment_db_only.default_parameters
example_clustering_config_override = {
  'data': {
    'dataset': 'fn1.7-default',
    'splits': [ 'train', 'dev', 'test' ], # specify the splits which are going to be clustered (possibly in a semi-supervised fashion, labels are not necessarily required), labels are explicitly removed from instances which are in the testsplits list below
    'testsplits': [ 'test' ], # specify the datasplit instances which are used for testing, i.e. during (semi-supervised) clustering, labels are removed from those instances
  }
}

example_experiment_config_with_hyperparameter_exchange = {
    'name': 'bfn-default_split-hyperparameter-tuning',
    'extrainfo': {'note': 'example experiment'},
    'hyperparameters': [
        {
            'key': ('local', 'alpha'),
            'values': [
                0,
                0.1,
            ]
        },
        {
            'key': ('local', 'clusterer', 'options', 'criterion'),
            'values': [
                "minw_0",
                "minw_0.1",
                "minw_0.2",
            ]
        },
        {
            'key': ('global', 'clusterer', 'options', 'criterion'),
            'values': [
                "minw_0",
                "minw_0.1",
                "minw_0.2",
                "minw_0.3",
            ]
        }
    ],
    'base_run_settings': merge_with_default_params(example_clustering_config_override)
}

def get_dburl(dburl=None, application_name=None):
    if dburl is None:
        dburl = get_dburl_from_env(application_name=application_name)
    elif application_name is not None:
        dburl = f'{dburl}?application_name={application_name}'
    return dburl


def get_hash_for_datasetsplit(datasetsplit, database):
    from ssc4frames.database import DBHandler
    import sqlalchemy as sa
    dbhandler = DBHandler(database)
    with dbhandler.sessionmaker() as session:
        stmt = sa.text(f'''
          select md5(array_agg(row(global_id,split)::text ORDER BY global_id)::text)::uuid as datasplit_hash from
          frameinstances join
          split_instances on frameinstances.id=split_instances.instance_id join
          datasetsplits on split_instances.datasetsplit_id=datasetsplits.id
          where datasetsplits.name='{datasetsplit}';
        ''')
        row = session.execute(stmt).one()
    return row[0]


def get_hash_for_embeddings(datasetsplit, embeddingmodel, database):
    from ssc4frames.database import DBHandler
    import sqlalchemy as sa
    dbhandler = DBHandler(database)
    with dbhandler.sessionmaker() as session:
        stmt = sa.text(f'''
          select md5(array_agg(embedding::text ORDER BY global_id)::text)::uuid as datasplit_hash from
          "{embeddingmodel}" join
          frameinstances on frameinstances.id="{embeddingmodel}".key join
          split_instances on split_instances.instance_id="{embeddingmodel}".key join
		  datasetsplits on datasetsplits.id=split_instances.datasetsplit_id
          where datasetsplits.name='{datasetsplit}';
        ''')
        row = session.execute(stmt).one()
    return row[0]


def load_from_file_if_string(option):
    if isinstance(option, str):
        try:
            return json.loads(option)
        except:
            with open(option, 'r') as jsonfile:
                jsonstring = jsonfile.read()
            return json.loads(jsonstring)
    else:
        return option


def import_from_conllu(fname):

    global pd

    from conllu import parse_incr
    from io import open
    import pandas as pd

    rows = []
    i = 0
    with open(fname, 'r', encoding='utf-8') as fh:
        for di, cdoc in enumerate(parse_incr(fh)):
            for luid, token in enumerate(cdoc):
                if token['upos'] in ['VERB']:
                    row = {
                        'FRAME_ID' : '-1',
                        'DATA_SOURCE' : os.path.basename(fname),
                        'FRAME_NAME' : '<unk>',
                        'TOKENIZED_SENTENCE' : [t['form'] for t in cdoc],
                        'GLOBAL_SENTENCE_ID' : cdoc.metadata['sent_id'],
                        'LU_INDEX' : [luid],
                        'LU_INDEX_PART' : [],
                        'LU' : cdoc[luid]['form'],
                        'LU_LEMMA' : cdoc[luid]['lemma'],
                        'LU_LEMMA_PART' : '',
                        'LU_LEMMA_FULL' : cdoc[luid]['lemma'],
                        'SUBSTITUTES' : '',
                        'i' : i
                    }
                    i += 1
                    rows.append(row)

    df = pd.DataFrame(rows)
    df.rename(columns={"LU_LEMMA_FULL": "lu_lemma"}, inplace=True)
    df['frame_label'] = '<unk>'
    df['global_id'] = df.apply(lambda r: f'{r.DATA_SOURCE}::{r.GLOBAL_SENTENCE_ID}::{str(r.LU_INDEX).replace(' ', '')}::[{str(r.TOKENIZED_SENTENCE)[1:20].replace(' ', '')}...]::{r.lu_lemma.replace(' ', '_')}', axis=1)
    return df


def get_experiment_runs_from_experiment_config(experiment_config):
    ## create parameters for individual runs
    hyperparam_list = itertools.product(
        *[[(tuple(param_dict['key']), value) for value in param_dict['values']] for param_dict in experiment_config['hyperparameters']]
    )

    experiment_runs = []
    for p in hyperparam_list:

        run_settings = {param_name: param_value for param_name, param_value in p}
        params = deepcopy(experiment_config['base_run_settings'])
        for key_tuple, value in run_settings.items():
            update_value(params, key_tuple, value)

        experiment_runs.append(ExperimentRun(setting=runexp.merge_with_default_params(params)))

    return experiment_runs


def compare_run_configurations(exp_runs, config_runs):

    ## check if the set of identifiers is the same
    run_identifier = set([(
        Clustering.get_identifier_from_settings({'data': run.setting['data'], 'local': run.setting['local']}),
        Clustering.get_identifier_from_settings({'data': run.setting['data'], 'local': run.setting['global']}),
    ) for run in exp_runs])

    config_identifier = set([(
        Clustering.get_identifier_from_settings({'data': run.setting['data'], 'local': run.setting['local']}),
        Clustering.get_identifier_from_settings({'data': run.setting['data'], 'local': run.setting['global']}),
    ) for run in config_runs])

    return run_identifier == config_identifier


def output_results(dbh, experiment_names, metrics, average_runs, output_format, results_folder, verbose):

    metric_select_str = ','.join([f"{metric.value} as {metric.name}" for metric in metrics])

    stmt = sa.text(
        f"SELECT name, {metric_select_str} FROM experiment_scores WHERE name IN ('{'\',\''.join(experiment_names)}')"
        )

    with dbh.sessionmaker() as session:
        res = session.execute(stmt)
        experiment_run_results = dict(zip(res.keys(), zip(*res)))

    experiment_run_results_df = pd.DataFrame(experiment_run_results)

    if average_runs:
        experiment_run_results_df = experiment_run_results_df.groupby('name', as_index=False)
        if verbose:
            click.echo(experiment_run_results_df.describe())
        experiment_run_results_df = experiment_run_results_df.agg({
            metric.name.lower(): ['mean','std'] for metric in metrics
        })

    if output_format=='latex':
        latex_str = experiment_run_results_df.to_latex(
            index=False,
            float_format="{:.2f}".format,
        )
        if results_folder is None:
            click.echo(latex_str)
        else:
            with open(results_folder.joinpath(f"results.tex"), 'w') as rfile:
                rfile.write(latex_str)
    elif output_format=='json':
        if average_runs:
            experiment_run_results_df.set_index(('name',''), inplace=True)
            results = json.loads(experiment_run_results_df.to_json(orient='index'))
            if results_folder is None:
                click.echo(results)
            else:
                for key in results.keys():
                    with open(results_folder.joinpath(f"{key}.json"), 'w') as rfile:
                        json.dump(results[key], rfile)
        else:
            results = experiment_run_results_df.to_json(orient='index')
            if results_folder is None:
                click.echo(results)
            else:
                with open(results_folder.joinpath(f"results.json"), 'w') as rfile:
                    json.dump(results, rfile)
    else:
        click.echo(experiment_run_results_df)


def get_experiment_hash(experiment_config):

    manager = ExperimentManager()

    experiments = manager.get_experiments_by_name(experiment_config['name'])

    if len(experiments) == 0:
        raise ValueError("Experiment not in database.")
    elif len(experiments) > 1:
        raise ValueError("Multiple experiments with given name found in database.")
    else:
        exp = next(iter(experiments.values()))

        clustering_identifier_list = (run.clustering.identifier for run in exp.runs if run.status == 'finished')

        m = hashlib.md5()
        for h in sorted(clustering_identifier_list):
            m.update(h.encode())
        return  m.hexdigest()


def exp_config_equals_db_config(experiment_config, experiments):
    config_matches = True
    
    if len(experiments) == 0:
        click.echo("Experiment not in database.")
        config_matches = False
    elif len(experiments) > 1:
        click.echo("Multiple experiments with given name found in database.")
        config_matches = False
    else:
        exp = next(iter(experiments.values()))
        ## check if the experiment in the database matches the expected experiment from config 
        config_runs = get_experiment_runs_from_experiment_config(experiment_config)

        ## check if number of runs is the same
        if len(config_runs) != len(exp.runs):
            click.echo("Differing numer of runs for experiment and config:")
            click.echo(f"Experiment runs: {len(exp.runs)}")
            click.echo(f"Config runs: {len(config_runs)}")
            config_matches = False
        else:
            config_matches = compare_run_configurations(exp.runs, config_runs)

    if not config_matches:
        raise ValueError("Experiment config does not match with experiment in database.")
    
    return config_matches


class CorpusfileType(enum.Enum):
    CONLLU_VERB = enum.auto()
    GFNCSV      = enum.auto()

    def __call__(self, *args, **kwargs):

        global pd, read_data_as_dataframe, map_to_unified_format
        import pandas as pd
        from ssc4frames.dataloader import read_data_as_dataframe, map_to_unified_format

        print(f'Loading file {args[0]} of type {self}')
        if self.name == 'GFNCSV':
            df = read_data_as_dataframe(args[0])
            df['fixed_label'] = df['FRAME_ID']
            return map_to_unified_format(f'gfn_{os.path.basename(os.path.dirname(args[0]))}', df)
        elif self.name == 'CONLLU_VERB':
            return import_from_conllu(args[0])
        return pd.DataFrame()
    

class JsonOption(click.ParamType):
    """The json-option type allows for passing a list or dict using json as
    parameter. If the passed string is not valid json, it is interpreted as
    a filename and the content of the file is used.
    """

    name = 'json-option'

    def convert(self, value, param, ctx):
        try:
            result = load_from_file_if_string(value)
        except Exception:
            self.fail(
                value + " could not be parsed.",
                param,
                ctx,
            )

        return result
    

class Metrics(enum.Enum):

    MICRO_F1   = "micro_f1__novelty_frame"

    PRECISION  = "clusteringinfo->'evalresults'->'novelty_+_frame'->'macro avg'->'precision'"
    RECALL     = "clusteringinfo->'evalresults'->'novelty_+_frame'->'macro avg'->'recall'"
    MACRO_F1   = "clusteringinfo->'evalresults'->'novelty_+_frame'->'macro avg'->'f1-score'"

    PURITY     = "clusteringinfo->'evalresults'->'frame_induction_alleval'->'pu'"
    INV_PURITY = "clusteringinfo->'evalresults'->'frame_induction_alleval'->'ipu'"
    PURITY_F1  = "clusteringinfo->'evalresults'->'frame_induction_alleval'->'puf1'"

    BCUBED_P   = "clusteringinfo->'evalresults'->'frame_induction_alleval'->'b^3p'"
    BCUBED_R   = "clusteringinfo->'evalresults'->'frame_induction_alleval'->'b^3r'"
    BCUBED_F1  = "clusteringinfo->'evalresults'->'frame_induction_alleval'->'b^3f1'"

    SILHOUETTE = "clusteringinfo->'mean_silhouette_score_by_local_cluster'" # _ta exists as well


## start with the click commands

@click.group()
def main():
    pass

@main.command()
def version():
    git_revision_short_hash = 'None'
    ssc4frames_version = 'None'
    try:
        import subprocess
        git_revision_short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__))).decode('ascii').strip()
    except:
        pass
    from importlib.metadata import version
    ssc4frames_version = f'ssc4frames.v{version("ssc4frames")}'
    print(f"Current git commit: {git_revision_short_hash}")
    print(f"SSC4frames version: {ssc4frames_version}")
    

@main.group()
def data():
    pass

@data.command
@click.option('--database', '-db', type=str)
@click.option('--dataset', '-d', type=str, multiple=True, required=True)
@click.option('--batchsize', '-bs', type=int, required=True, default=0)
def import_dataset(database, dataset, batchsize):

    database = get_dburl(database, application_name='ssc4frames_data')

    global Importer, importer
    from ssc4frames.dataimporter import Importer
    importer = Importer(dbconnectionstring=database)
    for d in dataset:
        if ':' in d:
            ds, offset = d.split(':',1)
        else:
            ds, offset = (d,0)
        basedata = ds.split('-')[0]
        offset = int(offset)
        logger.info(f''' Importing Data:
            database = {database}
            basedata = {basedata}
            dataset = {ds}
            offset = {offset}
            batchsize = {batchsize}
        ''')
        importer.import_data(
            datasetname=ds, 
            basedataname=basedata,
            batchsize=batchsize,
            offset=offset)
    return


@data.command()
@click.argument('datasetsplit-name', type=str, required=True)
@click.option('-b', '--batchsize', type=int, default=100)
@click.option('-e', '--embedding_table', type=str, multiple=True)
@click.pass_context
def instances(ctx, datasetsplit_name, batchsize, embedding_table):

    from ssc4frames.database import DBHandler
    import sqlalchemy as sa

    join_embedding_tables = [f'left join "{tablename}" t{i} on t{i}.key = fis.instance_id' for i, tablename in enumerate(embedding_table)]
    join_embedding_tables = '\n'.join(join_embedding_tables)
    join_embedding_columnnames = ''.join([f', t{i}.embedding as {tablename.replace("-","_")}' for i, tablename in enumerate(embedding_table)])

    dbhandler = DBHandler(get_dburl())
    with dbhandler.sessionmaker() as session:
    
        current_offset = 0
        next_offset = current_offset
        while True:
            stmt = sa.text(f'''
                select fis.* {join_embedding_columnnames}
                from frameinstances_split fis
                {join_embedding_tables}
                where fis.datasetsplit_name = :_datasetsplit_name_
                limit :_batchsize_ offset :_offset_
            ''')
            res = session.execute(stmt, {
                '_datasetsplit_name_': datasetsplit_name,
                '_batchsize_': batchsize,
                '_offset_': current_offset
            })
            if current_offset == 0:
                print('\t'.join(map(str,res.keys())))
            for r in res:
                print('\t'.join(map(str, r)))
                next_offset += 1
            if next_offset == current_offset: # no change, no more rows to fetch
                break
            current_offset = next_offset

    return


@data.command
@click.option('--database', '-db', type=str)
@click.option('--dataset_name', '-d', type=str, required=True)
@click.option('--dataset_lang', '-l', type=str, required=True)
@click.option('--split_suffix', type=str, default='default')
@click.option('--train_file', '-trf', type=click.Tuple([str, click.Choice(CorpusfileType, case_sensitive=False)]), required=True, multiple=True)
@click.option('--dev_file', '-df', type=click.Tuple([str, click.Choice(CorpusfileType, case_sensitive=False)]), required=False, multiple=True)
@click.option('--test_file', '-tef', type=click.Tuple([str, click.Choice(CorpusfileType, case_sensitive=False)]), required=True, multiple=True)
@click.option('--batchsize', '-bs', type=int, required=True, default=0)
def import_custom_dataset(database, dataset_name, dataset_lang, split_suffix,
                          train_file, dev_file, test_file,
                          batchsize):

    global pd
    import pandas as pd

    database = get_dburl(database, application_name='ssc4frames_data')

    df = pd.DataFrame()
    files_per_split = {
        'train': train_file,
        'dev': dev_file,
        'test': test_file
    }
    for split in ['train', 'dev', 'test']:
        print(f'Import {split} data')
        for filename, loader in files_per_split[split]:
            file_df = loader(filename)
            file_df['split'] = split
            if split == 'test':
                df['fixed_label'] = None
            df = pd.concat([df, file_df], sort=False)


    df.reset_index(drop=True, inplace=True)
    df['c'] = range(0, df.shape[0])

    global Importer, importer
    from ssc4frames.dataimporter import Importer
    importer = Importer(dbconnectionstring=database)
    logger.info(f''' Importing Data:
        database  = {database}
        basedata  = {dataset_name}
        language  = {dataset_lang}
        datasplit = {'-'.join([dataset_name, split_suffix])}
        batchsize = {batchsize}
    ''')
    importer.import_from_dataframe(
        datasetname='-'.join([dataset_name, split_suffix]),
        basedataname=dataset_name,
        language=dataset_lang,
        df=df,
        batchsize=batchsize,
        offset=0)

    return


@data.command
@click.option('--database', '-db', type=str)
def list_datasets(database):

    database = get_dburl(database, application_name='ssc4frames_data')

    from ssc4frames.database import DBHandler, DatasetSplit
    from sqlalchemy import select

    dbhandler = DBHandler(database)
    with dbhandler.sessionmaker() as session:
        stmt = select(DatasetSplit)
        datasetsplits = session.execute(stmt).all()
        for datasetsplit in datasetsplits:
            print(datasetsplit[0])
    return


@data.command
@click.option('--database', '-db', type=str)
@click.option('--dataset', '-d', type=str, multiple=True, required=True)
@click.option('--model', '-m', type=str, required=True, default='bert-base-uncased')
@click.option('--vdim', '-v', type=int, required=True, default=768)
@click.option('--unmasked', '-um', 'input_masking', flag_value='unmasked', multiple=True, is_flag=True)
@click.option('--masked', '-ms', 'input_masking', flag_value='masked', multiple=True, is_flag=True)
@click.option('--masksubword', '-mss', 'input_masking', flag_value='mask_subwords', multiple=True, is_flag=True)
@click.option('--mask_str', '-msstr', type=str, default=None)
@click.option('--pooling', '-p', type=click.Choice(pooling_strategies.keys(), case_sensitive=True), default='mean')
@click.option('--device', type=str, default='cpu')
@click.option('--batchsize', '-bs', type=int, required=True, default=32)
@click.option('--tablename', '-t', type=str)
def import_model_embeddings(database, dataset, model, vdim, input_masking, mask_str,
                            pooling, device, batchsize, tablename):

    import gc
    import torch

    database = get_dburl(database, application_name='ssc4frames_data')

    ## using unmasked input as default
    if not input_masking:
        input_masking = ('unmasked',)

    if 'masked' in input_masking and 'mask_subwords' in input_masking:
        raise ValueError('Masking must either be on token or subword level - not both.')

    for masking_strategy in input_masking:
        m_str = None ## only set mask_str when masking_strategy is masked
        if masking_strategy == 'unmasked':
            masking = False
            mask_subwords = False
        elif masking_strategy == 'masked':
            masking = True
            mask_subwords = False
            m_str = mask_str
        elif masking_strategy == 'mask_subwords':
            masking = True
            mask_subwords = True
        else:
            raise ValueError('Unknown masking strategy')

        __import_model_embeddings(database, dataset, model, vdim, masking, m_str, mask_subwords, pooling, device, batchsize, tablename)

        ## trigger garbage collection to remove transformer model from memory
        ## maybe: refactor to reuse loaded model for masked and unmasked embeddings
        gc.collect()
        torch.cuda.empty_cache()


def __import_model_embeddings(database, dataset, model, vdim, masking, mask_str, mask_subwords, pooling, device, batchsize, tablename=None):

    global DBHandler, Dataset, FrameInstance, sa
    from ssc4frames.database import DBHandler, Dataset, FrameInstance
    import sqlalchemy as sa

    # hack the system: simply get user,pw,host,and port info for tensorstorage from db string
    from urllib.parse import urlparse
    urlsegments = urlparse(database, allow_fragments=True)
    dbname = urlsegments.path.strip('/')

    # use the tensorstorage classes with backoff and insert on miss, use ids from database
    main_fmt = 'paradedb://{user}:{passwd}@{host}:{port}/{dbname}/{modeltablename}/?dim={vdim}'
    backoff_fmt = 'model+db://{modelname}/?masked={masked_YN}&{mask_string_option}masksubwords={mask_subwords_YN}&pooling={pooling}&device={device}&data=postgresql%2Bpsycopg2://{user}:{passwd}@{host}:{port}/{dbname}'
    
    user = urlsegments.username if urlsegments.username is not None else 'root'
    password = urlsegments.password if urlsegments.password is not None else 'root'
    host = urlsegments.hostname if urlsegments.hostname is not None else 'localhost'
    port = urlsegments.port if urlsegments.port is not None else 5432
    modeltablename = (tablename if tablename is not None else model.replace('/','_')) + ('-masked' if masking else '')

    main = main_fmt.format(
        user = user,
        passwd = password,
        host = host, 
        port = port, 
        modeltablename = modeltablename, 
        vdim = vdim,
        dbname = dbname
    )
    backoff = backoff_fmt.format(
        modelname = model,
        masked_YN = 'yes' if masking else 'no',
        mask_string_option = '' if mask_str is None else f'mask_str={mask_str}&',
        mask_subwords_YN = 'yes' if mask_subwords else 'no',
        pooling = pooling,
        device = device,
        user = user,
        passwd = password,
        host = host, 
        port = port,
        dbname = dbname
    )

    logger.info(main)
    logger.info(backoff)
    from ssc4frames.tensor_storage import TensorStorage
    store = TensorStorage.fromurl(main).with_backoff(backoff).get(insert_on_miss=True)
    dbhandler = DBHandler(database)
    for d in dataset:
        d_name, d_start_offset = (d,0)
        if ':' in d:
            d_name, d_start_offset = d.split(':',1)
        __import_embeddings_by_backoff_strategy_for_db_dataset(dbhandler, store, d_name, d_start_offset, batchsize)


def __import_embeddings_by_backoff_strategy_for_db_dataset(dbhandler, tensorstore_to_with_backoff_from, datasetname, startoffset, batchsize):
    global DBHandler, Dataset, FrameInstance, sa
    try: 
        with dbhandler.sessionmaker() as session:
            ds = session.execute(sa.select(Dataset).where(Dataset.name == datasetname)).scalar_one()
            logger.info(f'{ds.name} ({ds.id})')
            dsid = ds.id
    except Exception as e:
        logger.error(f"Dataset '{datasetname}' does not exist. {e}")
        return
    
    # get keys from db    
    offset_ = startoffset
    while True:
        with dbhandler.sessionmaker() as session:
            stmt = sa.select(FrameInstance).where(FrameInstance.dataset_id == dsid) \
                .order_by(FrameInstance.id) \
                .offset(offset_) \
                .limit(batchsize)
            
            frameinstances = session.execute(stmt).scalars().all()
            if len(frameinstances) == 0:
                break
            keys_batch = [fi.id for fi in frameinstances]
            logger.info(f'{datasetname} {offset_}: {keys_batch}')
        # insert by querying keys
        _ = tensorstore_to_with_backoff_from[keys_batch]
        offset_ += batchsize


@data.command
@click.option('--connectionstring', '-db', type=str, default="default")
def init_db_tables(connectionstring):
    from sqlalchemy import text as satext
    from ssc4frames.database import DBHandler
    if (connectionstring == 'default'):
        connectionstring = get_dburl_from_env()

    logger.info(f'Initializing Tables for {connectionstring}.')
    dbh = DBHandler(connectionstring)
    logger.info(f'DB Tables initialized')

    # TODO: run scripts here ideally, currently a manual process, @see make init-db
    # basedir_sql = os.path.join(os.path.dirname(dotenv_path),'sql')
    # with dbh.engine.connect() as con:
    #     for f in filter(lambda f: f.endswith('.sql'), os.listdir(basedir_sql)):
    #         sql_script_file = os.path.join(basedir_sql, f)
    #         logger.info(f"executing script '{sql_script_file}'.")
    #         with open(sql_script_file) as file:
    #             query = satext(file.read())
    #             con.execute(query)
    
    return

@data.command
@click.option('--source-database', '-s', type=str)
@click.option('--target-database', '-t', type=str)
@click.option('--dataset', '-d', type=str, multiple=True, required=True)
@click.option('--source-table', '-st', type=str, required=True, default="bert-base-uncased")
@click.option('--target-table', '-tt', type=str, required=False)
@click.option('--batchsize', '-bs', type=int, required=True, default=32)
def copy_db_embeddings(source_database, target_database, dataset, source_table, target_table, batchsize):

    source_database = get_dburl(source_database, application_name='copy_embeddings')
    target_database = get_dburl(target_database, application_name='copy_embeddings')

    from ssc4frames.database import DBHandler, Dataset, FrameInstance, Base
    import sqlalchemy as sa
    from sqlalchemy.dialects.postgresql import insert as pginsert
    from pgvector.sqlalchemy import Vector
    
    # get vector dimensionality from the from_store
    source_dbhandler = DBHandler(source_database)
    with source_dbhandler.sessionmaker() as session:
        stmt = sa.text(f'select vector_dims(embedding) from "{source_table}" limit 1')
        row = session.execute(stmt).one()
        vdim = row[0]

    if target_table is None:
        target_table = source_table

    target_sa_table_instance = sa.Table(
        target_table, Base.metadata,
        sa.Column("key", sa.BigInteger, primary_key=True, autoincrement=False),
        sa.Column("embedding", Vector(vdim))
    )
    target_dbhandler = DBHandler(target_database)

    source_sa_table_instance = target_sa_table_instance
    if source_table != target_table:
        source_sa_table_instance = sa.Table(
            source_table, Base.metadata,
            sa.Column("key", sa.BigInteger, primary_key=True, autoincrement=False),
            sa.Column("embedding", Vector(vdim))
        )

    # start importing
    for d in dataset:
        d_name, d_start_offset = (d,0)
        if ':' in d:
            d_name, d_start_offset = d.split(':',1)
        
        with target_dbhandler.sessionmaker() as target_session:
            ds = target_session.execute(sa.select(Dataset).where(Dataset.name == d_name)).scalar_one()
            logger.info(f'Target {ds.name} (id={ds.id}) {target_database} {target_table}')
            target_dsid = ds.id

        with source_dbhandler.sessionmaker() as source_session:
            ds = source_session.execute(sa.select(Dataset).where(Dataset.name == d_name)).scalar_one()
            logger.info(f'Source {ds.name} (id={ds.id}) {source_database} {source_table}')
            source_dsid = ds.id

        # get keys from target db
        offset_ = d_start_offset
        while True:
            with target_dbhandler.sessionmaker() as target_session:
                stmt = sa.select(FrameInstance).where(FrameInstance.dataset_id == target_dsid) \
                    .order_by(FrameInstance.id) \
                    .offset(offset_) \
                    .limit(batchsize)
                target_frameinstances = target_session.execute(stmt).scalars().all()
                if len(target_frameinstances) == 0:
                    break

                target_gid_to_fi = {fi.global_id: fi for fi in target_frameinstances}
                # find the corresponding frameinstance in the source db using the global id
                with source_dbhandler.sessionmaker() as source_session:
                    stmt = sa.select(FrameInstance)\
                        .where(FrameInstance.dataset_id == source_dsid)\
                        .where(FrameInstance.global_id.in_(target_gid_to_fi.keys()))
                    source_frameinstances = source_session.execute(stmt).scalars().all()    
                    # get the embeddings
                    source_id_to_fi = { fi.id: fi for fi in source_frameinstances }
                    # query_embeddings
                    stmt = sa.select(source_sa_table_instance).where(source_sa_table_instance.c.key.in_(source_id_to_fi))
                    rows = source_session.execute(stmt).all()
                    
                # from the retrieved rows, add information of the source frame instances to map them to the target frame instances
                def get_target_key(source_key):
                    source_fi = source_id_to_fi[source_key]
                    target_fi = target_gid_to_fi[source_fi.global_id]
                    target_key = target_fi.id
                    return target_key

                insert_items = [{'key': get_target_key(source_key), 'embedding': source_embedding} for source_key, source_embedding in rows]
                if len(insert_items) != len(target_frameinstances):
                    logger.warning(f'Import of {len(target_frameinstances)} embeddings from offset {offset_} unsuccessful.')
                    logger.warning(f'{d_name} {offset_}: {list(source_id_to_fi.keys())} ==> {list(map(lambda d: d['key'], insert_items))}')
                else:
                    logger.info(f'Import of {len(target_frameinstances)} embeddings from offset {offset_} successful.')
                    
                stmt = pginsert(target_sa_table_instance).on_conflict_do_nothing(index_elements=['key'])
                target_session.execute(stmt, insert_items)
            
            offset_ += batchsize
  
    return

@data.command
@click.argument('datasetsplit')
@click.option('--database', '-db', type=str)
def get_datasetsplit_hash(datasetsplit, database):

    database = get_dburl(database, application_name='ssc4frames_data')

    hash_value = get_hash_for_datasetsplit(datasetsplit, database)
    print(hash_value)

    return

@data.command
@click.argument('datasetsplit')
@click.argument('embeddingmodel')
@click.option('--database', '-db', type=str)
def get_embeddings_hash(datasetsplit, embeddingmodel, database):

    database = get_dburl(database, application_name='ssc4frames_data')

    hash_value = get_hash_for_embeddings(datasetsplit, embeddingmodel, database)
    print(hash_value)

    return

### CLustering Group

@main.group()
@click.pass_context
def clustering(ctx):
    require_experiment_imports()
    

@clustering.command()
@click.argument('config', type=JsonOption(), required=False, default=example_clustering_config_override)
@click.option('--no-wait', is_flag=True)
@click.pass_context
def run(ctx, config, no_wait):

    clustering_config_override = config
    clustering_config = merge_with_default_params(clustering_config_override)
    manager = ExperimentManager()
    manager.run_with_setting(clustering_config, same_thread=True, new_process=False, raise_worker_exception=True, await_key_confirmation=(not no_wait))

    return



@clustering.command()
@click.argument('cid', type=int, required=True)
@click.option('-e', '--embeddings', is_flag=True)
@click.option('-a', '--all', is_flag=True)
@click.option('-b', '--batchsize', type=int, default=100)
@click.pass_context
def instances(ctx, cid, embeddings, all, batchsize):
    
    from ssc4frames.database import DBHandler
    import sqlalchemy as sa

    allinstances_query = 'select * from frameinstances_split where datasetsplit_id = :_datasetsplit_id_ and split = any(:_splits_)'
    if all:
        allinstances_query = 'select * from frameinstances_split where datasetsplit_id = :_datasetsplit_id_'
        # allinstances_query = 'select * from frameinstances_split where datasetsplit_name = :_dataset_name_'

    dbhandler = DBHandler(get_dburl())
    with dbhandler.sessionmaker() as session:
    
        clustering = session.get(Clustering, cid)
        if clustering is None:
            print(f'Clustering with id {cid} not found.', file=sys.stderr)
            return
        # check if we have a local or a global clustering here
        if clustering.type == 'localglobal':
            # get the respective local clustering id (use identifier, so we can resolve merged clusterings)
            clusteringident_local = clustering.setting['global']['localclustering'].split('@')[0].split('[')[0]
            clustering_local = session.execute(sa.select(Clustering).where(Clustering.identifier == clusteringident_local)).scalar_one_or_none()
            if clustering_local is None:
                raise KeyError(f'Local clustering for global clustering {cid} not found')
            cid_local = clustering_local.id
        else:
            cid_local = cid
        
        if embeddings:
            # join result set with averaged embeddings i.e. just use the prepared frameinstances_split_vectorized__<local-cluster-id> view, we need to get the respective local clustering (if the current clustering is not a local clustering)
            viewname = f'frameinstances_split_vectorized__{cid_local}'
            allinstances_query = allinstances_query.replace('frameinstances_split', viewname)
            # check if view exists and if not create a non-materialized one
            from sqlalchemy import inspect
            inspector = inspect(session.get_bind())
            views = inspector.get_view_names() + inspector.get_materialized_view_names()
            if viewname not in views:
                # create the view
                datadict = clustering.setting['data']
                datadict['materialize'] = False
                raise IndexError(f'View for local clustering does not exist, consider (re-)creating it.')
                # TODO:
                # runexp.create_instances_view_if_not_exists(dbhandler, datadict, cid_local, emmodel, vectordim:int,  alphaval:float):
        
        current_offset = 0
        next_offset = current_offset
        while True:
            stmt = sa.text(f'''
                with allinstances as (
                    {allinstances_query} limit {batchsize} offset {current_offset}
                ), cluster_assigned as (
                    select * from instanceassignments where clusteringid = :_clusteringid_
                )
                select 
                    i.instance_id, i.datasetsplit_name, i.split, i.lu_lemma, i.frame_label as true_label, 
                    ci.clusterid, ci.clusterlabel, ci.tclusterlabel as transitive_clusterlabel, ci.assignmentinfo{'' if not embeddings else ', i.vector as embedding'}
                from allinstances i left outer join cluster_assigned ci on i.instance_id = ci.instance_id
            ''')
            res = session.execute(stmt, {
                '_clusteringid_': cid,
                '_datasetsplit_id_': clustering.datasetsplit_id,
                '_splits_': clustering.splits
            })
            if current_offset == 0:
                print('\t'.join(map(str,res.keys())))
            for r in res:
                print('\t'.join(map(str, r)))
                next_offset += 1
            if next_offset == current_offset: # no change, no more rows to fetch
                break
            current_offset = next_offset

    return


@clustering.command()
@click.argument('datasetsplitname', type=str, required=False, nargs=-1)
@click.pass_context
def list(ctx, datasetsplitname):
    from ssc4frames.database import DBHandler
    import sqlalchemy as sa
    
    # convert tuple to list
    dsnames = [ name for name in datasetsplitname ]

    dbhandler = DBHandler(get_dburl())
    with dbhandler.sessionmaker() as session:

        res = session.execute(
            sa.text('''
                select
                    cl.id as clusteringid, 
                    ds.name as datasetsplit_name,
                    cl.splits, 
                    cl.numinstances, 
                    cl.numclusters, 
                    cl.type,
                    cl.status,
                    cl.start,
                    cl.finish,
                    cl.identifier,
                    cl.setting,
                    cl.extrainfo
                from clusterings cl 
                join datasetsplits ds on ds.id = cl.datasetsplit_id
                where ds.name = any(:_dsnames_)
            '''), 
            {'_dsnames_': dsnames}
        )
        print('\t'.join(map(str,res.keys())))
        for r in res:
            print('\t'.join(map(str, r)))


@clustering.command()
@click.argument('cids', type=int, required=True, nargs=-1)
@click.pass_context
def info(ctx, cids):
    from ssc4frames.database import DBHandler
    import sqlalchemy as sa
    
    # convert tuple to list
    cids = [cid for cid in cids]

    dbhandler = DBHandler(get_dburl())
    with dbhandler.sessionmaker() as session:

        res = session.execute(
            sa.text('''
                select
                    id, 
                    datasetsplit_id, 
                    splits, 
                    numinstances, 
                    numclusters, 
                    type,
                    status,
                    start,
                    finish,
                    identifier,
                    setting,
                    extrainfo
                from clusterings 
                where id = any(:_clusteringids_)
            '''), 
            {'_clusteringids_': cids}
        )
        print('\t'.join(map(str,res.keys())))
        for r in res:
            print('\t'.join(map(str, r)))


@clustering.command()
@click.argument('cid', type=int, required=True)
@click.option('-e', '--embeddings', is_flag=True)
@click.option('-b', '--batchsize', type=int, default=100)
@click.pass_context
def clusters(ctx, cid, embeddings, batchsize):
    
    from ssc4frames.database import DBHandler
    import sqlalchemy as sa

    dbhandler = DBHandler(get_dburl())
    with dbhandler.sessionmaker() as session:
    
        clustering = session.get(Clustering, cid)
        if clustering is None:
            print(f'Clustering with id {cid} not found.', file=sys.stderr)
            return
        
        query_statement = f'''
            select cl.clusteringid, cl.id as clusterid, cl.label, cl.extrainfo->>'transitive_label' as transitive_label, cl.extrainfo
            from clusters cl
            where cl.clusteringid = :_clusteringid_
        '''
        
        if embeddings:
            # join result set with averaged embeddings i.e. use the prepared clusterembeddings__<cid> table
            # no need to check if we have a local or a global clustering, both create a table called clusterembeddings__<cid>
            # leave that to the user
            cluster_embeddings_tablename = f'clusterembeddings__{cid}'
            query_statement = f'''
                select cl.clusteringid, cl.id as clusterid, cl.label, cl.extrainfo->>'transitive_label' as transitive_label, cl.extrainfo, cle.embedding
                from clusters cl
                left join {cluster_embeddings_tablename} cle on cl.id = cle.clusterid
                where cl.clusteringid = :_clusteringid_
            '''
            
        current_offset = 0
        next_offset = current_offset
        while True:
            stmt = sa.text(f'''
                {query_statement}
                limit {batchsize} offset {current_offset}
            ''')
            res = session.execute(stmt, {
                '_clusteringid_': cid
            })
            if current_offset == 0:
                print('\t'.join(map(str,res.keys())))
            for r in res:
                print('\t'.join(map(str, r)))
                next_offset += 1
            if next_offset == current_offset: # no change, no more rows to fetch
                break
            current_offset = next_offset

    return
    


### Experiment Group

@main.group()
@click.option('--experiment_config', type=JsonOption()) #, default=example_experiment_config_with_hyperparameter_exchange)
@click.option('--experiment_hash')
@click.option('--experiment_name')
@click.option('--experiment_id')
@click.option('--datasetsplit_hash')
@click.option('--embeddings_hash')
@click.option('--embeddings_masked_hash')
@click.pass_context
def experiment(ctx, experiment_config, experiment_hash, experiment_name, experiment_id, datasetsplit_hash, embeddings_hash, embeddings_masked_hash):
    require_experiment_imports()

    ctx.ensure_object(dict)

    ctx.obj['EXPERIMENT_CONFIG'] = experiment_config
    ctx.obj['EXPERIMENT_NAME'] = experiment_name
    ctx.obj['EXPERIMENT_ID'] = experiment_id

    if any([datasetsplit_hash, embeddings_hash, embeddings_masked_hash]):

        hyperparameters = set([tuple(params['key']) for params in experiment_config['hyperparameters']])
        database = get_dburl_from_env()

        from ssc4frames.cli import get_hash_for_datasetsplit, get_hash_for_embeddings
        from pathlib import Path

        if experiment_hash:

            experiment_hash_db = get_experiment_hash(experiment_config)

            if Path(experiment_hash).is_file():
                with open(experiment_hash,"r") as f:
                    experiment_hash = f.read().strip()

            if experiment_hash != experiment_hash_db:
                print(experiment_hash)
                print(experiment_hash_db)
                raise ValueError('Experiment runs are different than expected')

        if datasetsplit_hash:
            ## check that datasetsplit is the same for each run (i.e. not part of hyperparameters)
            assert(('data', 'dataset') not in hyperparameters)

            datasetsplit = experiment_config['base_run_settings']['data']['dataset']
            datasetsplit_hash_db = str(get_hash_for_datasetsplit(datasetsplit, database))

            if Path(datasetsplit_hash).is_file():
                with open(datasetsplit_hash,"r") as f:
                    datasetsplit_hash = f.read().strip()

            if datasetsplit_hash != datasetsplit_hash_db:
                raise ValueError('Datasetsplit is different than expected')

        if embeddings_hash or embeddings_masked_hash:

            ## check that embeddingmodel is the same for each run (i.e. not part of hyperparameters)
            assert(('local', 'emmmodel') not in hyperparameters)

            emmodel = experiment_config['base_run_settings']['local']['emmodel']

            if embeddings_hash:
                embeddings_hash_db = str(get_hash_for_embeddings(datasetsplit, emmodel, database))

                if Path(embeddings_hash).is_file():
                    with open(embeddings_hash,"r") as f:
                        embeddings_hash = f.read().strip()

                if embeddings_hash != embeddings_hash_db:
                    raise ValueError('Embeddings are different than expected')

            if embeddings_masked_hash:

                embeddings_hash_db = str(get_hash_for_embeddings(datasetsplit, emmodel + '-masked', database))

                if Path(embeddings_masked_hash).is_file():
                    with open(embeddings_masked_hash,"r") as f:
                        embeddings_hash = f.read().strip()

                if embeddings_hash != embeddings_hash_db:
                    raise ValueError('Masked embeddings are different than expected')


@experiment.command()
def list():
    manager = ExperimentManager()
    experiments = manager.list_experiments()
    for k, v in experiments.items():
        print(f'{v.id}: {v.name} (#{len(v.runs)} {v.get_status()})')


@experiment.command()
@click.pass_context
def create(ctx):

    experiment_config = ctx.obj['EXPERIMENT_CONFIG']
    if experiment_config is None:
        click.echo("Experiment config (--experiment_config) is missing.")
        return    

    manager = ExperimentManager()
    experiments = manager.get_experiments_by_name(experiment_config['name'])
    if len(experiments) > 0:
        click.echo("Experiment with given name already exists.")
    else:

        experiment_runs = get_experiment_runs_from_experiment_config(experiment_config)

        ## create experiment
        experiment = manager.add_experiment(
            Experiment(name=experiment_config['name'],
                       extrainfo=experiment_config['extrainfo'],
                       runs=experiment_runs))
        click.echo(f'Added:\n{experiment}')

@experiment.command()
@click.option('--n_workers', type=int, default=4)
@click.option('--use_thread_pool', is_flag=True)
@click.pass_context
def run(ctx, n_workers, use_thread_pool):
    manager = ExperimentManager()
    experiment = get_experiment_from_ctxobj(ctx.obj, manager)
    manager.run_experiment_parallel(experiment, n_workers=n_workers, process_pool=(not use_thread_pool))
    return


@experiment.command()
@click.option('--verbose', is_flag=True)
@click.pass_context
def status(ctx, verbose):

    manager = ExperimentManager()

    experiment = get_experiment_from_ctxobj(ctx.obj, manager)

    exp = experiment
    click.echo(exp)

    for status, number in Counter([r.status for r in exp.runs]).items():
        click.echo(f"{status}: {number}")

    ## check if the experiment in the database matches the expected experiment from config 
    experiment_config = ctx.obj['EXPERIMENT_CONFIG']
    if experiment_config is None:
        click.echo("Omitting config santity check. Provide --config if sanity check is desired.")
    else:
        config_runs = get_experiment_runs_from_experiment_config(experiment_config)

        ## check if number of runs is the same
        if len(config_runs) != len(exp.runs):
            click.echo("Differing numer of runs for experiment and config:")
            click.echo(f"Experiment runs: {len(exp.runs)}")
            click.echo(f"Config runs: {len(config_runs)}")
        else:
            ## check if the set of identifiers is the same
            if compare_run_configurations(exp.runs, config_runs):
                click.echo('Experiment runs match config')
            else:
                click.echo('Experiment runs do not match config')

    if verbose:
        dburl = get_dburl_from_env()
        dbh = runexp.setup_database_handler(dburl)

        # get duration of local and global clusterings
        stmt = sa.text(
            f"SELECT clustering_duration, clustering_duration_local, local_cid FROM experiment_scores WHERE experiment_id={exp.id}"
        )

        with dbh.sessionmaker() as session:
            res = session.execute(stmt)
            clustering_durations = dict(zip(res.keys(), zip(*res)))

        clustering_durations_df = pd.DataFrame(clustering_durations)

        click.echo("Global clustering durations:")
        global_clustering_durations = clustering_durations_df.clustering_duration
        click.echo(global_clustering_durations.describe())

        click.echo("Local clustering durations:")
        local_clustering_durations = clustering_durations_df.drop_duplicates(subset=['clustering_duration_local', 'local_cid']).clustering_duration_local
        click.echo(local_clustering_durations.describe())


def get_experiment_from_ctxobj(ctx_obj, manager):
    if 'EXPERIMENT_ID' in ctx_obj and ctx_obj['EXPERIMENT_ID'] is not None:
        experiment_id = ctx_obj['EXPERIMENT_ID']
        experiment = manager.get_experiment_by_id(experiment_id)
        if experiment == None:
            click.echo(f"Experiment not in database (id {experiment_id})")
            sys.exit(1)
    else:
        if 'EXPERIMENT_NAME' in ctx_obj and ctx_obj['EXPERIMENT_NAME'] is not None:
            experiment_name = ctx_obj['EXPERIMENT_NAME']
        else:
            experiment_config = ctx_obj['EXPERIMENT_CONFIG']
            experiment_name = experiment_config['name']
        experiments = manager.get_experiments_by_name(experiment_name)
        if len(experiments) == 0:
            click.echo(f"Experiment not in database (name {experiment_name})")
            sys.exit(1)
        elif len(experiments) > 1:
            click.echo(f"Multiple experiments with given name found in database. (name: {experiment_name})")
            sys.exit(1)
        experiment = next(iter(experiments.values()))
    return experiment


@experiment.command()
@click.pass_context
def remove(ctx):

    manager = ExperimentManager()

    experiment = get_experiment_from_ctxobj(ctx.obj, manager)

    click.echo(experiment)
    if click.confirm('Remove experiment from database?'):
        click.echo("Removing experiment.")
        manager.delete(experiment)
    else:
        click.echo('Aborted.')


@experiment.command()
@click.pass_context
def get_hash(ctx):

    experiment_config = ctx.obj['EXPERIMENT_CONFIG']
    if experiment_config is None:
        click.echo("Experiment config (--experiment_config) is missing.")
        return    
    experiment_hash = get_experiment_hash(experiment_config)
    click.echo(experiment_hash)



@experiment.command()
@click.option('--metric', 'metrics', multiple=True,
              type=click.Choice(Metrics, case_sensitive=False), default=(Metrics.MICRO_F1,))
@click.option('-n', type=int, default=1)
@click.option('--verbose', is_flag=True)
@optgroup.group('Output experiment config',
                help='Output the configuaration of an experiment using the best hyperparameters')
@optgroup.option('--output_experiment_name')
@optgroup.option('--output_experiment_file', type=click.File('w'))
@optgroup.option('--extrainfo_note', default='supervised test')
@optgroup.option('--train_split', '-tr', 'train_splits', multiple=True, default=('train',))
@optgroup.option('--test_split', '-t', 'test_splits', multiple=True, default=('test',))
@optgroup.option('--no_random_seeds', '-sr', 'skip_random_seeds', is_flag=True)
@optgroup.option('--random_seed', '-r', 'random_seeds', multiple=True,
              default=(946684799, 539183563, 171258316, 744166688, 659689477))
@click.pass_context
def best_hyperparameters(ctx, metrics:Metrics, n, verbose,
                         output_experiment_name, output_experiment_file, extrainfo_note, train_splits, test_splits, skip_random_seeds, random_seeds):

    ## get best hyperparameters for an experiment that has been run regarding specific metric

    experiment_config = ctx.obj['EXPERIMENT_CONFIG']
    if experiment_config is None:
        click.echo("Experiment config (--experiment_config) is missing.")
        return    

    manager = ExperimentManager()
    experiments = manager.get_experiments_by_name(experiment_config['name'])

    if len(experiments) == 0:
        click.echo("Experiment not in database.")
    elif len(experiments) > 1:
        click.echo("Multiple experiments with given name found in database.")
    else:
        exp = next(iter(experiments.values()))

    ## Warn if not all experiment runs have status finished
    if exp.get_status() != str(set(['finished'])):
        click.echo('Not all runs have been finished.')

    dburl = get_dburl_from_env()
    dbh = runexp.setup_database_handler(dburl)

    metric_select_str = ','.join([f"{metric.value} as score_{metric.name}, RANK() over (PARTITION BY experiment_id ORDER BY {metric.value} DESC) rank_{metric.name}" for metric in metrics])
    metric_average_rank = '(' + '+'.join([f"rank_{metric.name}" for metric in metrics]) + ')' + f'/{len(metrics)}.0 as avg_rank'

    # get info and setting for best run regarding given metrics
    stmt = sa.text(
    f"""
    WITH
      ranked_clusterings AS (
        SELECT clusteringinfo, setting, local_cid, clustering_id, {metric_select_str} FROM experiment_scores WHERE experiment_id={exp.id}
      ),
      ranked_clusterings_avg AS (
        SELECT clusteringinfo, setting, local_cid, clustering_id, {','.join([f'score_{metric.name}, rank_{metric.name}' for metric in metrics])}, {metric_average_rank} FROM ranked_clusterings
      )
    SELECT clusteringinfo, setting, local_cid, clustering_id, {','.join([f'score_{metric.name}, rank_{metric.name}' for metric in metrics])}, RANK() over (ORDER BY avg_rank) rank_number FROM ranked_clusterings_avg ORDER BY rank_number FETCH NEXT {n} ROWS WITH TIES"""
        ).columns(clusteringinfo=sa.types.JSON, setting=sa.types.JSON)

    with dbh.sessionmaker() as session:
        res = session.execute(stmt)
        clustering_info_settings = dict(zip(res.keys(), zip(*res)))

    if len(clustering_info_settings["rank_number"]) > n:
        click.echo(f'Returned {len(clustering_info_settings["rank_number"])} settings because of ties')
    settings = []
    for i in range(len(clustering_info_settings["rank_number"])):

        setting_dict = {}
        setting_dict['Rank'] = clustering_info_settings["rank_number"][i]
        setting_dict['cid'] = clustering_info_settings["clustering_id"][i]
        setting_dict['local_cid'] = clustering_info_settings["local_cid"][i]
        for metric in metrics:
            setting_dict[metric.name] = clustering_info_settings[f"score_{metric.name.lower()}"][i]
            setting_dict[f"rank_{metric.name}"] = clustering_info_settings[f"rank_{metric.name.lower()}"][i]

        if verbose:
            click.echo(f'Clustering {i+1}, Rank: {clustering_info_settings["rank_number"][i]}')

            ## output chosen metrics
            for metric in metrics:
                click.echo(f'{metric.name}: {clustering_info_settings[f"score_{metric.name.lower()}"][i]}, Rank: {clustering_info_settings[f"rank_{metric.name.lower()}"][i]}')

            ## output other metrics
            click.echo(f"Frame identification: {clustering_info_settings['clusteringinfo'][i]['evalresults']['novelty_+_frame']['micro avg']}")
            click.echo(f"Frame induction: {clustering_info_settings['clusteringinfo'][i]['evalresults']['frame_induction_alleval']}")

        ## get settings fo all hyperparameters
        run_settings = clustering_info_settings['setting'][i]
        for hyperparameter_path in [hyperparameter['key'] for hyperparameter in experiment_config['hyperparameters']]:
            setting = run_settings
            for key in hyperparameter_path:
                setting = setting.get(key, {})
            if verbose:
                click.echo(hyperparameter_path)
                click.echo(setting)

            setting_dict[json.dumps(hyperparameter_path)] = setting

        settings.append(setting_dict)
        if verbose:
            click.echo(f'*****************')

    settings_df = pd.DataFrame(settings)
    print(settings_df)

    if output_experiment_file:

        ## get best settings - handling ties
        min_rank = settings_df['Rank'].min()
        non_hp_keys = ['Rank', 'cid', 'local_cid']
        non_hp_keys.extend([key_name for metric in metrics for key_name in [f'{metric.name}', f'rank_{metric.name}']])

        best_settings = (
            settings_df[settings_df['Rank'] == min_rank]
            .sort_values(by=settings_df.columns.tolist(), kind="mergesort")[[c for c in settings_df.columns if c not in non_hp_keys]]
            .iloc[0]
        )

        if output_experiment_name:
            experiment_config['name'] = output_experiment_name
        experiment_config['extrainfo']['note'] = extrainfo_note
        experiment_config['hyperparameters'] = [{'key': json.loads(key), 'values': [value]} for key, value in best_settings.items()]

        if not skip_random_seeds:

            ## don't add random seeds for specified (deterministic) local clusterers
            if experiment_config['base_run_settings']['local']['clusterer']['type'] not in {'ident', 'const', 'forward', 'ha'}:
                experiment_config['hyperparameters'].append(
                    {'key': ['local', 'clusterer', 'options', 'random_state'], 'values': list(random_seeds)}
                )

            experiment_config['hyperparameters'].append(
                {'key': ['global', 'clusterer', 'options', 'random_state'], 'values': list(random_seeds)}
            )

        experiment_config['base_run_settings']['data']['splits'] = list(train_splits + test_splits)
        experiment_config['base_run_settings']['data']['testsplits'] = list(test_splits)

        import numpy as np
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.bool_):
                    return super(NpEncoder, self).encode(bool(obj))

                return super(NpEncoder, self).default(obj)

        json.dump(experiment_config, output_experiment_file, indent=4, cls=NpEncoder)


@experiment.command()
@click.option('--metric', 'metrics', multiple=True,
              type=click.Choice(Metrics, case_sensitive=False), default=(Metrics.MICRO_F1,))
@click.option('--average_runs', is_flag=True)
@click.option('--latex', 'output_format', flag_value='latex', is_flag=True)
@click.option('--json', 'output_format', flag_value='json', is_flag=True)
@click.option('--results_folder', type=click.Path(exists=True, path_type=pathlib.Path))
@click.option('--verbose', is_flag=True)
@click.option('--check_config_runs', is_flag=True)
@click.pass_context
def results(ctx, metrics, average_runs, output_format, results_folder, verbose, check_config_runs):

    manager = ExperimentManager()

    if check_config_runs:
        if 'EXPERIMENT_CONFIG' in ctx.obj:
            ec = ctx.obj['EXPERIMENT_CONFIG']
            experiments = manager.get_experiments_by_name(ec['name'])
            exp_config_equals_db_config(ctx.obj['EXPERIMENT_CONFIG'], experiments)
        else:
            click.echo('To Check config runs please provide the --experiment_config parameter!')
            return
    
    e = get_experiment_from_ctxobj(ctx.obj, manager)
    output_results(manager._dbh, [ e.name ], metrics, average_runs, output_format, results_folder, verbose)


@experiment.command()
@click.pass_context
@click.option('--check_config_runs', is_flag=True)
@click.option('--no-config', 'omit_config', is_flag=True )
@click.option('--logs', 'show_logs', is_flag=False, flag_value=0, default=-1, type=int )
def inspect(ctx, check_config_runs, omit_config, show_logs):
    
    manager = ExperimentManager()

    if check_config_runs:
        if 'EXPERIMENT_CONFIG' in ctx.obj:
            ec = ctx.obj['EXPERIMENT_CONFIG']
            experiments = manager.get_experiments_by_name(ec['name'])
            exp_config_equals_db_config(ctx.obj['EXPERIMENT_CONFIG'], experiments)
        else:
            click.echo('To Check config runs please provide the --experiment_config parameter!')
            return
    
    e = get_experiment_from_ctxobj(ctx.obj, manager)

    # refresh to get all values
    if show_logs >= 0:
        e = manager.refresh_experiment(e, sa_options=(
            sa.orm.undefer_group('extrainfos'),
            sa.orm.joinedload(Experiment.runs).undefer_group('extrainfos').undefer_group('logs'),
        ))
    else:
        e = manager.refresh_experiment(e)        

    print(f'''=== === ===
Name: {e.name}
Status: {e.get_status()}
#Runs: {0 if e.runs == None else len(e.runs)}
Extrainfo: {pprint.pformat(e.extrainfo)}
--- RUNS ---''')

    for er in e.runs:
        print(f'''=== {er.experiment_id}.{er.id} ===
Status: {er.status}
Run-Requirements: {er.require}
Clustering: {er.clustering_id}
Extrainfo: {'{}' if er.extrainfo == None else '\n'+pprint.pformat(er.extrainfo)}
Config: 
{'<OMITTED>' if omit_config else pprint.pformat(er.setting)}
# Logs: {'<OMITTED>' if show_logs < 0 else (0 if er.logs == None else len(er.logs))}
=== LOGS {er.experiment_id}.{er.id} ==={
    '\n<OMITTED>' if show_logs < 0 else ( 
        '\n...' if show_logs >= 0 and er.logs is not None and len(er.logs) > show_logs else ''
    )
}{
    '' if show_logs < 0 else (
        '\n'+'\n'.join(er.logs[-show_logs:]) if er.logs is not None and show_logs > 0 else (
            '\n'+'\n'.join(er.logs) if er.logs is not None else ''
        )
    )
}
--- <END> {er.experiment_id}.{er.id} <END> ---''')
    return


@experiment.command()
@click.argument('reset_type',
              type=click.Choice(['all', 'started', 'failed']))
@click.pass_context
def reset(ctx, reset_type):

    ## reset run status for ( all | started | failed ) runs of the experiment

    manager = ExperimentManager()
    exp = get_experiment_from_ctxobj(ctx.obj, manager)
    click.echo(exp)

    if click.confirm(f'Reset {reset_type} experiment runs?'):

        click.echo(f"Resetting {reset_type} experiment runs.")
        if reset_type=='all':
            manager.reset_experiment(exp)
        else:
            for erun in exp.runs:
                if erun.status == reset_type:
                    manager.reset_experiment_run(experiment_run=erun)
    else:
        click.echo('Aborted.')


@main.group()
@click.option('--experiment_folder', type=click.Path(exists=True), required=True)
@click.pass_context
def experiments(ctx, experiment_folder):
    require_experiment_imports()
    ctx.ensure_object(dict)

    import json

    ctx.obj['EXPERIMENT_FOLDER'] = experiment_folder
    ctx.obj['EXPERIMENT_FILES'] = [f for f in pathlib.Path(experiment_folder).glob('*.json') if f.is_file()]
    ctx.obj['EXPERIMENT_NAMES'] = [json.loads(config_file.read_text(encoding="UTF-8"))['name']
                                   for config_file in ctx.obj['EXPERIMENT_FILES']]


@experiments.command()
@click.pass_context
def status(ctx):

    # get the following information for all configurations in the given folder:
    # - name
    # - config file
    # - how many experiments with that name exist in the database
    # if exactly one:
    # - does the configuration of the runs match the configuration in the file?
    # - how is the status of the runs

    experiments_info = []
    manager = ExperimentManager()

    for exp_name, exp_config_file in zip(ctx.obj['EXPERIMENT_NAMES'], ctx.obj['EXPERIMENT_FILES']):

        config_runs = get_experiment_runs_from_experiment_config(json.loads(exp_config_file.read_text(encoding="UTF-8")))
        experiments = manager.get_experiments_by_name(exp_name)
        experiment_info = {
            'name': exp_name,
            'filename': exp_config_file,
            'experiments_in_db': len(experiments),
            'status': None,
            'runs_in_config': len(config_runs),
            'runs_in_db': None,
            'configuration_matches': None
        }

        experiments_info.append(experiment_info)

        if len(experiments) == 0:
            pass
        elif len(experiments) > 1:
            pass
        else:
            exp = next(iter(experiments.values()))
            experiment_info['status'] = exp.get_status()
            experiment_info['runs_in_db'] = len(exp.runs)
            experiment_info['configuration_matches'] = compare_run_configurations(exp.runs, config_runs)

    click.echo(pd.DataFrame(experiments_info))


@experiments.command()
@click.option('--metric', 'metrics', multiple=True,
              type=click.Choice(Metrics, case_sensitive=False), default=(Metrics.MICRO_F1,))
@click.option('--average_runs', is_flag=True)
@click.option('--latex', 'output_format', flag_value='latex', is_flag=True)
@click.option('--json', 'output_format', flag_value='json', is_flag=True)
@click.option('--results_folder', type=click.Path(exists=True, path_type=pathlib.Path))
@click.option('--verbose', is_flag=True)
@click.pass_context
def results(ctx, metrics, average_runs, output_format, results_folder, verbose):

    dburl = get_dburl_from_env()
    dbh = runexp.setup_database_handler(dburl)

    output_results(dbh, ctx.obj['EXPERIMENT_NAMES'], metrics, average_runs, output_format, results_folder, verbose)


if __name__ == '__main__':
    main()
