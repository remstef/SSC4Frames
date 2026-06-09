from ssc4frames.run_experiment_db_only import merge_with_default_params
import click
from click_option_group import optgroup
import json

import os
import sys
import pathlib
import json
import enum

from ssc4frames.helpers import get_dburl_from_env, dotenv_path, pooling_strategies
import ssc4frames.loghelper as loghelper
logger = loghelper.setup_logger(os.path.dirname(__file__))

# conditional import


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

# define some helper variables, functions and click types


# to be merged with @see run_experiment_db_only.default_parameters
example_clustering_config_override = {
    'data': {
        'dataset': 'fn1.7-default',
        # specify the splits which are going to be clustered (possibly in a semi-supervised fashion, labels are not necessarily required), labels are explicitly removed from instances which are in the testsplits list below
        'splits': ['train', 'dev', 'test'],
        # specify the datasplit instances which are used for testing, i.e. during (semi-supervised) clustering, labels are removed from those instances
        'testsplits': ['test'],
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


def get_dburl(dburl=None, application_name='None'):
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
                        'FRAME_ID': '-1',
                        'DATA_SOURCE': os.path.basename(fname),
                        'FRAME_NAME': '<unk>',
                        'TOKENIZED_SENTENCE': [t['form'] for t in cdoc],
                        'GLOBAL_SENTENCE_ID': cdoc.metadata['sent_id'],
                        'LU_INDEX': [luid],
                        'LU_INDEX_PART': [],
                        'LU': cdoc[luid]['form'],
                        'LU_LEMMA': cdoc[luid]['lemma'],
                        'LU_LEMMA_PART': '',
                        'LU_LEMMA_FULL': cdoc[luid]['lemma'],
                        'SUBSTITUTES': '',
                        'i': i
                    }
                    i += 1
                    rows.append(row)

    df = pd.DataFrame(rows)
    df.rename(columns={"LU_LEMMA_FULL": "lu_lemma"}, inplace=True)
    df['frame_label'] = '<unk>'
    df['global_id'] = df.apply(
        lambda r: f'{r.DATA_SOURCE}::{r.GLOBAL_SENTENCE_ID}::{str(r.LU_INDEX).replace(' ', '')}::[{str(r.TOKENIZED_SENTENCE)[1:20].replace(' ', '')}...]::{r.lu_lemma.replace(' ', '_')}', axis=1)
    return df


def get_experiment_runs_from_experiment_config(experiment_config):
    require_experiment_imports()
    # create parameters for individual runs
    hyperparam_list = itertools.product(
        *[[(tuple(param_dict['key']), value) for value in param_dict['values']] for param_dict in experiment_config['hyperparameters']]
    )

    experiment_runs = []
    for p in hyperparam_list:

        run_settings = {
            param_name: param_value for param_name, param_value in p}
        params = deepcopy(experiment_config['base_run_settings'])
        for key_tuple, value in run_settings.items():
            update_value(params, key_tuple, value)

        experiment_runs.append(ExperimentRun(
            setting=runexp.merge_with_default_params(params)))

    return experiment_runs


def compare_run_configurations(exp_runs, config_runs):
    require_experiment_imports()
    # check if the set of identifiers is the same
    run_identifier = set([(
        Clustering.get_identifier_from_settings(
            {'data': run.setting['data'], 'local': run.setting['local']}),
        Clustering.get_identifier_from_settings(
            {'data': run.setting['data'], 'local': run.setting['global']}),
    ) for run in exp_runs])

    config_identifier = set([(
        Clustering.get_identifier_from_settings(
            {'data': run.setting['data'], 'local': run.setting['local']}),
        Clustering.get_identifier_from_settings(
            {'data': run.setting['data'], 'local': run.setting['global']}),
    ) for run in config_runs])

    return run_identifier == config_identifier


def output_results(dbh, experiment_names, metrics, average_runs, output_format, results_folder, verbose):
    require_experiment_imports()

    metric_select_str = ','.join(
        [f"{metric.value} as {metric.name}" for metric in metrics])

    stmt = sa.text(
        f"SELECT name, {metric_select_str} FROM experiment_scores WHERE name IN ('{'\',\''.join(experiment_names)}')"
    )

    with dbh.sessionmaker() as session:
        res = session.execute(stmt)
        experiment_run_results = dict(zip(res.keys(), zip(*res)))

    experiment_run_results_df = pd.DataFrame(experiment_run_results)

    if average_runs:
        experiment_run_results_df = experiment_run_results_df.groupby(
            'name', as_index=False)
        if verbose:
            click.echo(experiment_run_results_df.describe())
        experiment_run_results_df = experiment_run_results_df.agg({
            metric.name.lower(): ['mean', 'std'] for metric in metrics
        })

    if output_format == 'latex':
        latex_str = experiment_run_results_df.to_latex(
            index=False,
            float_format="{:.2f}".format,
        )
        if results_folder is None:
            click.echo(latex_str)
        else:
            with open(results_folder.joinpath(f"results.tex"), 'w') as rfile:
                rfile.write(latex_str)
    elif output_format == 'json':
        if average_runs:
            experiment_run_results_df.set_index(('name', ''), inplace=True)
            results = json.loads(
                experiment_run_results_df.to_json(orient='index'))
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
    require_experiment_imports()
    manager = ExperimentManager()

    experiments = manager.get_experiments_by_name(experiment_config['name'])

    if len(experiments) == 0:
        raise ValueError("Experiment not in database.")
    elif len(experiments) > 1:
        raise ValueError(
            "Multiple experiments with given name found in database.")
    else:
        exp = next(iter(experiments.values()))

        clustering_identifier_list = (
            run.clustering.identifier for run in exp.runs if run.status == 'finished')

        m = hashlib.md5()
        for h in sorted(clustering_identifier_list):
            m.update(h.encode())
        return m.hexdigest()


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
        # check if the experiment in the database matches the expected experiment from config
        config_runs = get_experiment_runs_from_experiment_config(
            experiment_config)

        # check if number of runs is the same
        if len(config_runs) != len(exp.runs):
            click.echo("Differing numer of runs for experiment and config:")
            click.echo(f"Experiment runs: {len(exp.runs)}")
            click.echo(f"Config runs: {len(config_runs)}")
            config_matches = False
        else:
            config_matches = compare_run_configurations(exp.runs, config_runs)

    if not config_matches:
        raise ValueError(
            "Experiment config does not match with experiment in database.")

    return config_matches


class CorpusfileType(enum.Enum):
    CONLLU_VERB = enum.auto()
    GFNCSV = enum.auto()

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

    MICRO_F1 = "micro_f1__novelty_frame"

    PRECISION = "clusteringinfo->'evalresults'->'novelty_+_frame'->'macro avg'->'precision'"
    RECALL = "clusteringinfo->'evalresults'->'novelty_+_frame'->'macro avg'->'recall'"
    MACRO_F1 = "clusteringinfo->'evalresults'->'novelty_+_frame'->'macro avg'->'f1-score'"

    PURITY = "clusteringinfo->'evalresults'->'frame_induction_alleval'->'pu'"
    INV_PURITY = "clusteringinfo->'evalresults'->'frame_induction_alleval'->'ipu'"
    PURITY_F1 = "clusteringinfo->'evalresults'->'frame_induction_alleval'->'puf1'"

    BCUBED_P = "clusteringinfo->'evalresults'->'frame_induction_alleval'->'b^3p'"
    BCUBED_R = "clusteringinfo->'evalresults'->'frame_induction_alleval'->'b^3r'"
    BCUBED_F1 = "clusteringinfo->'evalresults'->'frame_induction_alleval'->'b^3f1'"

    # _ta exists as well
    SILHOUETTE = "clusteringinfo->'mean_silhouette_score_by_local_cluster'"
