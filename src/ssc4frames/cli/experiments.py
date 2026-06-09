import pathlib
import click
import json
from ssc4frames.cli.main import main
from ssc4frames.cli.helpers import get_dburl_from_env, get_experiment_runs_from_experiment_config, compare_run_configurations, Metrics, output_results

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


@main.group()
@click.option('--experiment_folder', type=click.Path(exists=True), required=True)
@click.pass_context
def experiments(ctx, experiment_folder):
    require_experiment_imports()
    ctx.ensure_object(dict)

    import json

    ctx.obj['EXPERIMENT_FOLDER'] = experiment_folder
    ctx.obj['EXPERIMENT_FILES'] = [f for f in pathlib.Path(
        experiment_folder).glob('*.json') if f.is_file()]
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

        config_runs = get_experiment_runs_from_experiment_config(
            json.loads(exp_config_file.read_text(encoding="UTF-8")))
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
            experiment_info['configuration_matches'] = compare_run_configurations(
                exp.runs, config_runs)

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

    output_results(dbh, ctx.obj['EXPERIMENT_NAMES'], metrics,
                   average_runs, output_format, results_folder, verbose)
