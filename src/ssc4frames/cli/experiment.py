import sys
import pathlib
import click
from click_option_group import optgroup
import json
from ssc4frames.cli.main import main
from ssc4frames.cli.helpers import JsonOption, get_dburl_from_env, get_experiment_hash, get_experiment_runs_from_experiment_config, compare_run_configurations, Metrics, exp_config_equals_db_config, output_results

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

# Experiment Group


@main.group()
# , default=example_experiment_config_with_hyperparameter_exchange)
@click.option('--experiment_config', type=JsonOption())
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

        hyperparameters = set([tuple(params['key'])
                              for params in experiment_config['hyperparameters']])
        database = get_dburl_from_env()

        from ssc4frames.cli.helpers import get_hash_for_datasetsplit, get_hash_for_embeddings
        from pathlib import Path

        if experiment_hash:

            experiment_hash_db = get_experiment_hash(experiment_config)

            if Path(experiment_hash).is_file():
                with open(experiment_hash, "r") as f:
                    experiment_hash = f.read().strip()

            if experiment_hash != experiment_hash_db:
                print(experiment_hash)
                print(experiment_hash_db)
                raise ValueError('Experiment runs are different than expected')

        if datasetsplit_hash:
            # check that datasetsplit is the same for each run (i.e. not part of hyperparameters)
            assert (('data', 'dataset') not in hyperparameters)

            datasetsplit = experiment_config['base_run_settings']['data']['dataset']
            datasetsplit_hash_db = str(
                get_hash_for_datasetsplit(datasetsplit, database))

            if Path(datasetsplit_hash).is_file():
                with open(datasetsplit_hash, "r") as f:
                    datasetsplit_hash = f.read().strip()

            if datasetsplit_hash != datasetsplit_hash_db:
                raise ValueError('Datasetsplit is different than expected')

        if embeddings_hash or embeddings_masked_hash:

            # check that embeddingmodel is the same for each run (i.e. not part of hyperparameters)
            assert (('local', 'emmmodel') not in hyperparameters)

            emmodel = experiment_config['base_run_settings']['local']['emmodel']

            if embeddings_hash:
                embeddings_hash_db = str(get_hash_for_embeddings(
                    datasetsplit, emmodel, database))

                if Path(embeddings_hash).is_file():
                    with open(embeddings_hash, "r") as f:
                        embeddings_hash = f.read().strip()

                if embeddings_hash != embeddings_hash_db:
                    raise ValueError('Embeddings are different than expected')

            if embeddings_masked_hash:

                embeddings_hash_db = str(get_hash_for_embeddings(
                    datasetsplit, emmodel + '-masked', database))

                if Path(embeddings_masked_hash).is_file():
                    with open(embeddings_masked_hash, "r") as f:
                        embeddings_hash = f.read().strip()

                if embeddings_hash != embeddings_hash_db:
                    raise ValueError(
                        'Masked embeddings are different than expected')


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

        experiment_runs = get_experiment_runs_from_experiment_config(
            experiment_config)

        # create experiment
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
    manager.run_experiment_parallel(
        experiment, n_workers=n_workers, process_pool=(not use_thread_pool))
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

    # check if the experiment in the database matches the expected experiment from config
    experiment_config = ctx.obj['EXPERIMENT_CONFIG']
    if experiment_config is None:
        click.echo(
            "Omitting config santity check. Provide --config if sanity check is desired.")
    else:
        config_runs = get_experiment_runs_from_experiment_config(
            experiment_config)

        # check if number of runs is the same
        if len(config_runs) != len(exp.runs):
            click.echo("Differing numer of runs for experiment and config:")
            click.echo(f"Experiment runs: {len(exp.runs)}")
            click.echo(f"Config runs: {len(config_runs)}")
        else:
            # check if the set of identifiers is the same
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
        local_clustering_durations = clustering_durations_df.drop_duplicates(
            subset=['clustering_duration_local', 'local_cid']).clustering_duration_local
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
            click.echo(
                f"Multiple experiments with given name found in database. (name: {experiment_name})")
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
def best_hyperparameters(ctx, metrics: Metrics, n, verbose,
                         output_experiment_name, output_experiment_file, extrainfo_note, train_splits, test_splits, skip_random_seeds, random_seeds):

    # get best hyperparameters for an experiment that has been run regarding specific metric

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

    # Warn if not all experiment runs have status finished
    if exp.get_status() != str(set(['finished'])):
        click.echo('Not all runs have been finished.')

    dburl = get_dburl_from_env()
    dbh = runexp.setup_database_handler(dburl)

    metric_select_str = ','.join(
        [f"{metric.value} as score_{metric.name}, RANK() over (PARTITION BY experiment_id ORDER BY {metric.value} DESC) rank_{metric.name}" for metric in metrics])
    metric_average_rank = '(' + '+'.join(
        [f"rank_{metric.name}" for metric in metrics]) + ')' + f'/{len(metrics)}.0 as avg_rank'

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
        click.echo(
            f'Returned {len(clustering_info_settings["rank_number"])} settings because of ties')
    settings = []
    for i in range(len(clustering_info_settings["rank_number"])):

        setting_dict = {}
        setting_dict['Rank'] = clustering_info_settings["rank_number"][i]
        setting_dict['cid'] = clustering_info_settings["clustering_id"][i]
        setting_dict['local_cid'] = clustering_info_settings["local_cid"][i]
        for metric in metrics:
            setting_dict[metric.name] = clustering_info_settings[
                f"score_{metric.name.lower()}"][i]
            setting_dict[f"rank_{metric.name}"] = clustering_info_settings[f"rank_{metric.name.lower()}"][i]

        if verbose:
            click.echo(
                f'Clustering {i+1}, Rank: {clustering_info_settings["rank_number"][i]}')

            # output chosen metrics
            for metric in metrics:
                click.echo(f'{metric.name}: {clustering_info_settings[f"score_{metric.name.lower()}"][i]}, Rank: {
                           clustering_info_settings[f"rank_{metric.name.lower()}"][i]}')

            # output other metrics
            click.echo(
                f"Frame identification: {clustering_info_settings['clusteringinfo'][i]['evalresults']['novelty_+_frame']['micro avg']}")
            click.echo(
                f"Frame induction: {clustering_info_settings['clusteringinfo'][i]['evalresults']['frame_induction_alleval']}")

        # get settings fo all hyperparameters
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

        # get best settings - handling ties
        min_rank = settings_df['Rank'].min()
        non_hp_keys = ['Rank', 'cid', 'local_cid']
        non_hp_keys.extend([key_name for metric in metrics for key_name in [
                           f'{metric.name}', f'rank_{metric.name}']])

        best_settings = (
            settings_df[settings_df['Rank'] == min_rank]
            .sort_values(by=settings_df.columns.tolist(), kind="mergesort")[[c for c in settings_df.columns if c not in non_hp_keys]]
            .iloc[0]
        )

        if output_experiment_name:
            experiment_config['name'] = output_experiment_name
        experiment_config['extrainfo']['note'] = extrainfo_note
        experiment_config['hyperparameters'] = [{'key': json.loads(
            key), 'values': [value]} for key, value in best_settings.items()]

        if not skip_random_seeds:

            # don't add random seeds for specified (deterministic) local clusterers
            if experiment_config['base_run_settings']['local']['clusterer']['type'] not in {'ident', 'const', 'forward', 'ha'}:
                experiment_config['hyperparameters'].append(
                    {'key': ['local', 'clusterer', 'options',
                             'random_state'], 'values': list(random_seeds)}
                )

            experiment_config['hyperparameters'].append(
                {'key': ['global', 'clusterer', 'options',
                         'random_state'], 'values': list(random_seeds)}
            )

        experiment_config['base_run_settings']['data']['splits'] = list(
            train_splits + test_splits)
        experiment_config['base_run_settings']['data']['testsplits'] = list(
            test_splits)

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

        json.dump(experiment_config, output_experiment_file,
                  indent=4, cls=NpEncoder)


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
            exp_config_equals_db_config(
                ctx.obj['EXPERIMENT_CONFIG'], experiments)
        else:
            click.echo(
                'To Check config runs please provide the --experiment_config parameter!')
            return

    e = get_experiment_from_ctxobj(ctx.obj, manager)
    output_results(manager._dbh, [e.name], metrics,
                   average_runs, output_format, results_folder, verbose)


@experiment.command()
@click.pass_context
@click.option('--check_config_runs', is_flag=True)
@click.option('--no-config', 'omit_config', is_flag=True)
@click.option('--logs', 'show_logs', is_flag=False, flag_value=0, default=-1, type=int)
def inspect(ctx, check_config_runs, omit_config, show_logs):

    manager = ExperimentManager()

    if check_config_runs:
        if 'EXPERIMENT_CONFIG' in ctx.obj:
            ec = ctx.obj['EXPERIMENT_CONFIG']
            experiments = manager.get_experiments_by_name(ec['name'])
            exp_config_equals_db_config(
                ctx.obj['EXPERIMENT_CONFIG'], experiments)
        else:
            click.echo(
                'To Check config runs please provide the --experiment_config parameter!')
            return

    e = get_experiment_from_ctxobj(ctx.obj, manager)

    # refresh to get all values
    if show_logs >= 0:
        e = manager.refresh_experiment(e, sa_options=(
            sa.orm.undefer_group('extrainfos'),
            sa.orm.joinedload(Experiment.runs).undefer_group(
                'extrainfos').undefer_group('logs'),
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
                '\n...' if show_logs >= 0 and er.logs is not None and len(
                    er.logs) > show_logs else ''
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

    # reset run status for ( all | started | failed ) runs of the experiment

    manager = ExperimentManager()
    exp = get_experiment_from_ctxobj(ctx.obj, manager)
    click.echo(exp)

    if click.confirm(f'Reset {reset_type} experiment runs?'):

        click.echo(f"Resetting {reset_type} experiment runs.")
        if reset_type == 'all':
            manager.reset_experiment(exp)
        else:
            for erun in exp.runs:
                if erun.status == reset_type:
                    manager.reset_experiment_run(experiment_run=erun)
    else:
        click.echo('Aborted.')
