import sys
import click
from ssc4frames.cli.main import main
from ssc4frames.cli.helpers import get_dburl, JsonOption, example_clustering_config_override, merge_with_default_params

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

# CLustering Group


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
    manager.run_with_setting(clustering_config, same_thread=True, new_process=False,
                             raise_worker_exception=True, await_key_confirmation=(not no_wait))

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
            clusteringident_local = clustering.setting['global']['localclustering'].split(
                '@')[0].split('[')[0]
            clustering_local = session.execute(sa.select(Clustering).where(
                Clustering.identifier == clusteringident_local)).scalar_one_or_none()
            if clustering_local is None:
                raise KeyError(
                    f'Local clustering for global clustering {cid} not found')
            cid_local = clustering_local.id
        else:
            cid_local = cid

        if embeddings:
            # join result set with averaged embeddings i.e. just use the prepared frameinstances_split_vectorized__<local-cluster-id> view, we need to get the respective local clustering (if the current clustering is not a local clustering)
            viewname = f'frameinstances_split_vectorized__{cid_local}'
            allinstances_query = allinstances_query.replace(
                'frameinstances_split', viewname)
            # check if view exists and if not create a non-materialized one
            from sqlalchemy import inspect
            inspector = inspect(session.get_bind())
            views = inspector.get_view_names() + inspector.get_materialized_view_names()
            if viewname not in views:
                # create the view
                datadict = clustering.setting['data']
                datadict['materialize'] = False
                raise IndexError(
                    f'View for local clustering does not exist, consider (re-)creating it.')
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
                print('\t'.join(map(str, res.keys())))
            for r in res:
                print('\t'.join(map(str, r)))
                next_offset += 1
            if next_offset == current_offset:  # no change, no more rows to fetch
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
    dsnames = [name for name in datasetsplitname]

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
        print('\t'.join(map(str, res.keys())))
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
        print('\t'.join(map(str, res.keys())))
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
                print('\t'.join(map(str, res.keys())))
            for r in res:
                print('\t'.join(map(str, r)))
                next_offset += 1
            if next_offset == current_offset:  # no change, no more rows to fetch
                break
            current_offset = next_offset

    return
