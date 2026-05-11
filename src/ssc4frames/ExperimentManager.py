#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: remstef
"""

import traceback
from functools import partial
import itertools as it
import sqlalchemy as sa
import ssc4frames.loghelper as loghelper
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, ALL_COMPLETED
from multiprocessing import Lock
from ssc4frames.database import Experiment, ExperimentRun, DBHandler
import ssc4frames.run_experiment_db_only as runexp
import logging
import json
from typing import Iterable, Tuple, List
from random import shuffle
from ssc4frames.helpers import get_dburl_from_env

threadLocal = threading.local()

def static_init_logger(cls):
  cls.cls_logger = loghelper.setup_logger(cls.__name__) 
  return cls


class DbExperimentRunLogHandler(logging.Handler):
    '''
    Customized logging handler that puts logs to the database.
    '''
    def __init__(self, dbh:DBHandler, run_id:int):
      logging.Handler.__init__(self)
      self._dbh = dbh
      self._run_id = run_id

    def add_logentry(self, logmessage) -> None:
      # UPDATE table SET array_field = array_append(array_field,'new item') WHERE
      with self._dbh.sessionmaker() as session:
        session.execute(
        sa.update(ExperimentRun)
            .where(ExperimentRun.id==self._run_id)
            .values(logs=sa.text(f'array_append({ExperimentRun.logs.name}, :newlogentry)')),
          {'newlogentry': logmessage}
        )
      return

    def emit(self, record) -> None:
      msg = self.format(record)
      self.add_logentry(msg)
      return


@static_init_logger
class ExperimentWorker(object):
  '''
    Thread- or process pool worker 
  '''

  lock = None

  def __init__(self, dbh:DBHandler) -> None:
    self._dbh = dbh

  def get_logger_name_for_experiment_run(self, runid:int):
    return f'{self.cls_logger.name}.experimentrun{runid}'

  def setup_logger_for_experiment_run(self, runid:int) -> logging.Logger:
    db_log_handler = DbExperimentRunLogHandler(self._dbh, runid)
    db_log_handler.setFormatter(self.cls_logger.handlers[0].formatter)
    # 2 options: 
    # 1) add the handler to the current logger
    # 2) create a new separate logger for the clustering
    # prefer option 2) because of possible parallism issues with option 1)
    # base_logger.addHandler(db_log_handler) # <- this would be option 1)
    # this is option 2) ->:
    e_logger = logging.getLogger(self.get_logger_name_for_experiment_run(runid))
    e_logger.addHandler(db_log_handler)
    e_logger.setLevel(self.cls_logger.getEffectiveLevel())
    return e_logger
  
  def cleanup_logger_for_experiment_run(self, e_logger:logging.Logger, runid:int):
    assert self.get_logger_name_for_experiment_run(runid) == e_logger.name
    dbloghandlers = filter(lambda h: isinstance(h, DbExperimentRunLogHandler), e_logger.handlers)
    for dbloghandler in dbloghandlers:
      e_logger.removeHandler(dbloghandler)
    # not sure if this is necessary, but better be safe than sorry
    del logging.Logger.manager.loggerDict[e_logger.name]

  def update_experiment_run_values(self, run_id:int, **values) -> ExperimentRun:
    with self._dbh.sessionmaker() as session:
      session.execute(
        sa.update(ExperimentRun)
          .where(ExperimentRun.id == run_id)
          .values(**values)
      )
      res = session.get(ExperimentRun, run_id, options=(sa.orm.undefer_group('extrainfos'),))
    return res
  
  def refresh_experiment_run(self, run_id:int) -> ExperimentRun:
    with self._dbh.sessionmaker() as session:
      run = session.get(ExperimentRun, run_id, options=(sa.orm.undefer_group('extrainfos'),))
    return run


  def run_clustering(self, setting, logger, await_key_confirmation=False) -> dict:
    if isinstance(setting, dict) and not type(setting) == dict:
      setting = json.loads(json.dumps(setting))
    merged_settings = runexp.merge_with_default_params(setting)
    return runexp.run_with_params_DB(
      dbh=self._dbh,
      params__=merged_settings, 
      await_key_confirmation=await_key_confirmation, 
      logger=logger,
      lock=self.lock
    )
      
  def execute(self, run:ExperimentRun, raise_exception=False) -> ExperimentRun:
    run = self.refresh_experiment_run(run.id)
    if run.status != 'created':
      self.cls_logger.info(f'Run {run.id} of experiment {run.experiment_id} ({run.experiment.name}) not in expected "created" state but in "{run.status}". Copy run or experiment to rerun this experiment run or reset the experiment / run.')
      return run
    self.cls_logger.info(f'Running experiment {run.experiment_id} ({run.experiment.name}) run {run.id}.')
    # update run status
    run = self.update_experiment_run_values(run.id, status='started')
    e_logger = self.setup_logger_for_experiment_run(run.id)
    e_logger.info(f'Starting run {run.id} for experiment {run.experiment.name} ({run.experiment.id})')
    try:
      clustering_result = self.run_clustering(run.setting, e_logger)
      # update run {run.id}
      run = self.update_experiment_run_values(run.id, 
        status='finished',
        clustering_id=clustering_result['global']['clusteringid'],
        extrainfo=( run.extrainfo if run.extrainfo is not None else { } ) | { 'result': clustering_result }
      )
      e_logger.info(f'Sucessfully finished run {run.id} for experiment {run.experiment.id} ({run.experiment.name})')
    except BaseException as e:
      # update run {run.id}
      run = self.update_experiment_run_values(run.id, 
        status='failed',
        extrainfo=( run.extrainfo if run.extrainfo is not None else { } ) | { 
          'exception': {
            'name': e.__class__.__name__,
            'message': str(e),
            'traceback': traceback.format_exc()
        }}
      )
      e_logger.exception(e)
      e_logger.error(f'Run {run.id} for experiment {run.experiment.id} ({run.experiment.name}) failed.')
      if raise_exception:
        raise
    self.cleanup_logger_for_experiment_run(e_logger, run.id)
    return run

  @classmethod
  def submit_to_worker(cls, run:ExperimentRun, raise_exception=False) -> ExperimentRun:
    worker:ExperimentWorker = cls.get_worker()
    result = worker.execute(run, raise_exception=raise_exception)
    return result

  @classmethod
  def init_worker(cls, dburl:str, lock:Lock=None):
    cls.cls_logger.info(f'Using DB: {dburl}')
    if getattr(threadLocal, 'worker', None) is not None:
      raise LookupError('Global worker should be initialized only once!')
    dbh = runexp.setup_database_handler(dburl)
    threadLocal.worker = ExperimentWorker(dbh)
    threadLocal.worker.lock = lock
    return threadLocal.worker
  
  @classmethod
  def get_worker(cls):
    if getattr(threadLocal, 'worker', None) is None:
      raise LookupError(f'Global worker has not yet been initialized! Please run {cls.__name__}.init_worker(..) first.')
    return threadLocal.worker


@static_init_logger
class ExperimentManager(object):
  
  def __init__(self) -> None:
    self._dburl = get_dburl_from_env()
    self.cls_logger.info(f'DB URL: {self._dburl}')
    self._dbh = runexp.setup_database_handler(self._dburl)

  def new_experiment(self, name=None, commit=True) -> Experiment:
    # Experiment fields: id name status extrainfo runs
    new_experiment = Experiment(name=name)
    if commit:
      return self.add_experiment(new_experiment)
    return new_experiment

  def create_experiment_from_setting(self, setting:dict) -> Experiment:
    return self.add_experiment(
      # Experiment fields: id name status extrainfo runs
      Experiment(name='Unnamed', runs = [
        # ExperimentRun fields: id experiment_id/experiment setting status require clustering_id extrainfo logs
        ExperimentRun(setting=setting)
      ])
    )

  def add_experiment(self, new_experiment:Experiment) -> Experiment:
    '''
      add a new experiment without an id
    '''
    with self._dbh.sessionmaker() as session:
      session.add(new_experiment)
      session.commit()
      session.refresh(new_experiment)
    return new_experiment
  
  def add_experiment_run(self, new_experiment_run:ExperimentRun) -> ExperimentRun:
    '''
      add a new experiment run without an id
    '''
    with self._dbh.sessionmaker() as session:
      session.add(new_experiment_run)
      session.commit()
      session.refresh(new_experiment_run)
    return new_experiment_run

  def update_experiment_values(self, experiment_id:int, **values) -> Experiment:
    with self._dbh.sessionmaker() as session:
      session.execute(
        sa.update(Experiment)
          .where(Experiment.id == experiment_id)
          .values(**values)
      )
      res = session.get(Experiment, experiment_id, options=(sa.orm.undefer_group('extrainfos'), sa.orm.joinedload(Experiment.runs).undefer_group('extrainfos')))
    return res
  
  def update_experiment_run_values(self, experiment_run_id:int, **values) -> ExperimentRun:
    with self._dbh.sessionmaker() as session:
      session.execute(
        sa.update(ExperimentRun)
          .where(ExperimentRun.id == experiment_run_id)
          .values(**values)
      )
      res = session.get(ExperimentRun, experiment_run_id, options=(sa.orm.undefer_group('extrainfos'),))
    return res

  def delete(self, exp:Experiment) -> None:
    with self._dbh.sessionmaker() as session:
      for run in exp.runs:
        session.execute(sa.delete(ExperimentRun).where(ExperimentRun.id == run.id)) 
      session.execute(sa.delete(Experiment).where(Experiment.id == exp.id)) 
    return

  def refresh_experiment(self, experiment:Experiment, sa_options:tuple=(sa.orm.undefer_group('extrainfos'), sa.orm.joinedload(Experiment.runs).undefer_group('extrainfos'))) -> Experiment:
    with self._dbh.sessionmaker() as session:
      return session.get(Experiment, experiment.id, options=sa_options)

  def get_experiments_by_name(self, name:str=None) -> Iterable[Experiment]:
    with self._dbh.sessionmaker() as session:
      res = session.execute(
        sa.select(Experiment).where(Experiment.name == name)
      ).unique().scalars()
      experiments = res.all()
    return {e.id: e for e in experiments}

  def get_experiment_by_id(self, id:int, sa_options:tuple=(sa.orm.undefer_group('extrainfos'), sa.orm.joinedload(Experiment.runs).undefer_group('extrainfos'))) -> Experiment:
    with self._dbh.sessionmaker() as session:
      exp = session.get(Experiment, id, options=sa_options)
    return exp
  
  def get_experiment_run_by_id(self, id:int, sa_options:tuple=(sa.orm.undefer_group('extrainfos'),)) -> ExperimentRun:
    with self._dbh.sessionmaker() as session:
      erun = session.get(ExperimentRun, id, options=sa_options)
    return erun
  
  def reset_experiment(self, exp:Experiment) -> Experiment:
    '''
    resets the experiment's run's status'
    '''
    with self._dbh.sessionmaker() as session:
      exp = session.get(Experiment, exp.id)
      for erun in exp.runs:
        self.reset_experiment_run(experiment_run=erun)
      exp = session.get(Experiment, exp.id, options=(sa.orm.undefer_group('extrainfos'), sa.orm.joinedload(Experiment.runs).undefer_group('extrainfos')))
    return exp
  
  def copy_experiment(self, exp:Experiment) -> Experiment:
    '''
    copy experiment and its expriment runs
    '''
    new_experiment = Experiment(
      name=exp.name,
      extrainfo=exp.extrainfo,
      runs=[ ExperimentRun(
        setting=run.setting,
        require=run.require,
        status='created',
      ) for run in exp.runs ]
    )      
    return self.add_experiment(new_experiment)

  
  def reset_experiment_run(self, experiment_run:ExperimentRun) -> ExperimentRun:
    '''
    reset status of the experiment run to created and remove all infos, keep 'setting', 'experiment_id' and 'require'
    '''
    # ExperimentRun fields: id experiment_id/experiment setting status require clustering_id extrainfo logs
    return self.update_experiment_run_values(
      experiment_run_id=experiment_run.id,
      status='created',
      clustering_id=None,
      extrainfo={},
      logs=[],
    )
  
  def copy_experiment_run(self, er:ExperimentRun, to_experiment:Experiment=None) -> ExperimentRun:
    '''
    copy the experiment run, i.e. its settings. Move the experiment run to another 'to_experiment' if provided
    '''
    new_experiment_run = ExperimentRun(
      experiment_id=to_experiment.id if to_experiment is not None else er.experiment_id,  
      require=er.require, 
      setting=er.setting, 
      status='created'
    )
    return self.add_experiment_run(new_experiment_run)
  
  def list_experiments(self):
    with self._dbh.sessionmaker() as session:
      res = session.execute(
        sa.select(Experiment)
      ).unique().scalars()
      experiments = res.all()
    return {e.id: e for e in experiments}

  def run_with_setting(self, setting:dict, same_thread=True, new_process=False, raise_worker_exception=False) -> Experiment:
    # 1. create new Experiment and new ExperimentRun
    # 2. execute run
    new_experiment = self.create_experiment_from_setting(setting=setting)
    # run in main process, main thread or start a new process/thread?
    if same_thread:
      ExperimentWorker(self._dbh).execute(new_experiment.runs[0], raise_exception=raise_worker_exception)
      new_experiment_updated = self.refresh_experiment(new_experiment)
    else:
      new_experiment_updated = self.run_experiment_parallel(new_experiment, n_workers=1, process_pool=new_process, raise_worker_exception=raise_worker_exception)
    return new_experiment_updated

  def run_sequential(self, runs: List[ExperimentRun]|Tuple[ExperimentRun], same_thread=True, new_process=False, raise_worker_exception=False) -> Experiment:
    # run in main process, main thread or start a new process/thread?
    if same_thread:
      worker = ExperimentWorker(self._dbh)
      results = [ worker.execute(run, raise_exception=raise_worker_exception) for run in runs ]
    else:
      results = self.run_parallel(runs, n_workers=1, process_pool=new_process, raise_worker_exception=raise_worker_exception)
    return results

  def run_parallel(self, runs: List[ExperimentRun]|Tuple[ExperimentRun], n_workers=8, process_pool:bool=False, raise_worker_exception=False) -> List[ExperimentRun]:
    PoolClazz = ProcessPoolExecutor if process_pool else ThreadPoolExecutor
    self.cls_logger.info(f'Using executor type: {PoolClazz.__name__} with {n_workers} workers.')

    lock = Lock()

    with PoolClazz(max_workers=n_workers, initializer=ExperimentWorker.init_worker, initargs=(self._dburl, lock,)) as pool:
      results = pool.map(partial(ExperimentWorker.submit_to_worker, raise_exception=raise_worker_exception), runs, timeout=None, chunksize=1)
    total, success = 0, 0
    for i,r in enumerate(results):
      total += 1
      if r.status != 'finished':
        if r.status == 'failed':
          self.cls_logger.info(f'''Run {r.id} failed: {r.extrainfo['exception']['name']}.''')
        else:
          self.cls_logger.error(f'''Run {r.id} of experiment finished unsucessfully with status: {r.status}.''')
        continue
      success += 1
    self.cls_logger.info(f'Successfully executed {success} of {total} experiment runs.')
    return results

  def run_experiment_sequential(self, exp:Experiment, same_thread=True, new_process=False, raise_worker_exception=False) -> Experiment:
    self.run_sequential(exp.runs, same_thread=same_thread, new_process=new_process, raise_worker_exception=raise_worker_exception)
    experiment_updated = self.refresh_experiment(exp)
    return experiment_updated

  def run_experiment_parallel(self, exp: Experiment, n_workers=8, process_pool:bool=False, raise_worker_exception=False, shuffle_runs=True) -> Experiment:
    list_of_experimentruns = exp.runs
    if shuffle_runs:
      shuffle(list_of_experimentruns)
    self.run_parallel(list_of_experimentruns, n_workers=n_workers, process_pool=process_pool, raise_worker_exception=raise_worker_exception)
    exp = self.refresh_experiment(exp)
    return exp
