#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: remstef
"""

import abc
import os, sys
import torch
import numpy as np
import json
import re
from typing import override
from urllib.parse import urlparse, parse_qs, unquote
from pathlib import PurePosixPath
import ssc4frames.loghelper as loghelper

imported_libs_for_type = set()

def import_paradedb():
  global sqlalchemy, exists, MetaData, Table, Column, BigInteger, func, pginsert, Vector
  import sqlalchemy
  from sqlalchemy.sql import exists
  from sqlalchemy.sql.schema import MetaData
  from sqlalchemy import Table, Column, BigInteger, func
  from sqlalchemy.dialects.postgresql import insert as pginsert
  from pgvector.sqlalchemy import Vector

def import_modelextractor():
  global pd, EmbeddingsExtractor, load_embeddings, mean_pooling, pooling_strategies
  from ssc4frames.embeddings import EmbeddingsExtractor
  from ssc4frames.helpers import load_embeddings, mean_pooling, pooling_strategies
  import pandas as pd

def import_modelextractor_db():
  global FrameInstance, sessionmaker, sqlalchemy
  from ssc4frames.database import FrameInstance
  import sqlalchemy
  from sqlalchemy.orm import sessionmaker

def import_required_on_init(cls, import_func):
  if cls not in imported_libs_for_type:
    import_func()
    imported_libs_for_type.add(cls)

def static_init_logger(cls):
  cls.cls_logger = loghelper.setup_logger(cls.__name__) 
  return cls



@static_init_logger
class TensorStorage(metaclass=abc.ABCMeta):

  class Factory(object):

    @classmethod
    def get_clazz_from_url_scheme(cls, url):
      o = urlparse(url)
      return {
        'file' : FileBasedTensorStorage, 
        'paradedb' : ParadeDbTensorStorage,
        'model+df': ModelTensorStorageDF,
        'model+db': ModelTensorStorageDB
      }[o.scheme]
    
    @classmethod
    def fromurl(cls, url, **kwargs):
      return cls(url=url, **kwargs)

    def __init__(self, url, **params):
      self.tensorstorage_clazz = self.get_clazz_from_url_scheme(url)
      self.tensorstorage_init_args = params | self.tensorstorage_clazz.parse_args_from_url(url)
      self.tensorstorage_backoff_clazz = None
      self.tensorstorage_backoff_init_args = None
    
    def with_backoff(self, url, **params):
      self.has_backoff = True
      self.tensorstorage_backoff_clazz = self.get_clazz_from_url_scheme(url)
      self.tensorstorage_backoff_init_args = params | self.tensorstorage_backoff_clazz.parse_args_from_url(url)
      return self

    def get(self, **kwargs):
      print(self.tensorstorage_init_args)
      tsobj = self.tensorstorage_clazz(**self.tensorstorage_init_args)
      if self.tensorstorage_backoff_clazz is None:
        return tsobj
      tsobj_backoff = self.tensorstorage_backoff_clazz(**self.tensorstorage_backoff_init_args)
      return BackoffTensorStorage(
        main=tsobj,
        backoff=tsobj_backoff,
        **kwargs
      )

  @classmethod
  def fromurl(cls, url, **kwargs):
    return TensorStorage.Factory.fromurl(url=url, **kwargs)
  
  def __init__(self, device_embeddings='cpu', dtype_pt=torch.float32) -> None:
    self.device_embeddings = device_embeddings 
    self.dtype_pt = dtype_pt
    self.alive = False

  @classmethod
  def parse_args_from_url(cls, url):
    raise NotImplementedError()

  def ping(self):
    raise NotImplementedError()
  
  def ready(self):
    return self.alive

  def reconnect():
    raise NotImplementedError()

  def __getitem__(self, keys): 
    raise NotImplementedError()

  def __setitem__(self, keys, tensors): 
    raise NotImplementedError()
  
  def __delitem__(self, keys):
    raise NotImplementedError()
  
  def __contains__(self, keys):
    raise NotImplementedError()

  def __len__(self): 
    raise NotImplementedError()

  def loadData(self, df, batchsize, modelname, modeldevice, masking, pk='c'): # pk = primary key = unique key per embedding
    global EmbeddingsExtractor, load_embeddings, mean_pooling
    from ssc4frames.embeddings import EmbeddingsExtractor
    from ssc4frames.helpers import load_embeddings, mean_pooling

    if len(self) == df.shape[0]:
      self.__class__.cls_logger.info(f"Data has been populated already: '{len(self)} == {df.shape[0]}.")      
      return self
    if len(self) > df.shape[0]:
      self.__class__.cls_logger.warning(f"Data mismatch: 'Dataframe: {df.shape[0]} != Database (embeddings): {len(self)}.")
      if not input('Continue using embeddings? Y/N').strip().upper() == 'Y':
        sys.exit()
      else:
        return self
    if len(self) > 0 and len(self) < df.shape[0]:
      self.__class__.cls_logger.warning(f"Data mismatch: 'Dataframe: {df.shape[0]} != Database (embeddings): {len(self)}.")
      if not input('Continue importing embeddings? Y/N').strip().upper() == 'Y':
        sys.exit()

    # LOAD!!!
    self.__class__.cls_logger.info(f"Loading...")
    extractor = EmbeddingsExtractor(
      batch_size=0, # use own batching below
      modelname=modelname, 
      device=modeldevice,
      device_store='cpu',
      pooling=mean_pooling,
      masking=masking
    )
    
    # prepare batches
    n_batches = df.shape[0] // batchsize
    n_rest = df.shape[0] % batchsize
    self.__class__.cls_logger.info(f'batchsize: {batchsize}, #batches: {n_batches}, rest: {n_rest}')
    
    # start batch import
    for batch_i in range(n_batches):
      b = batch_i * batchsize
      e = b + batchsize
      if int(df.iloc[e-1][pk]) in self:
        self.__class__.cls_logger.info(f'Batch {batch_i+1}/{(n_batches + (1 if n_rest > 0 else 0))}: [{b},{e}[ batch already processed')
        continue
      subdf = df.iloc[b:e]
      self.__class__.cls_logger.info(f'Batch {batch_i+1}/{(n_batches + (1 if n_rest > 0 else 0))}: [{b},{e}[ insert ids [{subdf.iloc[0][pk]}, {subdf.iloc[-1][pk]}]')
      luembeddings = extractor.transform(subdf)['luembeddings']
      self[subdf[pk].tolist()] = luembeddings
        
    # last batch
    if n_rest > 0:
      b = n_batches * batchsize
      e = b + n_rest
      if not int(df.iloc[e-1][pk]) in self:
        subdf = df.iloc[b:e]
        self.__class__.cls_logger.info(f'Batch {(n_batches + 1)}/{(n_batches + 1)}: [{b},{e}[ insert ids [{subdf.iloc[0][pk]}, {subdf.iloc[-1][pk]}]')
        luembeddings = extractor.transform(subdf)['luembeddings']
        self[subdf[pk].tolist()] = luembeddings
      else:
        self.__class__.cls_logger.info(f'Batch {(n_batches + 1)}/{(n_batches + 1)}: [{b},{e}[ batch already processed.')
    
    return self
  
  @classmethod
  def copy(cls, from_store, to_store, keys, keymap, batchsize=2048, skip_batch_by_heuristic=True):
    '''
    from_store, to_store = TensorStorages
    keys = list like, generator or range
    keymap = function(key or keys) -> dict[key,new_key]
    '''
    if keys is None: # simply use range
      keys = range(len(from_store))
    # prepare batches
    numel = len(keys)
    n_batches = numel // batchsize
    n_rest = numel % batchsize
    cls.cls_logger.info(f'numel: {numel}, batchsize: {batchsize}, #batches: {n_batches}, rest: {n_rest}')
    
    # start batch copy
    for batch_i in range(n_batches):
      b = batch_i * batchsize
      e = b + batchsize
      keys_from = keys[b:e]
      keys_to = keymap(keys_from)

      if skip_batch_by_heuristic and keys_to[keys_from[-1]] in to_store:
        cls.cls_logger.info(f'Batch {batch_i+1}/{(n_batches + (1 if n_rest > 0 else 0))}: [{b},{e}[ batch already processed, ids [{keys_from[0]}=>{keys_to[keys_from[0]]}, {keys_from[-1]}=>{keys_to[keys_from[-1]]}]')
        continue
      cls.cls_logger.info(f'Batch {batch_i+1}/{(n_batches + (1 if n_rest > 0 else 0))}: [{b},{e}[ copying ids [{keys_from[0]}=>{keys_to[keys_from[0]]}, {keys_from[-1]}=>{keys_to[keys_from[-1]]}]')
      r_keys, r_vecs = from_store[keys_from]
      # map retrieved keys to the new keys
      k_keys = [keys_to[k] for k in r_keys]
      to_store[k_keys] = r_vecs
      
    # last batch
    if n_rest > 0:
      b = n_batches * batchsize
      e = b + n_rest
      keys_from = keys[b:e]
      keys_to = keymap(keys_from)
      if skip_batch_by_heuristic and keys_to[keys_from[-1]] in to_store:
        cls.cls_logger.info(f'Batch {(n_batches + 1)}/{(n_batches + 1)}: [{b},{e}[ batch already processed, ids [{keys_from[0]}=>{keys_to[keys_from[0]]}, {keys_from[-1]}=>{keys_to[keys_from[-1]]}]')
        return
      # else
      cls.cls_logger.info(f'Batch {(n_batches + 1)}/{(n_batches + 1)}: [{b},{e}[ copying ids [{keys_from[0]}=>{keys_to[keys_from[0]]}, {keys_from[-1]}=>{keys_to[keys_from[-1]]}]')
      r_keys, r_vecs = from_store[keys_from]
      # map retrieved keys to the new keys
      k_keys = [keys_to[k] for k in r_keys]
      to_store[k_keys] = r_vecs
    return 


@static_init_logger
class TensorBasedTensorStorage(TensorStorage):

  def __init__(self, embeddings, embeddings_index, device_embeddings='cpu', dtype_pt=torch.float32):

    super().__init__(device_embeddings=device_embeddings, dtype_pt=dtype_pt)
    self.embeddings = embeddings
    self.embeddings_index = embeddings_index

  def __getitem__(self, keys):
    if isinstance(keys, slice):
      start, stop, step = keys.indices(len(self))
      return range(start, stop, step), [self.embeddings[self.embeddings_index[i]] for i in range(start, stop, step)]
    elif isinstance(keys, tuple) or isinstance(keys, list):
      return keys, [self.embeddings[self.embeddings_index[i]] for i in keys]
    else:
      return keys, self.embeddings[self.embeddings_index[keys]]

  def __len__(self):
    if self.embeddings is not None:
      return self.embeddings.shape[0]
    else:
      return 0

  @override
  def ping(self):
    return True

  @override
  def ready(self):
    return True


@static_init_logger
class FileBasedTensorStorage(TensorBasedTensorStorage):

  name = 'file'

  def __init__(self, filename, load=True, save=True, overwrite=False, device_embeddings='cpu', dtype_pt=torch.float32):
    super().__init__(embeddings=None, embeddings_index=None,
                     device_embeddings=device_embeddings, dtype_pt=dtype_pt)
    self.load = load
    self.save = save 
    self.overwrite = overwrite
    self.filename = os.path.abspath(filename)
    self.filename_index = self.filename + '.index'
    if os.path.isfile(self.filename) and os.path.exists(self.filename):
      self.loadData(dataframe=None, batchsize=-1, modelname=None, modeldevice=None, masking=False)
    self.alive = True

  @classmethod
  def parse_args_from_url(cls, url):

    args = {}

    o = urlparse(url)
    args['filename'] = o.path[1:]

    return args

  @override
  def loadData(self, dataframe, batchsize, modelname, modeldevice, masking, pk='c'):
    
    global EmbeddingsExtractor, load_embeddings, mean_pooling
    from ssc4frames.embeddings import EmbeddingsExtractor
    from ssc4frames.helpers import load_embeddings, mean_pooling

    self.embeddings = load_embeddings(
        df=dataframe, 
        modelname=modelname, 
        masking=masking, 
        pooling=mean_pooling,
        device=modeldevice, 
        device_store=self.device_embeddings,
        filename=self.filename, 
        LOAD=self.load, 
        SAVE=self.save, 
        OVERWRITE=self.overwrite,
        batch_size=batchsize
    )
    saved = False
    if dataframe is not None:
      if os.path.exists(self.filename_index) and not self.overwrite:
        self.__class__.cls_logger.warn(f"Index file '{self.filename_index}' already exists and overwriting is not desired. \nSkipping this step.")
      else:
        self.embeddings_index = {value: index for index, value in enumerate(dataframe[pk].to_list())}

        if os.path.exists(self.filename_index):
          self.__class__.cls_logger.info(f"Embeddings file '{self.filename_index}' already exists.")
          if self.safe and self.overwrite:
            self.__class__.cls_logger.info(f"Saving lu embeddings index to '{self.filename_index}'.")
            with open(self.filename_index, 'w') as json_file:
              json.dump(sorted(self.embeddings_index.keys(), key=self.embeddings_index.get), json_file)
            saved = True
          else:
            self.__class__.cls_logger.info(f"Skip saving.")
        else:
          if self.save:
            self.__class__.cls_logger.info(f"Saving lu embeddings index to '{self.filename_index}'.")
            with open(self.filename_index, 'w') as json_file:
              json.dump(sorted(self.embeddings_index.keys(), key=self.embeddings_index.get), json_file)
            saved = True
    if self.load and os.path.exists(self.filename_index) and not saved:
      self.cls_logger.info(f"Loading embeddings index from '{self.filename_index}'.")
      with open(self.filename_index, 'r') as json_file:
        self.embeddings_index = {value: index for index, value in enumerate(json.load(json_file))}
    return self
  
  @override
  def ready(self):
    return self.embeddings is not None
  
  @override
  def __repr__(self) -> str:
    return f'''
      {super().__repr__()}
      filename: {self.filename}
      LOAD: {self.load}
      SAVE: {self.save}
      OVERWRITE: {self.overwrite}
      file exists: {os.path.isfile(self.filename) and os.path.exists(self.filename)}
      ping: {self.ping()}
      alive: {self.alive}
      ready: {self.ready()}
      size: {len(self) if self.ready() else 'N/A'}
    '''


@static_init_logger
class WeightedTensorStorage(TensorStorage):

  name = 'weighted'

  def __init__(self, store1: TensorStorage, store2: TensorStorage, alpha: float=.5):
    super().__init__(device_embeddings=None, dtype_pt=None)
    self.store1 = store1
    self.store2 = store2
    self.alpha = alpha
    self.alive = True

  @override
  def __getitem__(self, keys):
    keys1, embeddings1 = self.store1[keys]
    keys2, embeddings2 = self.store2[keys]

    ## assure that both stores return the same vectors
    assert set(keys1) == set(keys2)

    if isinstance(embeddings1, list):
      embeddings1 = torch.stack(embeddings1)
    if isinstance(embeddings2, list):
      embeddings2 = torch.stack(embeddings2)

    ## assure the same order for embeddings1 and embeddings2
    index_dict = {key: index for index, key in enumerate(keys1)}
    order = [index_dict[key] for key in keys2]

    return keys2, (1-self.alpha)*embeddings1[order,] + self.alpha*embeddings2

  @override
  def __len__(self):
    return len(self.store1)

  @override
  def loadData(self, *args, **kwargs):
    raise NotImplementedError('Please use the loadData method of the individual storages.')
  
  @override
  def ready(self):
    return self.store1.ready() and self.store2.ready()

  @override
  def __repr__(self):
    return f'''
      {super().__repr__()}
      ready: {self.ready()}
      store1: {self.store1}
      store2: {self.store2}
    '''


@static_init_logger
class BackoffTensorStorage(TensorStorage):

  name = 'backoff'

  def __init__(self, main: TensorStorage, backoff: TensorStorage, insert_on_miss: bool=True) -> TensorStorage:
    super().__init__(device_embeddings=None, dtype_pt=None)
    self.main = main
    self.backoff = backoff
    self.insert_on_miss = insert_on_miss 

  @override
  def __getitem__(self, keys):
    # get embeddings from main store
    r_keys, r_embeddings = self.main[keys]
    # check which keys have not been found
    key_misses = set(keys) - set(r_keys)
    if len(key_misses) <= 0: # if all keys have been found return them
      return r_keys, r_embeddings
    # if not, query the backoff store for the missing keys
    rb_keys, rb_embeddings = self.backoff[key_misses]
    if len(rb_keys) != len(key_misses):
      # identify the keys that could not be found by the store and by the backoff store and issue a warning
      key_strongmisses = key_misses - set(rb_keys)
      self.__class__.cls_logger.warning(f"Could neither find keys '{key_strongmisses}' in store '{self.main}' nor backoff store '{self.backoff}'.")
    if len(rb_keys) <= 0: # if none of the key_misses have been found, return the results from before
      return r_keys, r_embeddings
    # otherwise check if they should be inserted into the main store and insert them
    if self.insert_on_miss:
      self.main[rb_keys] = rb_embeddings
    # finally merge results by concatenation / stacking and return
    return r_keys+rb_keys, torch.cat((r_embeddings,rb_embeddings),dim=0)
    
  @override
  def __len__(self):
    # backoff store *should* contain more keys that the main store (since its a backoff so simply return the backoff store's size)
    return len(self.backoff)

  @override
  def loadData(self, *args, **kwargs):
    raise NotImplementedError('Please use the loadData method of the individual storages.')
  
  @override
  def ready(self):
    return self.main.ready() and self.backoff.ready()

  @override
  def __repr__(self):
    return f'''
      {super().__repr__()}
      store1: {self.main}
      store2: {self.backoff}
    '''

@static_init_logger
class ParadeDbTensorStorage(TensorStorage):

  name = 'paradedb'

  def __init__(self,
               datadbname, datatablename, username='root', password='root', hostname='localhost', port=5432,
               dim=768, device_embeddings='cpu', dtype_pt=torch.float32):
    import_required_on_init(self.__class__, import_paradedb)
    super().__init__(device_embeddings=device_embeddings, dtype_pt=dtype_pt)
    self.dim = dim
    self.prefer_upsert = True
    self.datadbname = datadbname
    self.datatablename = datatablename
    self.rootdbname = 'root'
    self.datadburl = f'postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{self.datadbname}'
    self.rootdburl = f'postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{self.rootdbname}'
    self._initdb()

 # example url: 'paradedb://user:pass@pgservername:pgserverport/datadbname/tablename?params#unusedotherinfo',
  @classmethod
  def parse_args_from_url(cls, url):

    args = {}

    urlsegments = urlparse(url, allow_fragments=True)
    queryargs = parse_qs(urlsegments.query, keep_blank_values=True)
    posixpath = PurePosixPath(unquote(urlsegments.path))
    # # parse dropifexists param
    # self.drop_db_if_exists = False
    # if 'dropdbifexists' in queryargs:
    #   self.drop_db_if_exists = re.search('(^$)|(^1$)|(^t[rue]{,3}$)|(^y[es]{,2}$)', queryargs['dropdbifexists'][0].lower()) is not None
    # self.drop_table_if_exists = False
    # if 'droptableifexists' in queryargs:
    #   self.drop_table_if_exists = re.search('(^$)|(^1$)|(^t[rue]{,3}$)|(^y[es]{,2}$)', queryargs['droptableifexists'][0].lower()) is not None
    # parse dim param
    args['dim'] = int(queryargs['dim'][0]) if 'dim' in queryargs else 768
    args['datadbname'] = posixpath.parts[1] # root == posixpath[0] == '/'
    args['datatablename'] = posixpath.parts[2] # root == posixpath[0] == '/'; dbname == posixpath[1]
    # setup postgres connection strings
    args['username'] = urlsegments.username if urlsegments.username is not None else 'root'
    args['password'] = urlsegments.password if urlsegments.password is not None else 'root'
    args['hostname'] = urlsegments.hostname if urlsegments.hostname is not None else 'localhost'
    args['port'] = urlsegments.port if urlsegments.port is not None else 5432

    return args
  
  @override
  def ping(self, rootdb=False):
    '''
    @see https://docs.sqlalchemy.org/en/20/core/pooling.html#custom-legacy-pessimistic-ping
    '''
    def nested_ping_on_connection(stmt, connection):
      try:
        # run a SELECT 1.   use a core select() so that
        # the SELECT of a scalar value without a table is
        # appropriately formatted for the backend
        return connection.scalar(stmt)
      except sqlalchemy.exc.DBAPIError as err:
        # catch SQLAlchemy's DBAPIError, which is a wrapper
        # for the DBAPI's exception.  It includes a .connection_invalidated
        # attribute which specifies if this connection is a "disconnect"
        # condition, which is based on inspection of the original exception
        # by the dialect in use.
        if err.connection_invalidated:
          # run the same SELECT again - the connection will re-validate
          # itself and establish a new connection.  The disconnect detection
          # here also causes the whole connection pool to be invalidated
          # so that all stale connections are discarded.
          return connection.scalar(stmt)
        else:
          raise

    engine = self.rootengine if rootdb else self.dataengine
    if engine is None:
      self.__class__.cls_logger.error(f"Connection to ParadeDB failed for some unknown reason. Please check the availability of the server (DB='{self.rootdbname if rootdb else self.datadbname}').")
      return False
    # prepare ping
    stmt = sqlalchemy.select(1)
    try:
      with engine.connect() as dbconnection:
        res = nested_ping_on_connection(stmt, dbconnection) # res=1 ideally
    except sqlalchemy.exc.DBAPIError as err:
      res = 0
    self.alive = res>0
    return self.alive
  
  @override
  def ready(self):
    return self.ping() ## AND DB + Table is initialized!!
  
  def _initdb(self):
    self.rootengine = sqlalchemy.create_engine(self.rootdburl, pool_pre_ping=True, pool_recycle=3600, isolation_level='AUTOCOMMIT', echo=False)
    self.dataengine = None
    
    metadata_obj = MetaData()
    self.tensor_table = Table(
      self.datatablename, metadata_obj,
      Column("key", BigInteger, primary_key=True, autoincrement=False),
      Column("embedding", Vector(self.dim))
    )

    def init_data_connection():
      self.dataengine = sqlalchemy.create_engine(self.datadburl, pool_pre_ping=True, pool_recycle=3600, isolation_level='AUTOCOMMIT', echo=False)
      
    def init_database():
      ret_val = 0
      try: 
        # with self.rootengine.connect() as rootconnection:
          # if self.drop_db_if_exists: # just drop, even if it didn't exist
          #   self.__class__.cls_logger.info(f"Dropping DB '{self.datadbname}' if exists.")
          #   if self.dataengine is not None:
          #     self.dataengine.dispose()
          #   rootconnection.execute(sqlalchemy.text(f'DROP DATABASE IF EXISTS "{self.datadbname}" WITH (FORCE);'))
          # check if DB exists
          # res = rootconnection.execute(sqlalchemy.text('SELECT 1 FROM pg_database WHERE datname=:dbname;'), {'dbname': self.datadbname})
          # if res.rowcount < 1:
          #   # self.__class__.cls_logger.info(f"Database '{self.datadbname}' does not exist and is beeing created.")
          #   # rootconnection.execute(sqlalchemy.text(f'CREATE DATABASE "{self.datadbname}";'))
          #   # ret_val = 1
          #   ret_val = -1
        if self.dataengine is None:
          init_data_connection()
      except sqlalchemy.exc.DBAPIError as err:
        self.__class__.cls_logger.error(f'Databases initialization failed: {err}')
        ret_val = -1
      self.alive = ret_val>=0
      return ret_val

    def init_tables():
      with self.dataengine.connect() as dataconnection:
        # if self.drop_table_if_exists:
        #   self.__class__.cls_logger.info(f"Dropping table '{self.datatablename}' in database '{self.datadbname}' if existent.")
        #   dataconnection.execute(sqlalchemy.text('DROP TABLE IF EXISTS ":tablename" CASCADE;'), {'tablename': self.datatablename})
        self.__class__.cls_logger.info(f"Creating table '{self.datatablename}' in database '{self.datadbname}' if not exists.")
        # create tables
        metadata_obj.create_all(bind=dataconnection, tables=[self.tensor_table], checkfirst=True)
      return 0

    nrows = 0 if ( init_database()<0 or init_tables()<0 ) else self._table_size()
    self.__class__.cls_logger.info(f"Database '{self.datadbname}'/'{self.datatablename}' #rows: {nrows}.")
    return nrows

  def _table_size(self):
    stmt = sqlalchemy.select(func.count('*')).select_from(self.tensor_table)
    with self.dataengine.connect() as dataconnection:
      count: int = dataconnection.execute(stmt).scalar()
      return count

  def set_dtype(self, dtype_pt):
    raise NotImplementedError('Not yet supported.')
  
  @override
  def __setitem__(self, keys, tensors):
    items = [{ 'key': key, 'embedding': tensors[i] } for i,key in enumerate(keys)]
    if self.prefer_upsert:
      stmt = pginsert(self.tensor_table).on_conflict_do_nothing(index_elements=['key'])
    else: # default insert (fails on duplicate pk entry)
      stmt = self.tensor_table.insert()
    # begin transaction (reverts to previous state if one of the actions fails)
    with self.dataengine.begin() as dataconnection:
        dataconnection.execute(stmt, items)
    return True

  @override
  def __getitem__(self, keys):
    # retrieve
    stmt = sqlalchemy.select(self.tensor_table).where(self.tensor_table.c.key.in_(keys))
    with self.dataengine.connect() as dataconnection:
      res = dataconnection.execute(stmt)
      rows = list(zip(*res))
      if len(rows) <= 0:
        return [], torch.empty((0,self.dim), dtype=self.dtype_pt, device=self.device_embeddings)
      arr = np.array(rows[1]) # 1 = embeddings, 0 = keys
      tensors = torch.as_tensor(arr, dtype=self.dtype_pt, device=self.device_embeddings)
      res_keys = list(rows[0])
    return res_keys, tensors
  
  @override
  def __delitem__(self, keys):
    stmt = sqlalchemy.delete(self.tensor_table).where(self.tensor_table.c.key.in_(keys))
    with self.dataengine.begin() as dataconnection:
      res = dataconnection.execute(stmt)
      num_deletions = res.rowcount
    return num_deletions
    
  @override
  def __contains__(self, key):
    stmt = sqlalchemy.select(exists().where(self.tensor_table.c.key == key))
    with self.dataengine.connect() as dataconnection:
      item_is_in_collection = dataconnection.execute(stmt).scalar()
    return item_is_in_collection

  @override
  def __len__(self):
    return self._table_size()
  
  def __del__(self):
    if self.rootengine is not None:
      self.rootengine.dispose()
    if self.dataengine is not None:
      self.dataengine.dispose()

  @override
  def __repr__(self):
    return f'''
      {super().__repr__()}
      connectionstrings: 
        - {self.rootdburl}
        - {self.datadburl}
      database: {self.datadbname}
      tablename: {self.datatablename}
      vectordim: {self.dim}
      alive: {"yes" if self.ready() else "no"}
      table: {self.datatablename}
      torch dtype: {self.dtype_pt}
      size: {len(self) if self.ready() else -1}
    '''
  

@static_init_logger
class ModelTensorStorage(TensorStorage):

  def __init__(self,
               modelid, databack, modeldevice, masked=True, mask_str=None, mask_subwords=False, pooling='mean',
               device_embeddings='cpu', dtype_pt=torch.float32):
    import_required_on_init(ModelTensorStorage.__class__, import_modelextractor)
    super().__init__(device_embeddings=device_embeddings, dtype_pt=dtype_pt)
    self.dim = -1
    self._model_ready = False
    self._data_ready = False
    self.sampledf = pd.DataFrame({ 
      'TOKENIZED_SENTENCE': [ 
        ['Ulrike', 'hat', 'sechs', 'Semester', 'Jura', 'studiert', ',', 'dann', 'hat', 'sie', 'sich', 'entschieden', ',', 'Fremdsprachenkorrespondentin', 'zu', 'werden', '.'],
        ['Wir', 'spielten', 'uns', 'vergnügt', 'den', 'Ball', 'zu', '.'],
        ['Paul', 'gibt', 'viel', 'Geld', 'für', 'seine', 'Hobbys', 'aus', '.'],
        ['Pat', 'spielte', 'den', 'Ball', 'zum', 'Torwart', '.'] 
      ],
      'LU_INDEX': [ [5], [1], [1], [1] ],
      'LU_INDEX_PART': [ [], [6], [7], [] ],
      'i': [ 1, 7, 23, 6 ],
      'c': [ 1, 2, 3, 4 ],
      'global_id': [
        "gfn::E-VALBU::framenet_des_deutschen_10_1282_1425_1::[5]::['Ulrike','hat','s...]::studieren::gfn::53",
        "gfn::FRAMENET::framenet_des_deutschen_100_103_106_9::[1]::['Wir','spielten',...]::zuspielen::gfn::531",
        "gfn::E-VALBU::framenet_des_deutschen_104_1138_2834_0::[1]::['Paul','gibt','vi...]::ausgeben::gfn::538",
        "gfn::FRAMENET::framenet_des_deutschen_100_103_106_8::[1]::['Pat','spielte','...]::spielen::gfn::531"
      ]})

    self.modelid = modelid
    self.masked = masked
    self.mask_str = mask_str
    self.mask_subwords = mask_subwords
    self.pooling = pooling
    self.databack = databack
    self.modeldevice = modeldevice

    self._init_model()

    
  #url='model+...://model_id?data=gfn&masked=n&device=cuda&data=...',
  @classmethod
  def parse_args_from_url(cls, url):
    args = {}

    urlsegments = urlparse(url, allow_fragments=True)
    queryargs = parse_qs(urlsegments.query, keep_blank_values=True)

    args['modelid'] = f'{urlsegments.hostname}{urlsegments.path.rstrip('/')}'
    # parse masked/unmasked param => if not masked then unmasked
    args['masked'] = True
    if 'masked' in queryargs:
      args['masked'] = re.search('(^$)|(^1$)|(^t[rue]{,3}$)|(^y[es]{,2}$)', queryargs['masked'][0].lower()) is not None

    if 'mask_str' in queryargs:
      args['mask_str'] = queryargs['mask_str'][0]

    args['mask_subwords'] = True
    if 'masksubwords' in queryargs:
      args['mask_subwords'] = re.search('(^$)|(^1$)|(^t[rue]{,3}$)|(^y[es]{,2}$)', queryargs['masksubwords'][0].lower()) is not None
    args['databack'] = queryargs['data'][0]
    args['modeldevice'] = queryargs['device'][0]

    if 'pooling' in queryargs:
      args['pooling'] = queryargs['pooling'][0]

    return args
  
  @override
  def ready(self):
    return self._data_ready and self._model_ready
  
  def _init_model(self):
    self.extractor = EmbeddingsExtractor(
      batch_size=0, # use own batching below
      modelname=self.modelid, 
      device=self.modeldevice,
      device_store=self.device_embeddings,
      pooling=pooling_strategies[self.pooling],
      masking=self.masked,
      mask_str=self.mask_str,
      mask_subwords=self.mask_subwords
    )
    luembeddings = self.extractor.transform(self.sampledf)['luembeddings']
    self.dim = luembeddings.size(1)
    self._model_ready = True
  
  @override
  def __getitem__(self, keys):
    subdf = self._get_df_from_databack(keys)
    # catch empty subdf
    if subdf.shape[0] > 0:
      luembeddings = self.extractor.transform(subdf)['luembeddings']
      return subdf.index.tolist(), luembeddings
    return [], torch.empty((0,self.dim), dtype=self.dtype_pt, device=self.device_embeddings)
  
  def _get_df_from_databack(self, keys):
    raise NotImplementedError('Please use a concrete ModelTensorStorage, i.e. ModelTensorStorageDF (model+df://...) or ModelTensorStorageDB (model+db://...)')


@static_init_logger
class ModelTensorStorageDF(ModelTensorStorage):

  name = 'modeldf'

  def __init__(self,
               modelid, databack, modeldevice, keycolname, masked=True, mask_str=None, mask_subwords=False, pooling='mean',
               device_embeddings='cpu', dtype_pt=torch.float32):

    super().__init__(modelid, databack, modeldevice, masked, mask_str, mask_subwords, pooling, device_embeddings=device_embeddings, dtype_pt=dtype_pt)

    self.keycolname = keycolname
    self._backed_dataframe = None

  # url='model://model_id?masked=n&device=cuda&data=gfn&keycol=global_id',
  @classmethod
  def parse_args_from_url(cls, url):

    args = super().parse_args_from_url(url=url)

    urlsegments = urlparse(url, allow_fragments=True)
    queryargs = parse_qs(urlsegments.query, keep_blank_values=True)
    args['keycolname'] = queryargs.get('keycol', [None])[0]

    return args


  def set_dataframe(self, dataframe, datasetname, set_index=True):
    # assert_backed_dataframe
    assert self.databack in datasetname, 'Make sure to use matching (loaded) datasets.'
    if set_index and self.keycolname is not None:
      self._backed_dataframe = dataframe.set_index(self.keycolname)
    else:
      self._backed_dataframe = dataframe
    self._data_ready = True
    return self


  def _load_dataframe(self):
    raise NotImplementedError('Not yet implemented.')


  @override
  def _get_df_from_databack(self, keys):
    try:
      subdf = self._backed_dataframe.loc[keys] # might throw: KeyError: "['key'] not in index"
      return subdf
    except KeyError as ex:
      self.__class__.cls_logger.warning(f'{ex.__class__.__name__}: {ex.args}')
    return pd.DataFrame([], columns=self.sampledf.columns) # return empty dataframe

      
  @override
  def __contains__(self, key):
    return key in self._backed_dataframe.index

  @override
  def __len__(self):
    return self._backed_dataframe.shape[0]
  
  def __del__(self):
    pass

  @override
  def __setitem__(self, keys, tensors):
    raise NotImplementedError('Not supported on this type of storage.')
  
  @override
  def __delitem__(self, keys):
    raise NotImplementedError('Not supported on this type of storage.')

  @override
  def __repr__(self):
    return f'''
      {super().__repr__()}
      modelid: {self.modelid}
      modeldevice: {self.modeldevice}
      key colum: {self.keycolname}
      backed dataset name: {self.databack}
      backed dataset length: {self._backed_dataframe.shape[0] if self.ready() else "!data not loaded!"}
      vectordim: {self.dim}
      model ready: {"yes" if self._model_ready else "no"}
      data ready: {"yes" if self._data_ready else "no"}
      ready: {"yes" if self.ready() else "no"}
      size: {len(self) if self.ready() else -1}
    '''


@static_init_logger
class ModelTensorStorageDB(ModelTensorStorage):

  name = 'modeldb'

  def __init__(self,
               modelid, databack, modeldevice, masked=True, mask_str=None, mask_subwords=False, pooling='mean',
               device_embeddings='cpu', dtype_pt=torch.float32):

    import_required_on_init(self.__class__, import_modelextractor_db)
    super().__init__(modelid, databack, modeldevice, masked, mask_str, mask_subwords, pooling, device_embeddings=device_embeddings, dtype_pt=dtype_pt)

    self._len_cached = -1
    self._prepare_db_connection()



  # url='model+db://model_id/?masked=n&device=cuda&data=postgresql%2Bpsycopg2://{user}:{password}@{host}:{port}/{databasename}'

  def _prepare_db_connection(self):
    self._engine = sqlalchemy.create_engine(self.databack, pool_pre_ping=True, pool_recycle=3600, isolation_level='AUTOCOMMIT', echo=False)
    self._sessionmaker = sessionmaker(self._engine)
    with self._engine.connect() as conn:
      self._data_ready = conn.execute(sqlalchemy.select(1)).scalar()
    return

  @override
  def _get_df_from_databack(self, keys):
    stmt = sqlalchemy.select(FrameInstance).where(FrameInstance.id.in_(keys))
    with self._sessionmaker() as session:
      res = session.execute(stmt).scalars()
      df = pd.DataFrame.from_records(
        [( fi.extrainfo['tokens'], fi.extrainfo['LU_INDEX'], fi.extrainfo['LU_INDEX_PART'], i, fi.id, fi.global_id ) for i, fi in enumerate(res) ], 
        columns=self.sampledf.columns,
        index='c', # use c as FrameInstance.id
      ) # columns = ['TOKENIZED_SENTENCE', 'LU_INDEX', 'LU_INDEX_PART', 'i', 'c', 'global_id']
    return df

      
  @override
  def __contains__(self, key):
    stmt = sqlalchemy.select(FrameInstance).where(FrameInstance.id == key).exists()
    with self._sessionmaker() as session:
      item_is_in_collection = session.execute(stmt).scalar()
    return item_is_in_collection
    

  @override
  def __len__(self):
    if self._len_cached < 0:
      stmt = sqlalchemy.select(sqlalchemy.func.count()).select_from(FrameInstance)
      with self._sessionmaker() as session:
        count: int = session.execute(stmt).scalar() 
        self._len_cached = count
    return self._len_cached
  

  def __del__(self):
    pass

  @override
  def __setitem__(self, keys, tensors):
    raise NotImplementedError('Not supported on this type of storage.')
  
  @override
  def __delitem__(self, keys):
    raise NotImplementedError('Not supported on this type of storage.')

  @override
  def __repr__(self):
    return f'''
      {super().__repr__()}
      modelid: {self.modelid}
      modeldevice: {self.modeldevice}
      backed dataset name: {self.databack}
      backed dataset length: {len(self) if self.ready() else "!data not loaded!"}
      vectordim: {self.dim}
      model ready: {"yes" if self._model_ready else "no"}
      data ready: {"yes" if self._data_ready else "no"}
      ready: {"yes" if self.ready() else "no"}
      size: {len(self) if self.ready() else -1}
    '''
  
  