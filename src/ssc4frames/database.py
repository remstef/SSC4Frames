from typing import List
from datetime import datetime
import hashlib
import json
import pprint
import os

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, deferred
from sqlalchemy import ForeignKey, String, UniqueConstraint, DateTime, Boolean, ARRAY, Integer, Text # JSON
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy_json import mutable_json_type
from sqlalchemy import inspect as sqlalchemyinspect
from sqlalchemy.orm.session import close_all_sessions

import pandas as pd
import ssc4frames.loghelper as loghelper

class DBHandler:

    def __init__(self, dbconnectionstring,
                 pool_size=5, max_overflow=10, autoflush=False,
                 **enginekwargs):
        
        self.logger = loghelper.setup_logger(f'{DBHandler.__name__}{id(self)}')
        self.dbconnectionstring = dbconnectionstring
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.enginekwargs = {
            'pool_pre_ping': True, 
            'pool_recycle': 3600, 
            'isolation_level': 'AUTOCOMMIT', 
            'echo': False, 
            'pool_size': self.pool_size, 
            'max_overflow': self.max_overflow
        } | enginekwargs

        self._prepare_session(autoflush=autoflush)
        self.prepare_db()


    def prepare_db(self):
        with self.engine.connect() as connection:
            Base.metadata.create_all(bind=connection, checkfirst=True)


    def _prepare_session(self, autoflush=False):

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        self.logger.info(f'Using DB: \'{self.dbconnectionstring}\'.')
        self.engine = create_engine(self.dbconnectionstring, **self.enginekwargs)
        self.sessionmaker = sessionmaker(self.engine, autoflush=autoflush)

        return None
    

    def dispose(self):
        # pass
        try:
            close_all_sessions()
            self.engine.dispose()
        except AttributeError as e:
            # catch and ignore
            # File "sqlalchemy/orm/session.py", line 5177, in close_all_sessions
            # AttributeError: 'NoneType' object has no attribute 'values' 
            pass


    def __del__(self):
        # self.dispose()
        del self.sessionmaker
        del self.engine


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    
    __tablename__ = 'datasets'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30), unique=True)
    language: Mapped[str] = mapped_column(String(3)) ## language identifier - ISO 639-3
    frame_instances: Mapped[List["FrameInstance"]] = relationship()


class FrameInstance(Base):
    
    __tablename__ = 'frameinstances'

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey(f'''{Dataset.__table__.name}.id''', deferrable=True, initially="DEFERRED"), index=True)
    lu_lemma: Mapped[str] = mapped_column(String(64), index=True)
    frame_label: Mapped[str] = mapped_column(String(128), index=True, nullable=True)
    global_id: Mapped[str] = mapped_column(String(256), unique=True)
    # for additional data:
    extrainfo: Mapped[dict] = deferred(mapped_column(mutable_json_type(dbtype=JSONB, nested=True), nullable=True), group='extrainfos')
    
    __table_args__ = (UniqueConstraint('id', 'dataset_id', name='_id_dataset_id_uc'),)


class DatasetSplit(Base):
    
    __tablename__ = "datasetsplits"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey(f'''{Dataset.__table__.name}.id''', deferrable=True, initially="DEFERRED"), index=True)
    name: Mapped[str] = mapped_column(String(30), unique=True, index=True)
    frame_instances_split: Mapped[List["SplitRelation"]] = relationship()
    __table_args__ = (UniqueConstraint('dataset_id', 'name', name='_dataset_id_name_uc'),)

    base_dataset: Mapped[Dataset] = relationship()

    def __repr__(self):
        return f'Splitname: {self.name}, Base-Dataset: {self.base_dataset.name}'

    def get_instance_df(self, devel=True, test=True):

        return pd.DataFrame.from_records([
            {
                'lu_lemma':    frame_instance_split.frame_instance.lu_lemma,
                'frame_label': frame_instance_split.frame_instance.frame_label,
                'split':       frame_instance_split.split,
                'uid':         frame_instance_split.frame_instance.id
            }
            for frame_instance_split in self.frame_instances_split
            if (frame_instance_split.split != 'dev' or devel) and (frame_instance_split.split != 'test' or test)])

class SplitRelation(Base):
    
    __tablename__ = "split_instances"
    
    datasetsplit_id: Mapped[int] = mapped_column(ForeignKey(f'''{DatasetSplit.__table__.name}.id''', deferrable=True, initially="DEFERRED"), primary_key=True, index=True)
    instance_id: Mapped[int] = mapped_column(ForeignKey(f'''{FrameInstance.__table__.name}.id''', deferrable=True, initially="DEFERRED"), primary_key=True, index=True)
    split: Mapped[str] = mapped_column(String(30), primary_key=True, index=True)
    frame_instance: Mapped["FrameInstance"] = relationship()


class Clustering(Base):
    
    __tablename__ = "clusterings"

    id: Mapped[int] = mapped_column(primary_key=True)
    datasetsplit_id: Mapped[int] = mapped_column(ForeignKey(f'''{DatasetSplit.__table__.name}.id''', deferrable=True, initially="DEFERRED"), index=True)
    splits: Mapped[list[str]] = mapped_column(ARRAY(String(30))) # ['train','dev','test']
    numinstances: Mapped[int] = mapped_column(Integer)
    numclusters: Mapped[int] = mapped_column(Integer, default=0)
    type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(64), nullable=True)
    start: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    finish: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=True)
    identifier: Mapped[str] = mapped_column(String(64), nullable=True)
    setting: Mapped[dict] = mapped_column(mutable_json_type(dbtype=JSONB, nested=True))
    logs: Mapped[list[str]] = deferred(mapped_column(ARRAY(Text), nullable=True))
    datasplit: Mapped["DatasetSplit"] = relationship(lazy="joined", innerjoin=True) # eager loading, its fast anyways, b/c small tables
    extrainfo: Mapped[dict] = deferred(mapped_column(mutable_json_type(dbtype=JSONB, nested=True), nullable=True), group='extrainfos') # anything not so important, but that might be necessary or usable later

    def __repr__(self) -> str:
        return f'''{self.__class__.__name__} {self.id}:
            Ident: {self.identifier}
            Type: {self.type}
            Data: {'<detached-and-unloaded>' if 'datasplit' in sqlalchemyinspect(self).unloaded and sqlalchemyinspect(self).detached else self.datasplit.name} ({self.datasetsplit_id}) {self.splits} ({self.numinstances} instances)
            Status: {self.status}
            Start: {self.start}
            Finish: {self.finish}
            Success: {self.success}
            Setting: {'<detached-and-unloaded>' if 'setting' in sqlalchemyinspect(self).unloaded and sqlalchemyinspect(self).detached else self.setting}
            #Logs: {'<detached-and-unloaded>' if 'logs' in sqlalchemyinspect(self).unloaded and sqlalchemyinspect(self).detached else (0 if self.logs == None else len(self.logs))}
        '''.replace(' '*12, '  ')

    @classmethod
    def get_identifier_from_settings(cls, settings_dict):

        ## make sure that settings contain only data and local or global settings
        settings_keys = set(settings_dict.keys())
        assert settings_keys == {'data', 'local'} or settings_keys == {'data', 'global'}

        identifier = json.dumps(settings_dict, sort_keys=True, indent=None, separators=(',', ':'))
        identifier = hashlib.md5(identifier.encode('utf-8')).hexdigest()

        return identifier



class Cluster(Base):
   
    __tablename__ = "clusters"

    id: Mapped[int] = mapped_column(primary_key=True)
    clusteringid: Mapped[int] = mapped_column(ForeignKey(f'''{Clustering.__table__.name}.id''', deferrable=True, initially="DEFERRED"), index=True)
    label: Mapped[str] = mapped_column(String(256), index=True) # label length might be 128+64+x so just make it 256 to be sure
    extrainfo: Mapped[dict] = deferred(mapped_column(mutable_json_type(dbtype=JSONB, nested=True), nullable=True), group='extrainfos') # debugging infos, e.g. lemma, interal id, unique label, ...
    clusterinstances: Mapped[List["ClusterAssignment"]] = relationship()
   
    __table_args__ = (UniqueConstraint('clusteringid', 'label', name=f'_clusteringid_label_uc'),)


class ClusterAssignment(Base):
   
    __tablename__ = "clusterassignments"

    instanceid: Mapped[int] = mapped_column(ForeignKey(f'''{FrameInstance.__table__.name}.id''', deferrable=True, initially="DEFERRED"), primary_key=True, index=True)
    clusterid: Mapped[int] = mapped_column(ForeignKey(f'''{Cluster.__table__.name}.id''', deferrable=True, initially="DEFERRED"), primary_key=True, index=True)
    extrainfo: Mapped[dict] = deferred(mapped_column(mutable_json_type(dbtype=JSONB, nested=True), nullable=True), group='extrainfos') # debugging infos, e.g. splitname, fixed_label input, ...
    frameinstance: Mapped["FrameInstance"] = relationship()


def create_new_clusterembedding_table_class(clustering_id:int, vectordimension:int):
    # create a dynamic class using type(classname, superclasses, attributedict)
    CurrentClusterEmbedding = type(
        f'ClusterEmbedding__{clustering_id}', 
        (Base,), 
        {
            '__tablename__':f'clusterembeddings__{clustering_id}', 
            'clusterid': mapped_column(ForeignKey(f'''{Cluster.__table__.name}.id''', deferrable=True, initially="DEFERRED"), primary_key=True),
            'embedding': mapped_column(Vector(vectordimension), nullable=True), # <== this means that we cannot change the model dimension, its better to have distint tables for experiment clusters, same as distinct model tables, for the same reason
            'aggregationtype': mapped_column(String(32)),
            'extrainfo': deferred(mapped_column(mutable_json_type(dbtype=JSONB, nested=True), nullable=True)) # debugging infos, e.g. instance ids, weight, source embeddings, ... 
        }
    )
    return CurrentClusterEmbedding


def get_model_embedding_table_class(tablename:str, vectordimension:int):
    CurrentInstanceEmbedding = type(
        f'InstanceEmbedding__{tablename.replace('-','_')}', 
        (Base,), 
        {
            '__tablename__': tablename, 
            'key': mapped_column(primary_key=True, autoincrement=False),
            'embedding': mapped_column(Vector(vectordimension), nullable=True), # <== this means that we cannot change the model dimension, its better to have distint tables for experiment clusters, same as distinct model tables, for the same reason
        }
    )
    return CurrentInstanceEmbedding


class Experiment(Base):
    
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), index=True, nullable=True)
    extrainfo: Mapped[dict] = deferred(mapped_column(mutable_json_type(dbtype=JSONB, nested=True), nullable=True), group='extrainfos') # anything not so important, but that might be necessary or usable later
    runs: Mapped[List["ExperimentRun"]] = relationship(lazy="joined", back_populates="experiment")

    def get_status(self) -> str:
        return str(set([r.status for r in self.runs]))
    
    def __repr__(self) -> str:
        return f'''{self.__class__.__name__} {self.id}:
            Name: {self.name}
            Status: {self.get_status()}
            Extrainfo: {'<detached-and-unloaded>' if 'extrainfo' in sqlalchemyinspect(self).unloaded and sqlalchemyinspect(self).detached else ('{}' if self.extrainfo == None else '\n'+pprint.pformat(self.extrainfo))}
            #Runs: {'<detached-and-unloaded>' if 'runs' in sqlalchemyinspect(self).unloaded and sqlalchemyinspect(self).detached else (0 if self.runs == None else len(self.runs))}
        '''.replace(' '*8, '')[:-1] # '[\n'+('\n,'.join(map(str, self.runs)))+'\n]'}
    

class ExperimentRun(Base):

    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey(f'''{Experiment.__table__.name}.id''', deferrable=True, initially="DEFERRED"), index=True)
    experiment: Mapped["Experiment"] = relationship(lazy="joined", back_populates="runs")
    setting: Mapped[dict] = mapped_column(mutable_json_type(dbtype=JSONB, nested=True), nullable=False)
    status: Mapped[str] = mapped_column(String(64), default='created', index=True)
    require: Mapped[dict] = mapped_column(mutable_json_type(dbtype=JSONB, nested=False), nullable=True)
    clustering_id: Mapped[int] = mapped_column(ForeignKey(f'''{Clustering.__table__.name}.id''', deferrable=True, initially="DEFERRED"), index=True, nullable=True)
    clustering: Mapped["Clustering"] = relationship(lazy="joined")
    extrainfo: Mapped[dict] = deferred(mapped_column(mutable_json_type(dbtype=JSONB, nested=True), nullable=True), group='extrainfos') # anything not so important, but that might be necessary or usable later
    logs: Mapped[list[str]] = deferred(mapped_column(ARRAY(Text), nullable=True), group='logs')
    
    def __repr__(self) -> str:
        # Setting: \n{pprint.pformat(self.setting,depth=3)}
        return f'''{self.__class__.__name__} {self.id}:
            Experiment: {self.experiment_id}
            Status: {self.status}
            Run-Requirements: {self.require}
            Clustering: {self.clustering_id}
            Extrainfo: {'<detached-and-unloaded>' if 'extrainfo' in sqlalchemyinspect(self).unloaded and sqlalchemyinspect(self).detached else ('{}' if self.extrainfo == None else '\n'+pprint.pformat(self.extrainfo))}
            #Logs: {'<detached-and-unloaded>' if 'logs' in sqlalchemyinspect(self).unloaded and sqlalchemyinspect(self).detached else (0 if self.logs == None else len(self.logs))}
        '''.replace(' '*8, '')[:-1]
