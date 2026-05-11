#!/usr/bin/env python
# coding: utf-8

import os
import ssc4frames.loghelper as loghelper; logger = loghelper.setup_logger(os.path.basename(__file__))

class Importer(object):

    def __init__(self, dbconnectionstring, pool_size=10, max_overflow=20) -> None:

        global Dataset, FrameInstance, DatasetSplit, SplitRelation, select, pginsert

        from ssc4frames.database import Dataset, FrameInstance, DatasetSplit, SplitRelation
        from sqlalchemy import select
        from sqlalchemy.dialects.postgresql import insert as pginsert

        from ssc4frames.database import DBHandler
        self.dbconnectionstring = dbconnectionstring
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.dbhandler = DBHandler(dbconnectionstring, pool_size=pool_size, max_overflow=max_overflow)


    def _load_dataframe(self, datasetname, basedataname):
        global load_data, map_to_unified_format, get_language
        from ssc4frames.dataloader import load_data, map_to_unified_format, get_language
        df = map_to_unified_format(basedataname, load_data(datasetname, devel=True, test=True, skipsentences=False))
        language = get_language(datasetname)
        return df, language


    def _import_dataframe(self, dataframe, datasetname, basedataname, language):
        with self.dbhandler.sessionmaker() as session:
            with session.begin():
                # check if dataset exists, otherwise add
                stmt = select(Dataset).where(Dataset.name == basedataname).limit(1)
                dataset = session.execute(stmt).scalar()
                if dataset == None:
                    dataset = Dataset(name=basedataname, language=language)
                    session.add(dataset)
                    # flush to fill in id
                    session.flush()
            
                # check if dataset split exists, otherwise add
                datasetsplitname = f'{basedataname}-default' if datasetname == basedataname else datasetname
                stmt = select(DatasetSplit).where(DatasetSplit.name == datasetsplitname).where(DatasetSplit.dataset_id == dataset.id).limit(1)
                datasplit = session.execute(stmt).scalar()
                if datasplit is None:
                    datasplit = DatasetSplit(name=datasetsplitname, dataset_id=dataset.id)
                    session.add(datasplit)
                    session.flush()

                # check for existing frame instances, otherwise add
                stmt = select(FrameInstance).where(FrameInstance.global_id.in_(dataframe.global_id.tolist()))
                existing_frameinstance_ids = { instance.global_id: instance.id for instance in session.scalars(stmt) }
                new_frameinstances = { 
                    r.global_id: FrameInstance(
                        dataset_id=dataset.id, 
                        lu_lemma=r.lu_lemma, 
                        frame_label=r.frame_label, 
                        global_id=r.global_id,
                        extrainfo={
                            'tokens': r.TOKENIZED_SENTENCE,
                            'LU_INDEX': r.LU_INDEX, 
                            'LU_INDEX_PART': r.LU_INDEX_PART,
                            'LU': r.LU, 
                            'LU_LEMMA': r.LU_LEMMA,
                            'LU_LEMMA_PART': r.LU_LEMMA_PART
                        }
                    ) for r in dataframe.itertuples() if r.global_id not in existing_frameinstance_ids }
                if len(new_frameinstances) > 0:
                    session.add_all(new_frameinstances.values())
                    session.flush()

                frame_instance_id_to_datasplit = [ (existing_frameinstance_ids[r.global_id] if r.global_id in existing_frameinstance_ids else new_frameinstances[r.global_id].id, r.split) for r in dataframe.itertuples() ]
                # add splitrelations, ignore if exists
                splitrelations = [ { 'datasetsplit_id': datasplit.id, 'instance_id': instance_id, 'split': split } for instance_id, split in frame_instance_id_to_datasplit ]
                if session.bind.dialect.name == 'postgresql':
                    stmt = pginsert(SplitRelation.__table__).on_conflict_do_nothing()
                else: # default insert (fails on duplicate pk entry)
                    stmt = SplitRelation.__table__.insert()
                session.execute(stmt, splitrelations)
        return None
        

    def import_data(self, datasetname, basedataname, batchsize, offset=0):

        df, language = self._load_dataframe(datasetname, basedataname)
        return self.import_from_dataframe(datasetname, basedataname, language, df, batchsize, offset)


    def import_from_dataframe(self, datasetname, basedataname, language, df, batchsize, offset=0):
        if offset > 0:
            df = df[offset:]
        if batchsize <= 0:
            self._import_dataframe(df, datasetname, basedataname, language)
        else:
            # prepare batches
            n_batches, n_rest = df.shape[0] // batchsize, df.shape[0] % batchsize
            logger.info(f'batchsize: {batchsize}, #batches: {n_batches}, rest: {n_rest}')
            # start batch import
            for batch_i in range(n_batches):
                b = batch_i * batchsize
                e = b + batchsize
                logger.info(f'Batch {batch_i+1}/{(n_batches + (1 if n_rest > 0 else 0))}: [{b},{e}[')
                subdf = df.iloc[b:e]
                self._import_dataframe(subdf, datasetname, basedataname, language)
            # last batch
            if n_rest > 0:
                b = n_batches * batchsize
                e = b + n_rest
                logger.info(f'Batch {(n_batches + 1)}/{(n_batches + 1)}: [{b},{e}[')
                subdf = df.iloc[b:e]
                self._import_dataframe(subdf, datasetname, basedataname, language)
        return None

    def _update_data_in_dataframe(self, df, datasetid):
        
        for j, (i, row) in enumerate(df.iterrows()):
            new_extra_info = {
                'tokens': row.TOKENIZED_SENTENCE,
                'LU_INDEX': row.LU_INDEX, 
                'LU_INDEX_PART': row.LU_INDEX_PART,
                'LU': row.LU, 
                'LU_LEMMA': row.LU_LEMMA,
                'LU_LEMMA_PART': row.LU_LEMMA_PART
            }
            with self.dbhandler.sessionmaker.begin() as session:
                stmt = select(FrameInstance).where(FrameInstance.global_id == row.global_id)
                res = session.execute(stmt)
                fi = res.scalar()
                if fi is None:
                    fi = FrameInstance(
                        dataset_id=datasetid, 
                        lu_lemma=row.lu_lemma, 
                        frame_label=row.frame_label, 
                        global_id=row.global_id,
                        extrainfo=new_extra_info
                    )
                    session.add(fi)
                    session.flush()
                    session.commit()
                else:
                    # else fi exists
                    if fi.extrainfo is None:
                        fi.extrainfo = new_extra_info
                    else:
                        fi.extrainfo |= new_extra_info
                    session.add(fi)
                    session.flush()
                    session.commit()
        return df.shape[0]


    def update_data(self, datasetname, batchsize, threads=12):
        with self.dbhandler.sessionmaker() as session:
            stmt = select(Dataset).where(Dataset.name == datasetname.split('-')[0])
            dataset = session.execute(stmt).scalars().first()
            datasetid = dataset.id

        df,_ = self._load_dataframe(datasetname, datasetname.split('-')[0])
        if batchsize <= 0:
            self._update_data_in_dataframe(df, datasetid)
        else:
            # prepare batches
            n_batches, n_rest = df.shape[0] // batchsize, df.shape[0] % batchsize
            logger.info(f'batchsize: {batchsize}, #batches: {n_batches}, rest: {n_rest}')

            global ThreadPoolExecutor, ProcessPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, ALL_COMPLETED
            
            with ProcessPoolExecutor(max_workers=threads, initializer=init_worker, initargs=(self.dbconnectionstring, self.pool_size, self.max_overflow) ) as executor:
                futures = [ ]    
                # start batch import
                for batch_i in range(n_batches):
                    b = batch_i * batchsize
                    e = b + batchsize
                    futures.append(executor.submit(p_fun, b, e, df.iloc[b:e], datasetid, f'Batch {batch_i+1}/{(n_batches + (1 if n_rest > 0 else 0))}: [{b},{e}['))

                # last batch
                if n_rest > 0:
                    b = n_batches * batchsize
                    e = b + n_rest
                    futures.append(executor.submit(p_fun, b, e, df.iloc[b:e], datasetid, f'Batch {(n_batches + 1)}/{(n_batches + 1)}: [{b},{e}['))

                # explicit waiting
                wait(futures, timeout=None, return_when=ALL_COMPLETED)  # ALL_COMPLETED is actually the default
                
            results = [f.result() for f in futures]
            print(sum(results))
                
        return None


# p_fun = lambda b,e: 
def p_fun(b,e,subdf,datasetid,msg):
    logger.info(msg)
    return global_process_importer._update_data_in_dataframe(subdf, datasetid)

def init_worker(db, poolsize, overflow):
    global global_process_importer
    global_process_importer = Importer(db, poolsize, overflow)

