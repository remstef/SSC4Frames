import click
from ssc4frames.newcli.main import main
from ssc4frames.newcli.helpers import get_dburl, logger, CorpusfileType, pooling_strategies, get_dburl_from_env, get_hash_for_embeddings, get_hash_for_datasetsplit


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

