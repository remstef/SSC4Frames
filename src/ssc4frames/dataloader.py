import csv
import os
import random
import pandas as pd

import ssc4frames.loghelper as loghelper
from ssc4frames.helpers import get_datadir_from_env

logger = loghelper.setup_logger(__name__)


__datasets = {
    'salsa' : {
        'lang': 'de',
        'loader': lambda devel, test, skipsentences: read_salsa_data(devel=devel, test=test, add_frames_to_eval=False, skipsentences=skipsentences)
    },
    'salsa-modified-split' : {
        'lang': 'de',
        'loader':lambda devel, test, skipsentences: read_salsa_data(devel=devel, test=test, add_frames_to_eval=True, skipsentences=skipsentences)
    },
    'salsa-with-proto-frames' : {
        'lang': 'de',
        'loader': lambda devel, test, skipsentences: read_salsa_data(devel=devel, test=test, add_frames_to_eval=False, remove_proto_frames=False, skipsentences=skipsentences)
    },
    'fn1.7' : {
        'lang': 'en',
        'loader': lambda devel, test, skipsentences: read_fn_data(version='1.7', exemplar=False, devel=devel, test=test, skipsentences=skipsentences)
    },
    'fn1.7-sample' : {
        'lang': 'en',
        'loader': lambda devel, test, skipsentences: read_fn_data(version='1.7', exemplar=False, nrows=10, devel=devel, test=test, skipsentences=skipsentences)
    },
    'fn1.7-exemplar' : {
        'lang': 'en',
        'loader': lambda devel, test, skipsentences: read_fn_data(version='1.7', exemplar=True, devel=devel, test=test, skipsentences=skipsentences)
    },
    'fn1.7-exemplar-sample' : {
        'lang': 'en',
        'loader': lambda devel, test, skipsentences: read_fn_data(version='1.7', exemplar=True, nrows=10, devel=devel, test=test, skipsentences=skipsentences)
    }
}

# add random bfn-splits to dataset
for i in range(150):
    ## create scope for split with outer lambda to force correct value
    __datasets[f'fn1.7-altsplit-{i}'] = {
        'lang': 'en',
        'loader': (lambda split: lambda devel, test, skipsentences: read_fn17_data_alternative_splits(split=split, nrows=float('inf'), devel=devel, test=test, skipsentences=skipsentences))(i)
    }

def datasets(log=True):
    if log:
        logger.info(f'Available datasets: {", ".join(__datasets.keys())}')
    return list(__datasets.keys())

def map_to_unified_format(basedataname, df):
    if basedataname.startswith('gfn'):
        df.rename(columns={"LU_LEMMA_FULL": "lu_lemma"}, inplace=True)
        df['frame_label'] = df.apply(lambda r: f'{basedataname}::{r.FRAME_ID}' if r.FRAME_ID is not None else '<unk>', axis=1)
        df['global_id'] = df.apply(lambda r: f'{basedataname}::{r.DATA_SOURCE}::{r.GLOBAL_SENTENCE_ID}::{str(r.LU_INDEX).replace(' ', '')}::[{str(r.TOKENIZED_SENTENCE)[1:20].replace(' ', '')}...]::{r.lu_lemma.replace(' ', '_')}::{r.frame_label}', axis=1)
        return df
    if basedataname.startswith('fn1.'):
        df.rename(columns={"LU_LEMMA_FULL": "lu_lemma"}, inplace=True)
        df['frame_label'] = df.apply(lambda r: f'{basedataname}::{r.FRAME_ID}::{r.FRAME_NAME}' if r.FRAME_ID is not None else '<unk>', axis=1)
        df['global_id'] = df.apply(lambda r: f'{basedataname}::{r.DATA_SOURCE}::{r.GLOBAL_SENTENCE_ID}::{str(r.LU_INDEX).replace(' ', '')}::[{str(r.TOKENIZED_SENTENCE)[1:20].replace(' ', '')}...]::{r.lu_lemma.replace(' ', '_')}::{r.frame_label}', axis=1)
        return df
    if basedataname.startswith('salsa'):
        df.rename(columns={"LU_LEMMA_FULL": "lu_lemma"}, inplace=True)
        df['frame_label'] = df.apply(lambda r: f'{basedataname}::{r.FRAME_ID}' if r.FRAME_ID is not None else '<unk>', axis=1)
        df['global_id'] = df.apply(lambda r: f'{basedataname}::{r.GLOBAL_SENTENCE_ID}::{str(r.LU_INDEX).replace(' ', '')}::[{str(r.TOKENIZED_SENTENCE)[1:20].replace(' ', '')}...]::{r.lu_lemma.replace(' ', '_')}::{r.frame_label}', axis=1)
        return df
    if basedataname.startswith('wiki'):
        df.rename(columns={"LU_LEMMA_FULL": "lu_lemma"}, inplace=True)
        df['frame_label'] = '<unk>'
        df['global_id'] = df.apply(lambda r: f'{basedataname}::{r.DATA_SOURCE}::{r.GLOBAL_SENTENCE_ID}::{str(r.LU_INDEX).replace(' ', '')}::[{str(r.TOKENIZED_SENTENCE)[1:20].replace(' ', '')}...]::{r.lu_lemma.replace(' ', '_')}::{r.frame_label}', axis=1)
        return df
    if basedataname.startswith('mix.'):
        # individual datasets should already be unified 
        return df
    #else
    return df

def get_language(dataset):
    return __datasets[dataset]['lang']

def load_data(dataset=None, devel=False, test=True, skipsentences=False):
    if dataset in __datasets:
        logger.info(f"Loading: '{dataset}'.")
        return __datasets[dataset]['loader'](devel, test, skipsentences)
    else:
        raise Exception(f'''Unknown dataset '{dataset}'. Available datasets: {__datasets.keys()}.''')

def read_data_as_dataframe(fname, split=None, nlines=None, usecols=None):
        df = pd.read_csv(
            fname,
            usecols=usecols,
            sep='\t',
            quoting=csv.QUOTE_MINIMAL,
            header=0,
            skip_blank_lines=True,
            encoding='utf-8',
            converters={
            'TOKENIZED_SENTENCE': str.split,
            'LU_INDEX': lambda x: list(map(int,x.split(','))),
            'LU_INDEX_PART': lambda x: list(map(int,map(float,x.split(',')))) if len(x) else [ ],
            'LU_LEMMA_PART': lambda x: x if len(x) else ''
            },
            nrows=nlines)
        df['i'] = range(0, df.shape[0])
        if split is not None:
            df['split'] = split
        return df

def convert_and_save(d, fname):
    d_ = d.drop(labels=['split', 'fixed_label', 'FRAME_LABEL'], axis=1, inplace=False, errors='ignore')
    d_.TOKENIZED_SENTENCE = d_.TOKENIZED_SENTENCE.apply(lambda x: ' '.join(x))
    d_.LU_INDEX = d_.LU_INDEX.apply(lambda x: ','.join([str(xi) for xi in x]))
    d_.LU_INDEX_PART = d_.LU_INDEX_PART.apply(lambda x: ','.join([str(xi) for xi in x]))
    d_.SUBSTITUTES = ' '
    d_.to_csv(
        fname,
        header=True,
        sep='\t',
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        encoding = 'utf-8')


def read_salsa_data(devel=False, test=True, add_frames_to_eval=True, remove_proto_frames=True, skipsentences=False):

    if skipsentences:
        usecols = ['LU_LEMMA_FULL', 'FRAME_ID', 'GLOBAL_SENTENCE_ID', 'split']
    else:
        usecols = None

    datadir = get_datadir_from_env()

    salsa = read_data_as_dataframe(os.path.join(datadir, 'salsa/salsa.csv'), usecols=usecols)

    if remove_proto_frames:
        ## remove proto frames
        salsa = salsa[~salsa['FRAME_ID'].str.endswith('-salsa')]
    else:
        ## remove proto frames from train
        salsa = salsa[~((salsa['FRAME_ID'].str.endswith('-salsa')) & (salsa.split=='train'))]

    if add_frames_to_eval:
        # identify frames only in train
        train_frames = set(salsa[salsa['split']=='train']['FRAME_ID'].unique())
        devel_frames = set(salsa[salsa['split']=='dev']['FRAME_ID'].unique())
        test_frames = set(salsa[salsa['split']=='test']['FRAME_ID'].unique())

        only_train_frames = train_frames.difference(devel_frames.union(test_frames))

        ## get ids for sentences containing frames that appear only in train
        train_sentences_for_devel_test = set(salsa[(salsa['split']=='train') & (salsa['FRAME_ID'].isin(only_train_frames))]['GLOBAL_SENTENCE_ID'].unique())

        # split these sentences randomly for addition in devel and test
        random.seed(42)
        add_devel = set(random.sample(sorted(train_sentences_for_devel_test), int(len(train_sentences_for_devel_test)/2)))
        add_test = train_sentences_for_devel_test - add_devel

        salsa.loc[salsa['GLOBAL_SENTENCE_ID'].isin(add_devel),'split']= 'dev'
        salsa.loc[salsa['GLOBAL_SENTENCE_ID'].isin(add_test),'split']= 'test'

    ## filter depending on value of devel and test
    if devel and not test:
        salsa = salsa[salsa['split'].isin(set(['train', 'dev']))]
    elif not devel and test:
        salsa = salsa[salsa['split'].isin(set(['train', 'test']))]

    salsa['fixed_label'] = salsa.apply(lambda r: f'F{r.FRAME_ID}' if r.split=='train' else None, axis=1)
    salsa.reset_index(drop=True, inplace=True)
    salsa['c'] = range(0, salsa.shape[0])

    if skipsentences:
        salsa.drop(labels=['FRAME_ID', 'GLOBAL_SENTENCE_ID'], axis=1, inplace=True)

    return salsa

def read_fn_dataframe(nrows=float('inf'), version='1.7', exemplar=False, devel=False, test=True, skipsentences=False):
    from conllu import parse_incr
    from io import open
    from pathlib import Path
    import ast

    datadir = get_datadir_from_env()
    basename = os.path.join(datadir, f'fn{version}/open_sesame_v1_data/fn{version}/fn{version}')
    
    fnames = {
        'train': f'{basename}.fulltext.train.syntaxnet.conll',
        'exemplar': f'{basename}.exemplar.train.syntaxnet.conll',
        'dev': f'{basename}.dev.syntaxnet.conll',
        'test': f'{basename}.test.syntaxnet.conll',
    }

    def get_data_rows(fname, split, nrows=float('inf'), skipsentences=False):
        with open(fname, 'r', encoding='utf-8') as fh:
            for di, cdoc in enumerate(filter(lambda cdoc: cdoc.metadata['LU'].endswith('.v'), parse_incr(fh, fields='ID FORM LEMMA PLEMMA POS PPOS FEAT PFEAT HEAD PHEAD DEPREL PDEPREL FILLPRED_LU PRED_FRAME APREDS'.lower().split()))):
                if di >= nrows:
                    break
                luidx = [i for i, t in enumerate(cdoc) if t['pred_frame'] != '_']
                if skipsentences:
                    row = {
                        'FRAME_ID' : cdoc.metadata['FRAMEID'],
                        'DATA_SOURCE' : cdoc.metadata['SOURCE'],
                        'LU_LEMMA_FULL' : ' '.join([cdoc[ix]['plemma'] for ix in luidx ]),
                        'split' : split
                    }
                else:
                    row = {
                        'FRAME_ID' : cdoc.metadata['FRAMEID'],
                        'DATA_SOURCE' : cdoc.metadata['SOURCE'],
                        'FRAME_NAME' : cdoc.metadata['FRAMENAME'],
                        'TOKENIZED_SENTENCE' : [ast.literal_eval(t['form']).decode('utf-8') for t in cdoc],
                        'GLOBAL_SENTENCE_ID' : cdoc.metadata['SID'],
                        'LU_INDEX' : luidx,
                        'LU_INDEX_PART' : [],
                        'LU' : cdoc.metadata['LU'],
                        'LU_LEMMA' : ' '.join([cdoc[ix]['plemma'] for ix in luidx ]),
                        'LU_LEMMA_PART' : '',
                        'LU_LEMMA_FULL' : ' '.join([cdoc[ix]['plemma'] for ix in luidx ]),
                        'SUBSTITUTES' : '',
                        'i' : di,
                        'split' : split
                    }
                yield row

    # collect the data
    consider_files = [ ]
    if exemplar:
        consider_files.append('exemplar')
    else: 
        consider_files.append('train')
    if devel:
        consider_files.append('dev')
    if test:
        consider_files.append('test')
    
    dataframes = [ pd.DataFrame.from_dict(get_data_rows(fname=str(Path(v).expanduser()), split=k, nrows=nrows, skipsentences=skipsentences), orient='columns') for k,v in fnames.items() if (k in consider_files) ]

    # use one main dataframe, combine train + dev or train + test or train + dev + test
    df = pd.concat(dataframes, sort=False)

    return df


def add_labels_to_bfn_df(df, exemplar=False, skipsentences=True):

    df.reset_index(drop=True, inplace=True)
    df['c'] = range(0, df.shape[0])
    if exemplar:
        df.loc[ df.split == 'exemplar', ( 'split' ) ] = 'train'

    # df['luextract'] = df.apply(lambda r: [ r.TOKENIZED_SENTENCE[i] for i in r.LU_INDEX + r.LU_INDEX_PART ], axis=1)
    df['fixed_label'] = df.apply(lambda r: f'F{r.FRAME_ID}' if r.split=='train' else None, axis=1)
    df['FRAME_LABEL'] = df.apply(lambda r: f'F{r.FRAME_ID}', axis=1)

    if skipsentences:
        df.drop(['FRAME_ID', 'FRAME_LABEL', 'DATA_SOURCE'], axis=1, inplace=True)

    return df


def read_fn_data(nrows=float('inf'), version='1.7', exemplar=False, devel=False, test=True, skipsentences=False):

    df = read_fn_dataframe(nrows=nrows, version=version, exemplar=exemplar, devel=devel, test=test, skipsentences=skipsentences)
    return add_labels_to_bfn_df(df, exemplar=exemplar, skipsentences=skipsentences)


def read_fn17_data_alternative_splits(split=0, nrows=float('inf'), devel=False, test=True, skipsentences=False):
    # read default splits
    dfn = read_fn_dataframe(nrows=nrows, version='1.7', exemplar=False, devel=True, test=True, skipsentences=skipsentences)

    # get alternative split
    datadir = get_datadir_from_env()    
    split_data = pd.read_csv(os.path.join(datadir, 'fn1.7/bfn_17_split_assignments_wide.csv'), sep="\t")
    split_data_grouped = split_data.groupby(str(split))
    test_sources= set(split_data_grouped.get_group('test')['DATA_SOURCE'])
    dev_sources= set(split_data_grouped.get_group('dev')['DATA_SOURCE'])
    print(split)
    print(test_sources)

    # Assign the splits based on DATA_SOURCE
    dfn['split'] = 'train'  # Default everything to 'train'
    dfn.loc[dfn['DATA_SOURCE'].isin(test_sources), 'split'] = 'test'
    dfn.loc[dfn['DATA_SOURCE'].isin(dev_sources), 'split'] = 'dev'

    ## remove test/devel
    if not devel:
        dfn = dfn[dfn.split != 'dev']
    if not test:
        dfn = dfn[dfn.split != 'test']

    return add_labels_to_bfn_df(dfn, exemplar=False, skipsentences=skipsentences)


