import os
import sys

from typing import override
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoModel
import torch
import gc

import wp_retok

import ssc4frames.loghelper as loghelper; logger = loghelper.setup_logger(os.path.basename(__file__))


class RandomEmbeddings(BaseEstimator, TransformerMixin):

    def __init__(self, emdimension=8, normalize=True, return_numpy=False) -> None:
        super().__init__()
        self.emdimension = emdimension
        self.return_numpy = return_numpy
        self.normalize = normalize

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        N = X.shape[0]
        # generate a uniformly distributed mean in [-1,1] and generate a normal distributed values with standard deviation 1
        M = torch.normal(mean=(torch.rand(N, self.emdimension)*2)-1, std=torch.ones(N, self.emdimension))
        if self.normalize:
            M_norm = (M / M.norm(dim=0).expand(M.size()))
            M = M_norm
        if self.return_numpy:
            return M.numpy()
        return {'df': X, 'luembeddings': M}


class EmbeddingsExtractor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 modelname='bert-base-german-cased',
                 batch_size=0,
                 device=None,
                 device_store=None,
                 pooling=lambda mat: mat.mean(dim=0),
                 return_numpy=False,
                 masking=False,
                 mask_str=None,
                 mask_subwords=False
    ):

        self.modelname = modelname
        self.batch_size = batch_size
        self.device_store = device_store
        self.device = device
        self.pooling = pooling
        self.return_numpy = return_numpy
        self.masking = masking
        self.mask_str = mask_str
        self.mask_subwords = mask_subwords
        if (self.mask_str is not None) and self.mask_subwords:
            raise ValueError("Masking subwords with given mask_string is not supported.")
        self.batchretokenizer = None
        self.model = None

    def fit(self, X, y=None):

        return self
        
    def __get_lu_embedding_normalized(self, embeddings, index, LU_INDEX, LU_INDEX_PART):
        # collect all wordpiece ids
        lu_all_wp_tokenids = []
        for lu_index in LU_INDEX + LU_INDEX_PART:
            lu_wp_tokenids = index[lu_index]
            lu_all_wp_tokenids += lu_wp_tokenids[1]
            # print(f'{len(lu_wp_tokenids[1])} embeddings for "{lu_wp_tokenids[0]}" ({lu_wp_tokenids[2]}) at position {LU_INDEX}: \n{embeddings[lu_wp_tokenids[1],:5]}...')
        # get average embedding
        lu_embedding = self.pooling(embeddings[lu_all_wp_tokenids])
        # lu_embedding -= lu_embedding.min() # shift to have positive values only
        # print(f'mean of {len(lu_all_wp_tokenids)} wordpieces: {lu_embedding[:5]}')
        lu_embedding_norm = (lu_embedding / lu_embedding.norm())
        # print(f'norm of {len(lu_all_wp_tokenids)} wordpieces: {lu_embedding_norm[:5]}')
        return lu_embedding_norm

    def transform(self, X, y=None, clear_model=False):

        if self.batchretokenizer is None:
            self.batchretokenizer = wp_retok.BatchReTokenizer(self.modelname, encoder_length='max_len_single_sentence_in_batch')
        
        if self.model is None:
            if self.modelname.startswith('nvidia/'):
                self.model = AutoModel.from_pretrained(self.modelname, trust_remote_code=True).to(self.device)
                self.model = self.model.embedding_model
            else:
                self.model = AutoModel.from_pretrained(self.modelname).to(self.device)
        
        ## re-tokenize and get input ids for the transformer model
        # tokenize batched
        
        tokenized_sentences = X.TOKENIZED_SENTENCE.tolist()
        lu_indexes = X.LU_INDEX.tolist()

        if self.masking == True:
            if self.mask_str is not None:
                ## use custom string for masking
                mask_token = self.mask_str
            elif self.modelname.startswith('nvidia/'):
                ## TODO what about nvidia? - does this have a mask token at all? - I don't think so
                ## maybe raise error: model has no mask?
                ## this is the old behavior
                mask_token = 'MASK'
            else:
                ## use mask_token from tokenizer
                mask_token = self.batchretokenizer.modeltokenizer.mask_token

            if self.mask_subwords:
                # get encoded mask-token
                mask_token_encoded = self.batchretokenizer.modeltokenizer.convert_tokens_to_ids(mask_token)
            else:
                masked_sentences = [None] * len(tokenized_sentences)
                for index, (tokenized_sentence, lu_index) in enumerate(zip(tokenized_sentences,lu_indexes)):
                    masked_sentence = tokenized_sentence.copy()
                    masked_sentence[lu_index[0]] = mask_token
                    masked_sentences[index] = masked_sentence

                tokenized_sentences = masked_sentences

        
        encoded_input, indexes, _ = self.batchretokenizer.batch_retokenize_and_encode_indexed(tokenized_sentences)

        if self.masking and self.mask_subwords:
            # substitute subwords for LU index with encoded mask token

            masked_encoded_input = [None] * len(encoded_input['input_ids'])

            for index, (encoded_sentence, token_subword_index, lu_index) in enumerate(zip(encoded_input['input_ids'],indexes,lu_indexes)):
                masked_encoded_sentence = encoded_sentence.clone()
                for subword_index in token_subword_index[lu_index[0]][1]:
                    masked_encoded_sentence[subword_index] = mask_token_encoded
                masked_encoded_input[index] = masked_encoded_sentence

            encoded_input['input_ids'] = torch.stack(masked_encoded_input)


        # apply transformer model and get all word embeddings
        if self.batch_size == 0:
            # move input ids for the transformer model to respective device (zero cost if already on the same device)
            for k in encoded_input.keys():
                encoded_input[k] = encoded_input[k].to(self.device)
            # get all embeddings in single batch
            with torch.no_grad():
                out_dict = self.model(**encoded_input, output_attentions=False, output_hidden_states=True, return_dict=True)
            embeddings = out_dict['last_hidden_state'].to(self.device_store)

        else:

            # get all embeddings in multiple batches
            input_ids = encoded_input["input_ids"]
            attention_mask =encoded_input["attention_mask"]
            token_type_ids = encoded_input["token_type_ids"]

            embeddings_list = []
            num_batches = int(len(input_ids)/self.batch_size)
            logger.info(f"get all {len(input_ids)} embeddings in {str(num_batches)} batches")

            for enum_i, i in enumerate(range(0, len(input_ids), self.batch_size), start=1):
                logger.info(f"batch {enum_i}/{num_batches}: input {i} to {i+self.batch_size}")
                inp_ids = input_ids[i:i+self.batch_size].to(self.device)
                att_mask = attention_mask[i:i+self.batch_size].to(self.device)
                toktyp_ids = token_type_ids[i:i+self.batch_size].to(self.device)

                with torch.no_grad():
                    out_dict = self.model(input_ids=inp_ids, attention_mask=att_mask, token_type_ids=toktyp_ids, output_attentions=False, output_hidden_states=True, return_dict=True)
                batch_embeddings = out_dict['last_hidden_state'].to(self.device_store)
                
                del inp_ids, att_mask, toktyp_ids, out_dict
                # gc.collect()
                torch.cuda.empty_cache()

                embeddings_list.append(batch_embeddings)

            embeddings = torch.cat(embeddings_list)
        
        del encoded_input
        if clear_model:
            del self.model
        # gc.collect()
        torch.cuda.empty_cache()

        # get LU embedding
        lu_embeddings = [ self.__get_lu_embedding_normalized(embeddings[i], indexes[i], X.iloc[i].LU_INDEX, X.iloc[i].LU_INDEX_PART) for i in range(embeddings.size(0)) ]

        # stack all
        M = torch.stack(lu_embeddings)

        del lu_embeddings
        # gc.collect()
        torch.cuda.empty_cache()

        if self.return_numpy:
            return M.numpy()
        else:
            return {'df': X, 'luembeddings': M}


