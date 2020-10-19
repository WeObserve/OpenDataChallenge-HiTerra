import pandas as pd
import numpy as np
import datetime as dt
import os, sys
from os.path import join
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import logging



class BatcherTrain:
    # This class prepares data to be used for NN training.
    # Function get_batch_rand returns randomly selected batch
    # of data that will will be used for training step at NN.
    # 
    
    def __init__(self, df_ts, # input dataframe of ts data
                 df_gr,
                 inp_time_len, # number of hours for prev-features (this many hours of a single feature will be concatenated)
                 out_time_len, # number of hours for labels 
                 col_dt='Datetime', # Datetime column name
                 col_val='Moisture', # ts column name to be predicted
                 col_gr = 'ChunkId',
                 col_feats_prev = ['Moisture', 'temperature'], # prev-feature column names
                 col_feats_current = ['Day', 'Hour', 'Month'], # current feature column names
                 minlen=25, # chunk lengths are randomly sampled, minimum chunk length
                 maxlen=100, # chunk lengths are randomly sampled, maximum chunk length
                 valid_inds = None # if set, datapoints at these locations indexes will be used only
                 ):
        
        # Set paraemeters
        self.df_ts = df_ts.copy()
        self.df_gr = df_gr.copy()
        self.inp_time_len = inp_time_len
        self.out_time_len = out_time_len
        self.col_dt = col_dt
        self.col_val = col_val
        self.col_gr = col_gr
        self.col_feats_prev = col_feats_prev
        self.col_feats_current = col_feats_current
        self.minlen = minlen
        self.maxlen = maxlen
        self.valid_inds = valid_inds

        self.df_gr = self.df_gr[self.df_gr[self.col_gr].isin(self.df_ts[self.col_gr])]

        # number of uyes and maximum number of hours among chunks
        self.n_chunks = self.df_ts[self.col_gr].nunique()
        self.chunk_ids = np.array(self.df_ts[self.col_gr].unique().tolist()).astype(np.long)
        self.chunk_lens = self.df_ts.groupby(self.col_gr)[self.col_dt].count().values
        self.maxlen = min(self.maxlen, self.chunk_lens.max()-1)

        # Add first and last indexes to self.df_gr
        self.df_ts = self.df_ts.sort_values([self.col_gr, self.col_dt])

        # add dummy rows to the end of array to avoid errors
        df_dummy = pd.concat([self.df_ts.iloc[-2:]] * (int(len(self.df_ts[self.col_dt].unique()) / 2) + 1))
        df_dummy['IsDummy'] = True
        self.df_ts['IsDummy'] = False
        self.df_ts = pd.concat((self.df_ts, df_dummy), axis=0, sort=False)

        self.df_ts = self.df_ts.reset_index(drop=True)
        self.df_ts['Index'] = self.df_ts.index
        self.df_gr = self.df_gr.join(self.df_ts.groupby(self.col_gr).first()['Index'].rename('IndexFirst'), self.col_gr)
        self.df_gr = self.df_gr.join(self.df_ts.groupby(self.col_gr).last()['Index'].rename('IndexLast'), self.col_gr)

        # Check if any hour is missing
        # n_nonconsecutive  = (self.df_ts[self.col_dt].diff() != dt.timedelta(hours=1)).sum() - 1
        # if n_nonconsecutive > 0:
        #     msg = f"Missing at least {n_nonconsecutive} hours in data!"
        #     logging.error(msg)
        #     raise Exception(msg)
        
        # If valid_inds is not set, set it so that no error occurs (start with inp_time_len till -maxlen)
        if self.valid_inds is None:
            self.valid_inds = np.arange(self.inp_time_len, self.df_ts.shape[0]-maxlen)

        # Total number of examples
        self.n_examples = self.valid_inds.shape[0]

        # Number of features
        self.n_feat_prev = self.inp_time_len * len(self.col_feats_prev)
        self.n_feat_current = len(self.col_feats_current)
        self.n_feat = self.n_feat_prev + self.n_feat_current

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def set_minmax_lens(self, minlen=None, maxlen=None, update_valid_inds=True):
        # Change minlen and maxlen values and update if upsampling is set
        if minlen is None and maxlen is None:
            raise Exception("Both lens are None!")

        if minlen:
            self.minlen = minlen
        if maxlen:
            self.maxlen = maxlen

        if update_valid_inds:
            self.valid_inds = np.arange(self.inp_time_len, self.df_ts.shape[0]-self.maxlen)
        
    def get_batch_inds(self, chunk_ids, inds_start_in_chunk, inds_end_in_chunk):

        df_gr_now = self.df_gr[self.df_gr[self.col_gr].isin(chunk_ids)]

        batchsize = df_gr_now.shape[0]

        # get start and end indexes and lengths for given chunk_ids
        inds_start = df_gr_now['IndexFirst'].values + inds_start_in_chunk
        inds_end = df_gr_now['IndexFirst'].values + inds_end_in_chunk + 1 
        # inds_end = df_gr_now['IndexLast'].values + 1

        # Lengths of each chunk
        lens = inds_end - inds_start - self.inp_time_len
        max_len = int(np.max(lens))

        # Initialize mask with ones, make 0 the ones that are dummy
        mask = np.ones((batchsize, max_len, self.out_time_len), dtype=np.float32)
        for i, l in enumerate(lens):
            mask[i,l:,:] = 0

        # Location indexes of the datapoints of chunks, one dimensional.
        # [c0_d0, c0_d1, ..., c0_dm, c1_d0, c1_d1, .... cn_dm] where
        # ci_dj is the j'th datapoint of the i'th chunk

        # inds_current = (np.repeat(inds_start.reshape((-1,1)), max_len, axis=1) + \
        #     np.arange(max_len).reshape((1,-1))).flatten()

        inds_current = (np.repeat(inds_start.reshape((-1,1)) + self.inp_time_len, max_len, axis=1) + \
            np.arange(max_len).reshape((1,-1))).flatten()

        # Get the current features and put them into a 3D tensor
        df_current = self.df_ts.iloc[inds_current]
        X_current = df_current[self.col_feats_current].values.reshape((batchsize, max_len, self.n_feat_current))

        # Location indexes of the prev-features of the datapoints of chunks, one dimensional.
        # For each datapoint of each chunk, there are inp_time_len indexes since they will be concatenated.
        # [c0_d0_p0, c0_d0_p1, ..., c0_d0_pk, ... c0_dm_pk, ..., cn_dm_pk] where
        # ci_dj_pk is the index of k'th previous datapoint of j'th datapoint of i'th chunk
        
        # inds_prev = np.repeat(np.repeat(inds_start.reshape(-1,1) - self.inp_time_len, max_len, axis=1).reshape(-1, max_len, 1), self.inp_time_len, axis=2)\
        #                 + np.arange(max_len).reshape((1, max_len, 1))\
        #                 + np.arange(self.inp_time_len).reshape((1,1,self.inp_time_len))

        inds_prev = np.repeat(np.repeat(inds_start.reshape(-1,1), max_len, axis=1).reshape(-1, max_len, 1), self.inp_time_len, axis=2)\
                        + np.arange(max_len).reshape((1, max_len, 1))\
                        + np.arange(self.inp_time_len).reshape((1,1,self.inp_time_len))

        # Get the previous features and put them into a 3D tensor
        X_prev = self.df_ts.iloc[inds_prev.flatten()][self.col_feats_prev].values.reshape((batchsize, max_len, self.n_feat_prev))

        # Concatenate the prev and current features
        X = np.concatenate((X_prev, X_current), axis=2).astype(np.float32)

        inds_out = np.repeat(np.repeat(inds_start.reshape(-1,1)+ self.inp_time_len, max_len, axis=1).reshape(-1, max_len, 1), self.out_time_len, axis=2)\
                        + np.arange(max_len).reshape((1, max_len, 1))\
                        + np.arange(self.out_time_len).reshape((1,1,self.out_time_len))

        # Get the label data into 3D matrix
        Y = self.df_ts.iloc[inds_out.flatten()][self.col_val].values.reshape((batchsize, max_len, self.out_time_len)).astype(np.float32)

        # # Get the label data into 2D matrix
        # Y = df_current[self.col_val].values.reshape((batchsize, max_len)).astype(np.float32)

        # LSTM wants time-dimension as the first dimension, so change dimensions
        X = X.swapaxes(0,1)
        Y = Y.swapaxes(0,1)
        mask = mask.swapaxes(0,1)
        # Y = np.transpose(Y * mask)
        # mask = mask.transpose()
        
        return X, Y, mask
    
    def get_batch_rand(self, batchsize, minlen=None, maxlen=None):
        # Returns random batches

        if minlen is None:
            minlen = self.minlen
        
        if maxlen is None:
            maxlen = self.maxlen

        maxlen = maxlen - self.inp_time_len

        inds_now = np.random.choice(np.arange(0, self.n_chunks), batchsize, replace=False)
        chunk_ids_now = self.chunk_ids[inds_now]
        maxlens_now = self.chunk_lens[inds_now] - self.inp_time_len

        lens_now = np.random.randint(minlen, maxlen, batchsize)
        lens_now = np.minimum(lens_now, maxlens_now)
        inds_start_in_chunk = np.random.randint(0, maxlens_now - lens_now + self.inp_time_len)
        inds_end_in_chunk = inds_start_in_chunk + lens_now + self.inp_time_len
        # return chunk_ids_now, inds_start_in_chunk, inds_end_in_chunk

        # Get the end indexes and if the index exceeds the last index of 
        # the dataframe, modify
        # inds_end = inds_start + inds_len
        # inds_end = np.minimum(inds_end, self.df_ts.shape[0]-1)

        out = self.get_batch_inds(chunk_ids_now, inds_start_in_chunk, inds_end_in_chunk)

        return out