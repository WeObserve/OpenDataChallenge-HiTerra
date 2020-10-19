import pandas as pd
import numpy as np
import datetime as dt
import os, sys
from os.path import join
import time
import pickle
from importlib import reload

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import plotly.offline as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import logging


class TesterMain:

    def __init__(self, df,
                df_chunk,
                inp_time_len,
                init_len,
                test_period,
                out_filter_len,
                col_feats_prev,
                col_feats_curr,
                normalizer,
                mape_lens=[1, 12, 24]
    ):
        self.df = df.copy()
        self.df_chunk = df_chunk.copy()
        self.inp_time_len = inp_time_len
        self.init_len = init_len
        self.test_period = test_period
        self.out_filter_len = out_filter_len
        self.col_feats_prev = col_feats_prev
        self.col_feats_curr = col_feats_curr
        self.normalizer = normalizer
        self.mape_lens = mape_lens

        if self.test_period not in self.mape_lens:
            self.mape_lens.append(self.test_period)

        ds_chunk_counts = self.df.groupby('ChunkId').ChunkId.count()
        self.df = self.df[self.df.ChunkId.isin(list(ds_chunk_counts[ds_chunk_counts >\
             self.inp_time_len + self.test_period + self.init_len].index))].copy()

        self.df_chunk = self.df_chunk[self.df_chunk.ChunkId.isin(self.df.ChunkId.unique())].copy()

        self.chunk_ids_test = self.df_chunk.ChunkId.values

        dfs = []
        for idnow in self.chunk_ids_test:
            dfnow = self.df[self.df.ChunkId == idnow].copy()
            dfs.append(dfnow)

        self.testers = []
        for dfnow in dfs:
            testernow = Tester(dfnow,
                    self.init_len,
                    self.test_period,
                    self.inp_time_len,
                    self.out_filter_len,
                    skip=0,
                    normalizer=self.normalizer,
                col_feats_prev = self.col_feats_prev,
                col_feats_current= self.col_feats_curr,
                col_out='Yhat',
                    batchsize=-1,
                    mape_lens=self.mape_lens,
                )
            self.testers.append(testernow)

    def init(self):
        for testernow in self.testers:
            testernow.init()

    def run(self, model):
        
        self.results = []
        self.ynows = []
        self.inds = []
        self.mapes = []
        self.mape = None
        self.df_result = None

        ind = -1
        for testernow in self.testers:
            
            ind += 1
            # try:
            resnow, ynow = testernow.test_model(model, keep_results=True)
            self.results.append(resnow)
            self.ynows.append(ynow)
            self.inds.append(ind)
            self.mapes.append(testernow.mape)
            # except Exception as e:
            #     print(e)
            #     self.results.append(None)
            #     self.ynows.append(None)
            #     self.mapes.append(None)
            #     continue

        self.df_result = pd.concat([testernow.df for testernow in self.testers])
        df_result2 = self.df_result[~self.df_result['Yhat'].isnull()].copy()
        self.mape = np.mean(np.abs(df_result2['Yhat_unnorm'].values - df_result2['Moisture_unnorm'].values)\
        / df_result2['Moisture_unnorm'].values)

        self.mapes = np.array(self.mapes)

        print(f"MAPE IS: {self.mape}")

    def visualise_rand(self, additionals=None):

        ind = np.random.choice(self.inds)

        self.testers[ind].visualise(additionals=additionals)


class Tester:
    # Tester class that takes test data as dataframe and obtains predictions
    # using lstm model and MAPE scores
    # This class is same as the Batcher class, differences are commented.

    def __init__(self, df,
                 init_len,
                 test_period, # If set to 24, outputs daily predictions
                 #              At each test_period, starts a new test of length test_period
                 inp_time_len,
                 out_filter_len,
                 skip=0, # if set to 24, excludes predictions of first 24 hours from the MAPE calculations
                 normalizer=None,
                 col_dt='Datetime',
                 col_val='Moisture',
                 col_feats_prev = ['Moisture', 'temperature'],
                 col_feats_current = ['temperature', 'Day', 'Hour', 'Month'],
                 col_out = 'Yhat', # Output column name for predictions
                 batchsize=-1,
                 mape_lens=[1,12,24]): # Calculate mape predictions for first 1, 12 and 24 hours separately

        # Set parameters
        self.df = df.copy()
        self.init_len = init_len
        self.test_period = test_period
        self.inp_time_len = inp_time_len
        self.out_filter_len = out_filter_len
        
        self.skip = skip
        self.normalizer = normalizer
        self.col_dt = col_dt
        self.col_val = col_val
        self.col_feats_prev = col_feats_prev
        self.col_feats_current = col_feats_current
        self.col_out = col_out
        self.batchsize = batchsize
        self.mape_lens = mape_lens

        if self.col_out in self.df.columns:
            msg = f"Output column {col_out} is already in database!"
            logging.error(msg)
            raise Exception(msg)
        
        self.df = self.df.sort_values(self.col_dt)
        self.df = self.df.reset_index()
        
        n_nonconsecutive  = (self.df[self.col_dt].diff() != dt.timedelta(hours=1)).sum() - 1
        if n_nonconsecutive > 0:
            msg = f"Missing at least {n_nonconsecutive} hours in data!"
            logging.error(msg)
            raise Exception(msg)
        
        self.n_feat_prev = self.inp_time_len * len(self.col_feats_prev)
        self.n_feat_current = len(self.col_feats_current)
        self.n_feat = self.n_feat_prev + self.n_feat_current

        # Keep mape values in arrays for later visualisation
        self.mapes = {key: [] for key in self.mape_lens}
        self.df[self.col_out] = None

        # Normalize data
        if self.normalizer:
            self.df = self.normalizer.normalize(self.df, init=False, inplace=False)
            self.df[self.col_val+'_unnorm'] = self.normalizer.denormalize_arr(self.df[self.col_val], self.col_val)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def init(self):

        logging.info("Initializing tester object.")
        
        # At each test_period, there is another chunk. 
        # For each chunk, there is an initialization part where the network runs through true consumption values.
        # Chunk lenghts are init_len + test_period + skip

        # Initialization start indexes
        self.inds_init_starts = np.arange(self.inp_time_len, self.df.shape[0] - self.init_len - self.test_period - self.skip, self.test_period)
        # Initialization end indexes
        self.inds_init_ends = self.inds_init_starts + self.init_len
        # Test end indexes (test start indexes are inds_init_ends)
        self.inds_test_ends = self.inds_init_ends + self.test_period + self.skip

        # If last chunk is exceeds df data, remove it
        if self.inds_test_ends[-1] > self.df.shape[0]-1:
            logging.warning(f"Clipping last period since insufficient data. Last period "+\
                f"starts at {self.df.iloc[self.inds_test_ends[-2]][self.col_dt]}")
            self.inds_init_starts = self.inds_init_starts[:-1]
            self.inds_init_ends = self.inds_init_ends[:-1]
            self.inds_test_ends = self.inds_test_ends[:-1]

        # Set test chunk ids for later visualisation
        self.df['TestChunkId'] = None
        for i in range(self.inds_init_ends.shape[0]):
            self.df.loc[int(self.inds_init_ends[i]):int(self.inds_test_ends[i]), 'TestChunkId'] = i

        # Number of chunks
        self.n_data = self.inds_init_starts.shape[0]

        # Same as batcher, except skip is added
        inds_current = (np.repeat(self.inds_init_starts.reshape((-1,1)), self.init_len+self.test_period+self.skip, axis=1) + \
                        np.arange(self.init_len+self.test_period+self.skip).reshape((1,-1))).flatten()

        df_current = self.df.iloc[inds_current]
        X_current = df_current[self.col_feats_current].values.reshape((self.n_data, self.init_len+self.test_period+self.skip, self.n_feat_current))

        # Same as batcher except skip is added
        inds_prev = np.repeat(np.repeat(self.inds_init_starts.reshape(-1,1) - self.inp_time_len, self.init_len+self.test_period+self.skip, axis=1).reshape(-1, self.init_len+self.test_period+self.skip, 1), self.inp_time_len, axis=2)\
                        + np.arange(self.init_len+self.test_period+self.skip).reshape((1, self.init_len+self.test_period+self.skip, 1))\
                        + np.arange(self.inp_time_len).reshape((1,1,self.inp_time_len))

        df_prev = self.df.iloc[inds_prev.flatten()]
        X_prev = df_prev[self.col_feats_prev].values.reshape((self.n_data, self.init_len+self.test_period+self.skip, self.n_feat_prev))

        self.X = np.concatenate((X_prev, X_current), axis=2).astype(np.float32)

        self.Y = df_current[self.col_val].values.reshape((self.n_data, self.init_len+self.test_period+self.skip)).astype(np.float32)
        self.inds_y = (np.repeat(self.inds_init_ends.reshape((-1,1)) + self.skip, self.test_period, axis=1) + \
                        np.arange(self.test_period).reshape((1,-1))).flatten()

        self.X = self.X.swapaxes(0,1)
        self.Y = self.Y.transpose()

        # Get unnormalized target values for easy MAPE calculation
        self.Y_nonorm = None
        if self.normalizer:
            self.Y_nonorm = self.normalizer.denormalize_arr(self.Y, self.col_val)

        self.ind = 0
        self.is_finished = False
    
    def reset_iter(self):
        # Data is run with batches in order. 
        # Resets iter for the next test.
        self.is_finished = False
        self.ind = 0
        
    def get_batch(self, batchsize=-1):
        
        # Gets a batch data, updates self.ind, sets is_finished to true
        # If all data is run.
        
        # If batchsize is -1, use all data at once
        if batchsize == -1:
            batchsize = self.n_data
        else:
            batchsize = min(batchsize, self.n_data)

        X_now = self.X[:,self.ind:self.ind+batchsize,:]
        Y_now = self.Y[:, self.ind:self.ind+batchsize]

        self.ind += batchsize

        if self.ind >= self.n_data:
            self.is_finished = True

        return X_now, Y_now

    def get_mapes(self, Yhat):
        # Calculate mapes for different mape_len's

        Yhat = Yhat[self.init_len+self.skip:, :]
        Y_nonorm = self.Y_nonorm[self.init_len+self.skip:, :]

        Yhat = self.normalizer.denormalize_arr(Yhat, self.col_val)
        
        R = np.abs(np.abs(Yhat - Y_nonorm) / Y_nonorm)

        results = {}
        for mape_len in self.mape_lens:
            results[mape_len] = R[:mape_len,:].mean()

        self.mape = results[self.mape_lens[-1]]

        return results

    def test_model(self, model, keep_results=False):
        # using model, predict the consumptions and calculate mapes
        # works iteartively using batches.

        self.reset_iter()

        self.Yhat = np.zeros_like(self.Y)

        ind_prev = 0

        # For each batch in order:
        while self.is_finished is False:

            X_now, Y_now = self.get_batch(self.batchsize)

            # Calculate predictions using forward_moving function of the lstm model.
            Yhat_now = model.forward_moving(X_now, Y_now, self.init_len,
                self.n_feat_prev-1, len(self.col_feats_prev), self.out_filter_len).numpy().transpose()

            # Set the corresponding predictions
            self.Yhat[:,ind_prev:self.ind] = Yhat_now

            ind_prev += Y_now.shape[1]

        self.Yhat_sub = self.Yhat[self.init_len+self.skip:,:]

        # Get mapes
        results = self.get_mapes(self.Yhat)

        # if keep results, append mape values to the arrays
        if keep_results:
            for key, val in results.items():
                self.mapes[key].append(val)

        # Set the prediction columns and also the de-normalized predictions
        self.df.loc[self.inds_y, self.col_out] = self.Yhat_sub.transpose().flatten()
        self.df[self.col_out+'_unnorm'] = self.normalizer.denormalize_arr(self.df[self.col_out], self.col_val)

        return results, self.Yhat

    def visualise(self, additionals=None):

        dfnow = self.df

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        ynow = 0

        fig.add_trace(
                go.Scatter(x=dfnow['Datetime'],
                        y=dfnow['Moisture_unnorm'],
                        name='True Moisture',
                        mode='lines+markers',
                        marker={'size': 2, 'color': 'blue'},
                        line={'color':'blue', 'width': 1}
                        ),
                secondary_y=False,
            )

        for i in self.df.TestChunkId.unique():
            if i is None:
                continue
            dftmp = self.df[self.df['TestChunkId'] == i]
            colornow = 'green' if i%2 == 0 else 'orange'

            fig.add_trace(
                go.Scatter(x=dftmp['Datetime'],
                        y=dftmp['Yhat_unnorm'],
                        name=None,
                        mode='lines+markers',
                        marker={'size': 2, 'color': colornow},
                        line={'color':colornow, 'width': 1.5}
                        ),
            )
        
        if additionals:
            for additional in additionals:
                fig.add_trace(
                        go.Scatter(x=dfnow['Datetime'],
                                y=dfnow[additional],
                                name='True Moisture',
                                mode='lines+markers',
                                marker={'size': 2, 'color': 'blue'},
                                line={'color':'blue', 'width': 1}
                                ),
                        secondary_y=True,
                    )
        
        fig.show()
    