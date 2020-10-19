


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



class LSTM(nn.Module):
    
    def __init__(self, size_in, size_lstm, hiddens_before=None, hiddens_after=None,
                 lr_init=0.0001, use_gpu=False):
        
        super(self.__class__, self).__init__()
        
        # Set parameters
        self.size_in = size_in
        self.size_lstm = size_lstm
        self.hiddens_before = hiddens_before
        self.hiddens_after = hiddens_after
        self.lr_init = lr_init
        self.use_gpu = use_gpu

        # Standard layers before LSTM

        self.layers_before = nn.ModuleList()
        self.n_layers_before = 0
        size_in_now = size_in
        if hiddens_before is not None:
            sizes_in = [self.size_in] + hiddens_before[:-1]
            for ind in range(len(sizes_in)):
                self.layers_before.append(nn.Linear(sizes_in[ind], hiddens_before[ind]))
            self.n_layers_before = len(hiddens_before)
            size_in_now = hiddens_before[-1]

        # LSTM layers

        self.lstm = nn.LSTM(input_size=size_in_now, hidden_size=size_lstm, num_layers=1)

        # Standard layers after LSTM

        self.layers_after = nn.ModuleList()
        self.n_layers_after = 0

        if hiddens_after is not None:
            sizes_in = [self.size_lstm] + hiddens_after[:-1]
            for ind in range(len(sizes_in)):
                self.layers_after.append(nn.Linear(sizes_in[ind], hiddens_after[ind]))
            self.n_layers_after = len(hiddens_after)
        
        # Initialize iterators
        self.epoch = 0
        self.iter = 0
        self.losses_now = []
        self.losses_all = []
        
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr_init)

        if self.use_gpu:
            self.move_gpu()

    def move_gpu(self):
        if self.use_gpu:
            logging.warning("Already on GPU, moving anyway!")

        self.cuda()
        self.use_gpu = True

    def move_cpu(self):
        if self.use_gpu is False:
            logging.warning("Already on CPU, moving anyway!")
        
        self.cpu()
        self.use_gpu = False
    
    def forward_seq(self, X, states=None):
        # Forward pass
        
        if self.layers_before:
            for ind in range(self.n_layers_before):
                X = F.relu(self.layers_before[ind](X))

        Y, states = self.lstm(X, states)
        
        if self.layers_after:
            for ind in range(self.n_layers_after):
                Y = self.layers_after[ind](Y)
                if ind < self.n_layers_after -1:
                    Y = F.relu(Y)

        return Y, states

    def mse_loss(self, Yhat, Y, mask):
        # Calculate mean-squarred error loss value
        # Predictions where mask is 0 are ignored.

        mask.requires_grad = False
        losses = F.mse_loss(Yhat, Y, reduction='none')

        return torch.sum(losses * mask) / torch.sum(mask)

    def save_network(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
    def train_model(self, batcher_train, tester=None, test_every=1, max_epochs=10, batchsize=100, save_every=None, save_folder=None):
        
        # Pytorch function that tells the network that training mode is on.
        self.train()

        time_batch = 0
        time_fb = 0

        # Number of iterations for each epoch (calculated given n_examples, sequence length(minlen is chosen) and batchsize)
        n_iter = ((batcher_train.n_examples // batcher_train.minlen) + 1) // batchsize + 1

        # For each epoch:
        for self.epoch in range(self.epoch, max_epochs):
            
            # Save network if save_every is set and hit
            if save_every is not None and self.epoch>0 and self.epoch % save_every == 0:
                self.save_network(os.path.join(save_folder, 'network_current.pkl'))
                batcher_train.save(os.path.join(save_folder, 'batcher_current.pkl'))
                if tester:
                    tester.save(os.path.join(save_folder, 'tester_current.pkl'))
            
            self.iter = 0
            lognow = f"Epoch: {self.epoch}, Iter: {self.iter}"
            islog = False

            if self.epoch != 0:
                # if not the first epoch, calculate mean loss value for the previous epoch and create log string
                lossnow = np.mean(self.losses_now)
                self.losses_all.append(lossnow)
                lognow += f", L: {lossnow:.6f}"
                lognow += f", t-b: {time_batch:.2f}"
                lognow += f", t-fp: {time_fb:.2f}"
                self.losses_now = []
                islog = True

            if tester and self.epoch % test_every == 0:
                # If tester is given, get mape values on test data and add to the log string
                results, yhat = tester.test_model(self,keep_results=True)
                lognow += ", MAPES:" + ", ".join([f"{key}: {val:.6f}" for key,val in results.items()])
                islog = True

            if islog:
                logging.info(lognow)
                print(lognow)

            time_batch = 0
            time_fb = 0
            
            torch.set_grad_enabled(True)

            if self.epoch == 0:
                logging.info(f"Epoch-0:")

            # For each iteration:
            for self.iter in range(self.iter, n_iter):
                
                # Make parameter gradients zero
                self.optimizer.zero_grad()

                t0 = time.time()
                # Get batch data and convert to torch tensors
                X, Y, mask = batcher_train.get_batch_rand(batchsize)
                X = torch.from_numpy(X)
                Y = torch.from_numpy(Y)
                mask = torch.from_numpy(mask)

                # Move to gpu if using gpu
                if self.use_gpu:
                    X = X.cuda()
                    Y = Y.cuda()
                    mask = mask.cuda()
                time_batch += time.time() - t0
                
                t0 = time.time()

                # Forward pass
                Yhat, states = self.forward_seq(X)
                
                # Calculate loss
                loss = self.mse_loss(Yhat, Y, mask)
                
                # Apply backward pass
                loss.backward()
                
                # Update network parameters
                self.optimizer.step()

                # Keep loss values for later visualisation
                self.losses_now.append(loss.item())

                time_fb += time.time() - t0

                if self.epoch == 0:
                    # report loss values at first epoch since changes are sudden tat epoch0
                    logging.info(f"Epoch-0, Iter-{self.iter}: Loss: {loss.item():.6f}")
                    print(f"Epoch-0, Iter-{self.iter}: Loss: {loss.item():.6f}")
                
                
    def forward_moving(self, X, Y, test_start, ind_last_y, len_col_prev, outlen):
        # Forward function for test
        # Predictions from previous time-points are used as input feature for current time
        
        batchsize = X.shape[1]
        test_len = X.shape[0]

        # Convert data to torch tensors
        Y = torch.from_numpy(Y)
        X = torch.from_numpy(X)

        # intialize output array
        y_out = torch.zeros((batchsize, test_len))
        y_counts = torch.zeros((batchsize, test_len))

        y_out.requires_grad = False
        y_counts.requires_grad = False

        # yhat keeps predictions for each time-step
        yhat = None

        # states keep lstm states that are given input to the next time-step
        states = None

        # move data to gpu if necessary
        if self.use_gpu:
            X = X.cuda()
            Y = Y.cuda()

        # For each time
        for t in range(test_len):
            
            # If test is started (after test_start, true values of consumptions are assumed to be unknown)
            # if t >= test_start:

            #     # Replace the true consumption values with predictions from previous time-stamps
            #     # One drawback is that number of predicted values (columns) assumed to be single
            #     # Need to modify if trying to predict multiple values.

            #     k = np.arange(1,t-test_start+1)
            #     X[t,:,ind_last_y - k*len_col_prev] = X[t-1,:,ind_last_y - k*len_col_prev + 2].data.clone()
            #     X[t,:,ind_last_y] = yhat.data.clone()

            if t >= test_start:

                k = np.arange(0, (ind_last_y + 1 - len_col_prev) // len_col_prev)
                
                X[t,:,k*len_col_prev + len_col_prev-1] = X[t-1,:,(k+1)*len_col_prev+len_col_prev-1].clone()
                # X[t,:,ind_last_y] = yhat.data.clone()
                X[t, :, ind_last_y] = y_out[:, t-1]

            # Set current data input
            xnow = X[t]
            xnow = xnow.unsqueeze(0)

            # Calculate current predictions and put into y_out
            yhat, states = self.forward_seq(xnow, states=states)
            # yhat = yhat.mean(dim=2)
            # yhat = yhat[:,:,0]
            # y_out[:,t] = yhat.data
            yhat = yhat[0].data

            outlen = min(test_len - t, outlen)
            y_out[:, t:t+outlen] = (y_counts[:, t:t+outlen] * y_out[:, t:t+outlen] + yhat[:, :outlen]) /\
                (y_counts[:, t:t+outlen] + 1)

            y_counts[:, t:t+outlen] += 1
                
        return y_out


