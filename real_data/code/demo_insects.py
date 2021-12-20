#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:43:58 2021

@author: luca
"""

import numpy as np
from scipy.io import loadmat

folder_dataset = '../real_world_datasets/'
filename = "Insects_AbruptImbalanced.mat"

# questo dataset contiene 6 diversi concept
# noi li consideriamo come 6 diverse distribuzioni
concept0 = 0
dataset = loadmat(folder_dataset + filename)
stationary_data = dataset['phi_{}'.format(concept0)]

ndata, d = stationary_data.shape
print(stationary_data.shape)

N = 4096
l_sequence = 1000
cp = 500

idx_training = np.random.choice(ndata, N, replace=False)
# dataset di training
data_training = stationary_data[idx_training, :]

idx_test = np.setdiff1d(np.arange(0, ndata), idx_training)
# questo crea una sequenza stazionaria
idx_batch = np.random.choice(idx_test, l_sequence, replace=True)
sequence0 = np.copy(stationary_data[idx_batch, :])

concept1 = 1
alternative_data = dataset['phi_{}'.format(concept1)]
ndata1, d = alternative_data.shape
# questo crea una sequenza con change point a cp con phi1=concept1
idx_batch = np.random.choice(idx_test, cp, replace=True)
pre = np.copy(stationary_data[idx_batch, :])

idx_batch = np.random.choice(ndata1, l_sequence-cp, replace=False)
post = np.copy(alternative_data[idx_batch, :])

sequence1 = np.concatenate((pre,post))