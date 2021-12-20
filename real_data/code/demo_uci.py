#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:30:54 2021

@author: luca
"""

import numpy as np
from scipy.io import loadmat
import sys

folder_dataset = '../real_world_datasets/'
def load_dataset_and_basepath(dataset_name: str):
    data = None
    scalingFactorMU = None
    sigmaNoise = None
    if dataset_name == 'particle':
        data = loadmat(folder_dataset + 'MiniBooNE_PID_standardized.mat')
        scalingFactorMU = 0.5  # 0.05
        sigmaNoise = 1e-3
    elif dataset_name == 'protein':
        data = loadmat(folder_dataset + 'protein_standardized.mat')
        scalingFactorMU = 0.1
        sigmaNoise = 0
    elif dataset_name == 'credit':
        data = loadmat(folder_dataset + 'new_credit.mat') #'creditCards_standardized.mat')
        scalingFactorMU = 0.1  # 0.08
        sigmaNoise = 0
    elif dataset_name == 'sensorless':
        data = loadmat(folder_dataset + 'sensorDrive_standardized.mat')
        scalingFactorMU = 0.1  # 0.0001
        sigmaNoise = 1e-3
    elif dataset_name == 'nino':
        data = loadmat(folder_dataset + 'el_nino.mat')
        scalingFactorMU = 0.1
        sigmaNoise = 0
    elif dataset_name == 'spruce':
        data = loadmat(folder_dataset + 'spruce.mat')
        scalingFactorMU = 0.1
        sigmaNoise = 0.1
    elif dataset_name == 'lodgepole':
        data = loadmat(folder_dataset + 'lodgepole.mat')
        scalingFactorMU = 0.1
        sigmaNoise = 0.1
    # elif dataset_name == 'gaussian32':
    #     data = loadmat(folder_dataset + 'gaussian_32.mat')
    #     scalingFactorMU = 0.1  # ?
    #     sigmaNoise = 0
    else:
        ValueError('Unknown Dataset Name')
    data = data['dataset']
    ndata, dim = data.shape
    data = data + np.random.multivariate_normal(np.zeros(dim), sigmaNoise * np.eye(dim), ndata)

    return data, scalingFactorMU

# questo permette di inserire il nome del dataset da usare quando lanci lo script
# i nomi li trovi nella funzione sopra
if len(sys.argv) != 2:
    print("Missing arguments:", sys.argv[0], "<dataset>")
    sys.exit()

dataset = sys.argv[1]

data, scalingFactorMu = load_dataset_and_basepath(dataset)
ndata, d = data.shape
print("shape:")
print(data.shape)

mu = np.zeros(d)
sigma = np.eye(d)
# total variance of the dataset
totvar = np.sum(data.var(0))
# "change magnitude"
sKL = 0.01
factor = np.sqrt(sKL * totvar / d)

N = 4096
l_sequence = 1000
cp = 500

idx_training = np.random.choice(ndata, N, replace=False)
# dataset di training
data_training = data[idx_training, :]

idx_test = np.setdiff1d(np.arange(0, ndata), idx_training)
# questo crea una sequenza stazionaria
idx_batch = np.random.choice(idx_test, l_sequence, replace=True)
sequence0 = np.copy(data[idx_batch, :])

# questo crea una sequenza con change point a cp
shift = factor * np.random.multivariate_normal(mu,sigma)

idx_batch = np.random.choice(idx_test, cp, replace=True)
pre = np.copy(data[idx_batch, :])

idx_batch = np.random.choice(idx_test, l_sequence-cp, replace=True)
batch_power = np.copy(data[idx_batch, :])
post = batch_power - shift

sequence1 = np.concatenate((pre,post))
