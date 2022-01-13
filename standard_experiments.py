import algorithms.incremental_QuantTree as iqt
import algorithms.qtLibrary.libquanttree as qt
import numpy as np
import pandas as pd
import logging
from experiments_dataHandler import Data
import matplotlib.pyplot as plt
from algorithms.neural_network import  Neural_Network


NU = 32
BINS_NUMBER = 32
N_VALUES = [64] # [64, 128, 256, 512, 1024]
INITIAL_BINS = np.ones(BINS_NUMBER) / BINS_NUMBER
M_values = [32]
EXPERIMENTS = 5000
trees_number = 20
STATISTIC = qt.pearson_statistic
DIMENSIONS = [8]
ALPHA = [0.001]
results = None
SKL_list =[0] # ,1 , 2, 6, 10


