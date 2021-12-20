import algorithms.incremental_QuantTree as iqt
import algorithms.qtLibrary.libquanttree as qt
import numpy as np
import pandas as pd
import logging
from experiments_dataHandler import Data
from algorithms.neural_network import  Neural_Network


logger = logging.getLogger('Logging_data_creation')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

NU = 32
BINS_NUMBER = 32
N_VALUES = [64, 1024] # [64, 128, 256, 512, 1024]
INITIAL_BINS = np.ones(BINS_NUMBER) / BINS_NUMBER
M = [32, 1024]
EXPERIMENTS = 10000
STATISTIC = qt.pearson_statistic
DIMENSIONS = 2
ALPHA = [0.001]
results = None


def add_row_to_dataframe(N, pi_values, stat, results=None):
    bins_names = ['Pi_number_'+ str(index) for index in range(len(INITIAL_BINS))]
    partial_row = zip(bins_names, pi_values)
    row = {
        'Initial Training Size':     M,
        'Rest of the data\'s size' : N,
        'Dimensions':                DIMENSIONS,
        'Stat value':                stat,
    }
    row.update(partial_row)
    if results is None:
        results = pd.DataFrame(row, index=[0])
    else:
        results = results.append(row, ignore_index=True)
    return results


def create_data(results):
    for N in N_VALUES:
        logger.debug('Testing with ' + str(N) + ' data')
        handler = Data()
        incremental = iqt.Incremental_Quant_Tree(INITIAL_BINS)
        training_set, rest_of_the_data = handler.generate_data_for_exp(M, N)
        incremental.build_histogram(training_set)
        incremental.modify_histogram(rest_of_the_data)
        for exp in range(EXPERIMENTS):
            batch = handler.generate_batch()
            stat = STATISTIC(incremental, batch)
            results = add_row_to_dataframe(N, incremental.pi_values, stat, results)
    results.to_csv('FPR_exp_results.csv')


class Forest:
    def __init__(self):
        return

    def train(self, training_set):
        return

    def store_necessary_things(self):
        return

    def play_with_batch(self, batch):
        self.store_necessary_things()
        return


class Incremental_Forest(Forest):

    def __init__(self):
        return

    def update_forest(self, data):
        return


class QuantForest(Forest):
    def __init__(self):
        return



def run_experiment(N, M, SKL):
    if N < M:
        return
    else:
        incremental_Forest = Incremental_Forest()
        quantForest = QuantForest()
        data = Data()
        training_set, rest_of_the_data = data.generate_data_for_exp(M, N)

        incremental_Forest.train(training_set)
        incremental_Forest.update_forest(rest_of_the_data)
        quantForest.train(training_set + rest_of_the_data)

        for skl in SKL:
            batch = data.generate_batch(skl)
            incremental_Forest.play_with_batch(batch)
            quantForest. play_with_batch(batch)
    return




frame = pd.read_csv('offline_results.csv')

frame_0 = frame.loc[frame['Rest of the data\'s size'] == 1024]
values = list(frame_0['Stat value'])
probs = list(frame_0.iloc[0, - BINS_NUMBER :])

tree = qt.QuantTree(probs)
tree.ndata = int(frame_0.loc[0, 'Rest of the data\'s size']) + int(frame_0.loc[0, 'Initial Training Size'])
network = Neural_Network()
for index in range(100):
    threshold = network.predict_value(probs, ndata=tree.ndata, alpha=[(index+1)/1000])
    # threshold = qt.ChangeDetectionTest(tree, int(NU), STATISTIC).estimate_quanttree_threshold(ALPHA, 50000)
    false_alarms = [value for value in values if value > threshold]
    logger.debug('FPR = ' + str(len(false_alarms) / len(values)) + '. Alpha is '+ str((index+1)/1000))

