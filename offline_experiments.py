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
M_values = [32, 1024]
EXPERIMENTS = 2
trees_number = 2
STATISTIC = qt.pearson_statistic
DIMENSIONS = [8]
ALPHA = [0.001]
results = None
SKL_list =[0, 1] # , 2, 6, 10


def add_row_to_dataframe(M, N, skl, pi_values,
                                       stat_value,tree_type, tree_code, results):
    bins_names = ['Pi_number_'+ str(index) for index in range(len(INITIAL_BINS))]
    partial_row = zip(bins_names, pi_values)
    row = {
        'Initial Training Size':     M,
        'Total data size' :          N,
        'Dimensions':                DIMENSIONS,
        'Stat value':                stat_value,
        'skl':                       skl,
        'Tree tpye':                tree_type,
        'Tree code':                tree_code
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
        training_set, rest_of_the_data = handler.generate_data_for_exp(M_values, N)
        incremental.build_histogram(training_set)
        incremental.modify_histogram(rest_of_the_data)
        for exp in range(EXPERIMENTS):
            batch = handler.generate_batch()
            stat = STATISTIC(incremental, batch)
            results = add_row_to_dataframe(N, incremental.pi_values, stat, results)
    results.to_csv('FPR_exp_results.csv')

def store_experiments_result(M, skl, trees_and_values, results = None):
    for tree_number in range(len(trees_and_values)):
        tree = trees_and_values[tree_number]['Tree']
        stat_value = trees_and_values[tree_number]['Stat']
        tree_type = trees_and_values[tree_number]['Forest_type']
        tree_code = trees_and_values[tree_number]['tree_code']
        results = add_row_to_dataframe(M, tree.ndata, skl, tree.pi_values,
                                       stat_value,tree_type, tree_code, results)
    return results

#TODO Scrivere tutte le foreste
class Forest:
    def __init__(self, statistic):
        self.statistic = statistic
        return

    def train(self, training_set):
        for tree in self.forest:
            tree.build_histogram(training_set)
        return

    '''Computes the statistic between each of its trees and the batch and returns the results'''
    def play_with_batch(self, batch):
        to_return = []
        index = 0
        for tree in self.forest:
            index +=1
            stat = self.statistic(tree, batch)
            forest_type = self.type_of_forest
            tree_code = index
            to_return.append\
                ({'Tree':tree, 'Stat': stat, 'Forest_type': forest_type, 'tree_code': tree_code})
        return to_return

class Incremental_Forest(Forest):

    def __init__(self, forest_size, statistic):
        super().__init__(statistic)
        self.type_of_forest = 'Incremental'
        bins = np.ones(BINS_NUMBER) / BINS_NUMBER
        self.forest = [iqt.Incremental_Quant_Tree(bins) for index in range(forest_size)]
        return

    def update_forest(self, data):
        for tree in self.forest:
            tree.modify_histogram(data)
        return


class QuantForest(Forest):
    def __init__(self, forest_size, statistic):
        super().__init__(statistic)
        self.type_of_forest = 'Normal'
        bins = np.ones(BINS_NUMBER) / BINS_NUMBER
        self.forest = [qt.QuantTree(bins) for index in range(forest_size)]
        return



def run_experiment(nu, N, M, SKL, statistic, dimension, results):
    if N < M:
        return
    else:
        incremental_Forest = Incremental_Forest(forest_size=trees_number, statistic=statistic)
        quantForest = QuantForest(forest_size=trees_number, statistic=statistic)
        data = Data(dimensions=dimension)
        training_set, rest_of_the_data = data.generate_data_for_exp(M, N)

        incremental_Forest.train(training_set)
        incremental_Forest.update_forest(rest_of_the_data)
        quantForest.train( np.vstack((training_set, rest_of_the_data)))

        for skl in SKL:
            for experiment in range(EXPERIMENTS):
                batch = data.generate_batch(nu, skl)
                incremental_results = incremental_Forest.play_with_batch(batch)
                results = store_experiments_result(M, skl, incremental_results, results)
                normal_results = quantForest.play_with_batch(batch)
                results = store_experiments_result( M, skl, normal_results, results)


    return results


for N in N_VALUES:
    for M in M_values:
        for dimension in DIMENSIONS:
            results = run_experiment(NU, N, M, SKL_list, STATISTIC, dimension, results)
data_frame = pd.DataFrame(results)
data_frame.to_csv('offline_results.csv')

print('Fase 1 finita')

frame = pd.read_csv('offline_results.csv')

frame_0 = frame.loc[frame['Total data size'] == 1024]
values = list(frame_0['Stat value'])
probs = list(frame_0.iloc[0, - BINS_NUMBER :])

tree = qt.QuantTree(probs)
tree.ndata = int(frame_0.loc[0, 'Total data size']) + int(frame_0.loc[0, 'Initial Training Size'])
network = Neural_Network()
for index in range(100):
    threshold = network.predict_value(probs, ndata=tree.ndata, alpha=[(index+1)/1000])
    # threshold = qt.ChangeDetectionTest(tree, int(NU), STATISTIC).estimate_quanttree_threshold(ALPHA, 50000)
    false_alarms = [value for value in values if value > threshold]
    logger.debug('FPR = ' + str(len(false_alarms) / len(values)) + '. Alpha is '+ str((index+1)/1000))

