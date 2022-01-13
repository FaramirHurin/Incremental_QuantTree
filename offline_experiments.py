import algorithms.incremental_QuantTree as iqt
import algorithms.qtLibrary.libquanttree as qt
import numpy as np
import pandas as pd
import logging
from experiments_dataHandler import Data
import matplotlib.pyplot as plt
from algorithms.neural_network import  Neural_Network
from algorithms.auxiliary_project_code import create_bins_combination


logger = logging.getLogger('Logging_data_creation')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

NU = 32
BINS_NUMBER = 32
N_VALUES = [256] # [64, 128, 256, 512, 1024]
INITIAL_BINS = create_bins_combination(BINS_NUMBER)
M_values = [32]
EXPERIMENTS = 5000
trees_number = 2
STATISTIC = qt.pearson_statistic
DIMENSIONS = [16]
ALPHA = [0.001]
results = None
SKL_list =[0] # ,1 , 2, 6, 10


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


def store_experiments_result(M, skl, trees_and_values, results = None):
    for tree_number in range(len(trees_and_values)):
        tree = trees_and_values[tree_number]['Tree']
        stat_value = trees_and_values[tree_number]['Stat']
        tree_type = trees_and_values[tree_number]['Forest_type']
        tree_code = trees_and_values[tree_number]['tree_code']
        results = add_row_to_dataframe(M= M, N=tree.ndata, skl=skl, pi_values=tree.pi_values,
                                       stat_value=stat_value,tree_type=tree_type, tree_code=tree_code, results=results)
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



def run_experiment(nu, N, M, SKL, statistic, dimension, experiments, results):
    i = 0
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
            i += 1
            logger.debug('Running exp number: ' + str(i))
            for experiment in range(experiments):
                batch = data.generate_batch(nu, skl)
                incremental_results = incremental_Forest.play_with_batch(batch=batch)
                results = store_experiments_result(M= M, skl=skl, trees_and_values=incremental_results, results=results)
                normal_results = quantForest.play_with_batch(batch=batch)
                results = store_experiments_result(M= M, skl=skl, trees_and_values=normal_results, results=results)
    return results

def create_experiments_results(N_VALUES, M_values, NU, SKL_list, STATISTIC, experiments, results):
    for N in N_VALUES:
        for M in M_values:
            for dimension in DIMENSIONS:
                results = run_experiment(nu=NU, N=N, M=M, SKL=SKL_list, statistic=STATISTIC,
                                         dimension=dimension, experiments=experiments, results=results)
    data_frame = pd.DataFrame(results)
    data_frame.to_csv('offline_results.csv')


"""logger.debug('Create experiments')
create_experiments_results(N_VALUES=N_VALUES, M_values=M_values, NU=NU, SKL_list=SKL_list,
                           STATISTIC=STATISTIC, experiments=EXPERIMENTS, results=results)

"""
frame = pd.read_csv('offline_results.csv')
TREE_TYPE = 'Incremental'
frame_0 = frame.loc[(frame['Total data size'] == 256) & (frame['Tree tpye']== TREE_TYPE)]
tree_codes = np.unique(np.array(frame_0.loc[:, 'Tree code']))

network = Neural_Network()

ALPHA_TO_CHECK = 8
alpha_list = [(index+1)/(ALPHA_TO_CHECK * 15) for index in range(ALPHA_TO_CHECK)]
FPR_list = []

USES_NN = True

logger.debug('Does it use NN?' + str(USES_NN))

for alpha in alpha_list:
    accepted_for_alpha = 0
    if TREE_TYPE == 'Normal':
        tree_code = tree_codes[0]
        this_tree = frame_0.loc[frame_0['Tree code'] == tree_code]
        pi_values = np.array(this_tree.iloc[0, -BINS_NUMBER:])
        n_data = int(this_tree.loc[:, 'Total data size'].head(1))
        tree = qt.QuantTree(pi_values)
        tree.ndata = n_data
        threshold = qt.ChangeDetectionTest(model=tree, nu=NU,
                                         statistic=qt.pearson_statistic).estimate_quanttree_threshold(alpha=[alpha],
                                                                                                  B=18000)
    for tree_code in tree_codes:
        this_tree = frame_0.loc[frame_0['Tree code'] == tree_code]
        pi_values = np.array(this_tree.iloc[0, -BINS_NUMBER:])
        n_data = int(this_tree.loc[:, 'Total data size'].head(1))
        stat_values = this_tree.loc[:, 'Stat value'].values
        assert sum(pi_values) == 1
        if USES_NN:

            threshold = network.predict_value(bins=pi_values, ndata=n_data, alpha=[alpha])[0]

            tree = qt.QuantTree(pi_values)
            tree.ndata = n_data
            alternative_trheshold = qt.ChangeDetectionTest(model=tree, nu=NU, statistic=qt.pearson_statistic).\
                estimate_quanttree_threshold(alpha=[alpha], B=8000)
            print(threshold, alternative_trheshold)

        elif TREE_TYPE == 'Incremental':
            tree = qt.QuantTree(pi_values)
            tree.ndata = n_data
            threshold = qt.ChangeDetectionTest(model=tree, nu=NU, statistic=qt.pearson_statistic).\
                estimate_quanttree_threshold(alpha=[alpha], B=9000)

        numberOfAccepted_for_this_tree = len([value for value in stat_values if  value > threshold])
        accepted_for_alpha += numberOfAccepted_for_this_tree
    FPR_list.append(accepted_for_alpha / frame_0.shape[0])
    print(alpha, FPR_list[-1])

plt.plot(alpha_list, alpha_list)
plt.plot(alpha_list, FPR_list)
if USES_NN:
    plt.title('False positve rate: with NN')
else:
    plt.title('False positve rate: without NN')
plt.show()