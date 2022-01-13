import random

from algorithms.neural_network import Neural_Network
from algorithms.auxiliary_project_code import create_bins_combination
import algorithms.qtLibrary.libquanttree as qt
from sklearn.metrics import r2_score, mean_squared_error

BINS_NUMBER = 32

network = Neural_Network()
NU = 32

thresholds_hat = []
thresholds = []
thresholds2 = []

print('Everything')


for index in range(20):
    ndata = int(random.uniform(40, 100))
    bins = create_bins_combination(32)
    alpha = random.uniform(0.001, 0.05)
    threshold_hat = network.predict_value(bins, ndata, [alpha])
    thresholds_hat.append(threshold_hat)

    tree = qt.QuantTree(bins)
    tree.ndata = ndata
    threshold = qt.ChangeDetectionTest(tree, NU, qt.pearson_statistic).estimate_quanttree_threshold([alpha], 10000)
    threshold2 = qt.ChangeDetectionTest(tree, NU, qt.pearson_statistic).estimate_quanttree_threshold([alpha], 10000)
    thresholds.append(threshold)
    thresholds2.append(threshold2)

    print(threshold, threshold2, threshold_hat[0])

print('R2 is' + str(r2_score(thresholds, thresholds_hat)))
print('R2 between two runs of Monte Carlo is ' + str(r2_score(thresholds, thresholds2)))

print('Mean squared error is ' + str(mean_squared_error(thresholds, thresholds_hat)))
print('Mean squared error between two runs of MC is ' + str(mean_squared_error(thresholds, thresholds2)))