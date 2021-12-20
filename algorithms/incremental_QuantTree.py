import numpy as np

from algorithms.qtLibrary.libquanttree import QuantTree
from algorithms.neural_network import Neural_Network


class Incremental_Quant_Tree(QuantTree):

    def __init__(self, pi_values):
        super().__init__(pi_values)
        self.ndata = 0

    def build_histogram(self, data, do_PCA=False):
        super().build_histogram(data, do_PCA)
        self.ndata = len(data)

    def modify_histogram(self, data, definitive=True):
        self.pi_values = self.pi_values * self.ndata
        bins = self.find_bin(data)

        vect_to_add = np.zeros(len(self.pi_values))
        for index in range(len(self.pi_values)):
            vect_to_add[index] = np.count_nonzero(bins == index)

        self.pi_values = self.pi_values + vect_to_add
        self.ndata = self.ndata + len(data)
        self.pi_values = self.pi_values / self.ndata
        return


class Online_Incremental_QuantTree:

    def __init__(self, pi_values, alpha, statistic):
        self.tree = Incremental_Quant_Tree(pi_values)
        # bins_number = len(pi_values)
        self.alpha = alpha
        self.statistic = statistic
        self.network = Neural_Network()

        self.buffer = None
        self.change_round = None
        self.round = 0
        self.last_threshold = None
        self.last_statistic = None
        return

    def build_histogram(self, data):
        self.tree.build_histogram(data)

    def play_round(self, batch, alpha):
        threshold = self.network.predict_value(self.tree.pi_values, self.tree.ndata, alpha)
        self.last_threshold = threshold
        # threshold2 = qt.ChangeDetectionTest(self.tree, len(batch), self.STATISTIC).estimate_quanttree_threshold(self.alpha, 4000)
        stat = self.statistic(self.tree, batch)
        self.last_statistic = stat
        change = stat > threshold
        if not change:
            self.update_model(batch)
        else:
            self.change_round = self.round
        self.round += 1
        return change

    def restart(self):
        self.round = 0
        self.change_round = None
        self.buffer = None

    def update_model(self, batch):
        if self.tree.ndata < 1000:
            self.tree.modify_histogram(batch)
        """if self.buffer is not None:
            self.tree.modify_histogram(self.buffer)
        self.buffer = batch
        """
        return

