import numpy as np

from algorithms.qtLibrary.libccm import random_gaussian, compute_roto_translation, rotate_and_shift_gaussian
from algorithms.qtLibrary.libquanttree import QuantTreeUnivariate


def create_bins_combination(bins_number):
    NORMALIZER = 2
    dirich_param = np.random.dirichlet(np.ones(bins_number)) * 20
    bins = np.zeros(bins_number)
    for index in range(NORMALIZER):
        bins += np.random.dirichlet(dirich_param)
    bins = bins/NORMALIZER
    bins = (bins + np.ones(bins_number)/bins_number)/2
    sorted_bins = np.sort(bins)
    if min(bins) > 0.0001 and sum(bins) == 1:
        return sorted_bins
    else:
        return create_bins_combination(bins_number)


class Data_set_Handler:

    def __init__(self, dimensions_number):
        self.dimensions_number = dimensions_number
        self.gauss0 = random_gaussian(self.dimensions_number)
        return

    def return_equal_batch(self, B):
        return np.random.multivariate_normal(self.gauss0[0], self.gauss0[1], B)

    def generate_similar_batch(self, B, target_sKL):
        rot, shift = compute_roto_translation(self.gauss0, target_sKL)
        gauss1 = rotate_and_shift_gaussian(self.gauss0, rot, shift)
        return np.random.multivariate_normal(gauss1[0], gauss1[1], B)


class Alternative_threshold_computation:

    def __init__(self, pi_values, nu, statistic):
        self.pi_values = pi_values
        self.nu = nu
        self.statistic = statistic
        return


    def compute_cut(self):
        definitive_pi_values = np.zeros(len(self.pi_values))
        histogram = np.zeros(len(self.pi_values)+1)
        bins = []
        interval_still_to_cut = [0, 1]
        left_count = 1
        right_count = 1
        for value in self.pi_values:
            bernoulli_value = np.random.binomial(1, 0.5)
            if bernoulli_value == 0:
                interval_still_to_cut[0] = interval_still_to_cut[0] + value
                histogram[left_count] = interval_still_to_cut[0]
                definitive_pi_values[left_count-1] = value
                left_count +=1
            else:
                interval_still_to_cut[1] = interval_still_to_cut[1] - value
                histogram[-right_count-1] = interval_still_to_cut[1]
                definitive_pi_values[- right_count] = value
                right_count += 1

        histogram = np.transpose(histogram)
        histogram[0] = 0
        histogram[-1] = 1
        self.pi_values = definitive_pi_values
        tree = QuantTreeUnivariate(self.pi_values)
        tree.leaves = histogram
        return tree


    def compute_threshold(self, alpha, B):
        alpha = alpha[0]
        stats = []
        histogram = self.compute_cut()
        for b_count in range(B):
            W = np.random.uniform(0, 1, self.nu)
            thr = self.statistic(histogram, W)
            stats.append(thr)
        stats.sort()
        threshold = stats[int((1-alpha)*B)]
        return threshold
