import pandas as pd
import numpy as np
import logging
import pickle

from algorithms.auxiliary_project_code import create_bins_combination, Alternative_threshold_computation
from algorithms.qtLibrary.libquanttree import pearson_statistic, QuantTreeUnivariate, ChangeDetectionTest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import  mean_absolute_error
from sklearn.metrics import  r2_score

from sklearn.preprocessing import Normalizer


logger = logging.getLogger('Logging_data_creation')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

STATISTIC = pearson_statistic

class Neural_Network:

    def __init__(self):
        self.training_set_location ='algorithms/trainingSet_NN.csv'
        try:
            with open('Networks.pickle', 'rb') as file:
                dictionary = pickle.load(file)
            self.model = dictionary['Model']
            self.normalizer = dictionary['Normalizer']
        except:
            self.model, self.normalizer = self.create_trained_modelAndNormalizer()
            self.store_model_and_normalizer(self.model, self.normalizer)
        return

    def create_trained_modelAndNormalizer(self):
        print('Training')
        with open(self.training_set_location, 'rb') as file:
            frame = pd.read_csv(file)
        normalizer = Normalizer()
        training_thresholds = frame.iloc[:-100, -1]
        training_rich_histograms = frame.iloc[:-100, 1:-1]
        training_used_histograms = normalizer.fit_transform(training_rich_histograms)

        test_thresholds = frame.iloc[-100:, -1]
        test_rich_histograms = frame.iloc[-100:, 1:-1]
        test_used_histograms = normalizer.transform(test_rich_histograms)

        n_nodes = 32
        train_an_other_network = True
        error = 0
        best_error = 1000
        best_complexity = 0
        index = 1
        layers_list = [n_nodes, n_nodes]
        while train_an_other_network and len(layers_list) < 10:
            layers_list.append(n_nodes)
            model = MLPRegressor(tuple(layers_list), verbose=True, early_stopping=False,
                                 learning_rate_init = 0.0005,
                                 alpha = 0, batch_size = 100,
                                 max_iter=800, solver= 'adam',    # lbfgs
                                 learning_rate='adaptive', n_iter_no_change=40)  # invscaling
            model.fit(training_used_histograms, training_thresholds)

            test_predictions = model.predict(test_used_histograms)
            last_error = mean_absolute_error(test_thresholds, test_predictions)
            logger.debug('Test performsnce')
            print(last_error, r2_score(test_thresholds, test_predictions))
            if last_error < best_error:
                best_error = last_error
                best_complexity = layers_list
            if last_error < 2 and last_error < error -0.4:
                train_an_other_network = False
            else:
                error = last_error

        final_normalizer = Normalizer()
        x_data = frame.iloc[:, 1:-1]
        to_normalize = np.array(x_data)
        all_used_histograms = final_normalizer.fit_transform(to_normalize)
        all_used_thresholds = frame.iloc[:, -1]

        print('Final model training')

        final_model = model = MLPRegressor(tuple(best_complexity), verbose=True, early_stopping=False,
                                 learning_rate_init = 0.001,
                                 alpha = 0, batch_size = 100,
                                 max_iter=800, solver= 'adam',    # lbfgs
                                 learning_rate='adaptive')
        final_model.fit(all_used_histograms, all_used_thresholds)

        return final_model, final_normalizer

    def predict_value(self, bins: np.array, ndata: int, alpha: float):
        bins = list(bins)
        rich_histogram = list(np.sort(bins))
        rich_histogram.append(ndata)
        rich_histogram.append(alpha[0])
        input_data = np.array(rich_histogram)
        used_input_data = self.normalizer.transform(input_data.reshape(1, -1))
        predicted = self.model.predict(used_input_data.reshape(1, -1))
        return predicted

    def store_model_and_normalizer(self, model, normalizer):
        with open('Networks.pickle', 'wb') as file:
            pickle.dump({'Model': model, 'Normalizer': normalizer}, file)
        return




def create_NeuralNetwork_training_set( binsNumber: int, dataNumber: int, from_here_asymptotic: int, nu: int,
                            max_data_number = 3000, statistic = STATISTIC):

    allData = np.zeros([dataNumber, binsNumber + 3])
    thresholds = np.zeros(dataNumber)
    for number in range(dataNumber):
        if number % 10 == 1:
            logger.debug('Data number ' + str(number) + ' out of ' + str(dataNumber))
            logger.debug('Last computed threshold is ' + str(thresholds[number - 1]))
        allData[number][:-3] = create_bins_combination(binsNumber)
        allData[number][-3] = np.random.randint(30, max_data_number)
        allData[number][-2] = np.random.random()/9
        alpha = [allData[number][-2]]

        if allData[number][-1] < from_here_asymptotic:
            tree = QuantTreeUnivariate(allData[number][:-3])
            tree.ndata = int(allData[number][-3])
            test = ChangeDetectionTest(tree, nu, statistic)
            thresholds[number] = test.estimate_quanttree_threshold(alpha, 6000)
        else:
            thresholds[number] = Alternative_threshold_computation(allData[number][:-2].compute_threshold(alpha, 4000))
        allData[number][-1] = thresholds[number]

    frame = pd.DataFrame(allData)
    frame.to_csv('trainingSet_NN.csv')


create_NeuralNetwork_training_set(binsNumber=32, nu=32, dataNumber=500000,from_here_asymptotic=4000,max_data_number=3000)