import pandas as pd
import numpy as np
import logging
import pickle

from algorithms.auxiliary_project_code import create_bins_combination, Alternative_threshold_computation
from algorithms.qtLibrary.libquanttree import pearson_statistic, QuantTreeUnivariate, ChangeDetectionTest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import  mean_squared_error
from sklearn.metrics import  r2_score
import os
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger('Logging_data_creation')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

STATISTIC = pearson_statistic

class Neural_Network:

    def __init__(self):
        self.training_set_location ='algorithms/trainingSet_NN.csv'
        self.training_set_location2 = 'algorithms/trainingSet2_NN.csv'
        self.training_set_location3 = 'algorithms/trainingSet3_NN.csv'
        if os.path.exists('Networks.pickle'):
            with open('Networks.pickle', 'rb') as file:
                dictionary = pickle.load(file)
            self.model = dictionary['Model']
            self.normalizer = dictionary['Nomalizer']
        else:
            print('Picke not found')
            self.model, self.normalizer = self.create_trained_model_and_normalizer()   # , self.normalizer
            self.store_model_and_normalizer(model=self.model, normalizer=self.normalizer)
        return

    def create_trained_model_and_normalizer(self):
        print('Training')
        with open(self.training_set_location, 'rb') as file:
            frame1 = pd.read_csv(file)
        with open(self.training_set_location2, 'rb') as file2:
            frame2 = pd.read_csv(file2)
        with open(self.training_set_location3, 'rb') as file3:
            frame3 = pd.read_csv(file3)

        normalizer = StandardScaler()

        frame = frame1.append(frame2, ignore_index=True)
        frame = frame.append(frame3,  ignore_index=True)

        training_thresholds = frame.iloc[100:, -1]
        training_rich_histograms = frame.iloc[100:, 1:-1]
        training_used_histograms =  np.array(training_rich_histograms)
        normalizer.fit(training_used_histograms)
        training_used_histograms = normalizer.transform(training_used_histograms)





        test_thresholds = frame.iloc[:100:, -1]
        test_rich_histograms = frame.iloc[:100, 1:-1]

        test_used_histograms = test_rich_histograms

        n_nodes = 16
        train_an_other_network = True
        error = 0
        best_error = 1000
        best_complexity = 0
        layers_list = [n_nodes]
        index = 0


        while train_an_other_network and len(layers_list) < 5:
            index = index + 1
            logger.debug('Network number ' + str(index))
            model = MLPRegressor(tuple(layers_list), verbose=True, early_stopping=True,
                                 learning_rate_init = 0.005,
                                 alpha = 0.005, batch_size = 50,
                                 max_iter=1000, solver= 'adam',    # lbfgs
                                 learning_rate='adaptive', n_iter_no_change=50)  # invscaling
            model.fit(training_used_histograms, training_thresholds)

            test_predictions = model.predict(normalizer.transform(test_used_histograms))
            last_error = mean_squared_error(test_thresholds, test_predictions)
            for index in range(len(test_predictions)):
                logger.debug('Prediction = ' + str(test_predictions[index]) +
                             ', True value = ' +str(test_thresholds.values[index]))
            logger.debug('This network\'s performance: MSE and R2')
            print(last_error, r2_score(test_thresholds, test_predictions))

            if last_error < best_error:
                best_model = model
                best_error = last_error

            if last_error < 1.5:
                train_an_other_network = False
            else:
                layers_list.append(n_nodes)

        print('Best error =' + str(best_error))

        return best_model, normalizer

    def predict_value(self, bins: np.array, ndata: int, alpha: float):
        bins = list(bins)
        rich_histogram = list(np.sort(bins))
        rich_histogram.append(ndata)
        rich_histogram.append(alpha[0])
        input_data = np.array(rich_histogram)
        input_data = self.normalizer.transform(input_data.reshape(1, -1))
        predicted = self.model.predict(input_data.reshape(1, -1))
        return predicted

    def store_model_and_normalizer(self, model, normalizer):
        with open('Networks.pickle', 'wb') as file:
            pickle.dump({'Model': model, 'Nomalizer': normalizer}, file)
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
