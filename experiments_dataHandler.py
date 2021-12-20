from algorithms.auxiliary_project_code import Data_set_Handler


class Data:
    def __init__(self, dimensions):
        self.handler = Data_set_Handler(dimensions)
        return

    def generate_data_for_exp(self, M, N):
        training_set = self.handler.return_equal_batch(M)
        rest_of_the_data = self.handler.return_equal_batch(N - M)
        return training_set, rest_of_the_data

    def generate_batch(self, nu, skl=0):
        if skl == 0:
            return self.handler.return_equal_batch(nu)
        else:
            return self.handler.generate_similar_batch(nu, skl)
