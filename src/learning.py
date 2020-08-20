import os
from model_builder import ModelBuilder


class Learning:
    def __init__(self, conf):
        self.__user = os.getenv('DATA_USER_ID')
        self.__job = os.getenv('DATA_JOB_ID')
        self.__conf = conf
        self.__properties = conf['algorithm']['parameters']
        self.__model = None

    def buildModel(self):
        label = self.__conf['algorithm']['name']
        if label == 'k_nearest_neighbors':
            self.__model = ModelBuilder._buildKNN(self.__properties)
        elif label == 'random_forest':
            self.__model = ModelBuilder._buildRandomForest(self.__properties)
        else:
            # TODO
            print('error, not implemented yet')
