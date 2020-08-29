import os
import pickle
import pandas as pd
import numpy as np
from model_builder import ModelBuilder
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold


class Learning:
    def __init__(self, conf, container_name):
        self.__user = os.getenv('DATA_USER_ID')
        self.__job = os.getenv('DATA_JOB_ID')
        self.__conf = conf
        self._container_name = container_name

        if 'parameters' in conf['algorithm']:
            self.__properties = conf['algorithm']['parameters']
        else:
            self.__properties = None

        self.__model = None
        self.__data = None
        self.__labels = None
        self.__feature_count = 0

    def __makeData(self):

        file_type = self.__conf['input']['file']['type']

        file = None
        if file_type == 'features':
            file = self.__conf['input']['file']['filename']
        else:
            file = 'features.csv'

        dataset = os.path.join(os.path.abspath(
            os.getenv('DATA_BASE_PATH')), file)

        try:
            read_data = pd.read_csv(dataset, skiprows=1, header=None)
        except Exception as error:
            raise Exception(str(error))

        num_col = len(read_data.columns)

        num_data = read_data.apply(pd.to_numeric, errors='coerce')
        data = num_data.loc[:, :num_col - 2].values

        self.__labels = read_data.loc[:, num_col - 1].values

        std_sc = StandardScaler()
        self.__data = std_sc.fit_transform(data)

        try:
            self.__feature_count = len(pd.read_csv(
                dataset, index_col=0, nrows=1).columns)
        except Exception as error:
            raise Exception(str(error))

    def __getConfusionMatrixHeader(self):
        return list(np.insert(np.unique(self.__labels), 0, '', axis=0))

    def __saveTrainedModel(self):
        model_filename = self.__conf['model'] + '.model'
        model_save_path = os.path.join(os.path.abspath(
            os.getenv('DATA_BASE_PATH')), model_filename)

        try:
            pickle.dump(self.__model, open(model_save_path, 'wb'))
        except Exception as error:
            raise Exception(str(error))

    def __openExistingModel(self):
        model_filename = self.__conf['model']

        model_file_path = os.path.join(
            os.getenv('DATA_BASE_PATH'),
            model_filename)

        try:
            return pickle.load(open(model_file_path, 'rb'))
        except Exception as error:
            raise Exception(str(error))

    def buildModel(self):
        self.__makeData()

        label = self.__conf['algorithm']['name']
        if label == 'k_nearest_neighbors':
            self.__model = ModelBuilder._buildKNN(self.__properties)
        elif label == 'random_forest':
            self.__model = ModelBuilder._buildRandomForest(
                self.__properties, self.__feature_count)
        else:
            raise Exception('Algorithm is not implemented yet.')

    def trainModel(self):
        self.__model.fit(self.__data, self.__labels)

        if self.__conf['process'] == 'train':
            self.__saveTrainedModel()

    def crossValidate(self):
        ten_fold = KFold(n_splits=10, random_state=100, shuffle=True)
        return cross_val_predict(
            self.__model, self.__data, self.__labels, cv=ten_fold)

    def predict(self):
        if self.__conf['process'] == 'test':
            file_type = self.__conf['input']['file']['type']
            test_file_path = None

            if file_type == 'features':
                filename = self.__conf['input']['file']['filename']
                test_file_path = os.path.join(
                    os.getenv('DATA_BASE_PATH'),
                    filename)
            elif file_type == 'raw':
                test_file_path = os.path.join(
                    os.getenv('DATA_BASE_PATH'),
                    'features.csv')
            else:
                raise Exception('Input file type does not exists.')

            try:
                test_data = pd.read_csv(test_file_path)
                columns = list(test_data)

                if 'label' in columns:
                    test_data = test_data.drop('label', 1)

            except Exception as error:
                raise Exception(str(error))

            num_test_data = test_data.apply(
                pd.to_numeric, errors='coerce').values
            saved_model = self.__openExistingModel()
            return saved_model.predict(num_test_data)
        else:
            raise Exception('Process entry not expected.')

    def getMetrics(self, cross_validation):
        results = []
        accuracy = metrics.accuracy_score(self.__labels, cross_validation)
        results.append(round(accuracy*100.0, 2))

        f1_score = metrics.f1_score(
            self.__labels, cross_validation, average='weighted')
        results.append(round(f1_score*100.0))

        kappa = metrics.cohen_kappa_score(self.__labels, cross_validation)
        results.append(round(kappa*100.0))

        return results

    def writeConfusionMatrixToFile(self, cross_validation):
        headers = self.__getConfusionMatrixHeader()
        conf_mat = metrics.confusion_matrix(self.__labels, cross_validation)
        conf_mat_join = np.column_stack((np.unique(self.__labels), conf_mat))

        output_path = os.path.join(
            os.getenv('DATA_BASE_PATH'),
            'matrix.csv')

        try:
            pd.DataFrame(conf_mat_join).to_csv(
                output_path, header=headers, index=False)
        except Exception as error:
            raise Exception(str(error))

    def writePredictionsToFile(self, predictions):
        output_path = os.path.join(
            os.getenv('DATA_BASE_PATH'),
            'predictions.csv')

        try:
            pd.DataFrame(predictions).to_csv(
                output_path, header=None, index=True)
        except Exception as error:
            raise Exception(str(error))
