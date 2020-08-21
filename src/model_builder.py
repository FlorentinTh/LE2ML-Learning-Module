import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class ModelBuilder:
    @staticmethod
    def _buildKNN(properties):
        n_neighbors = properties['num_neighbors']

        algorithm = properties['search_algorithm']
        if properties['search_algorithm'] == 'linear':
            algorithm = 'brute'

        metric = properties['distance']

        return KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            algorithm=algorithm,
            n_jobs=-1)

    @staticmethod
    def _buildRandomForest(properties, feature_count):
        n_estimators = properties['num_trees']
        criterion = properties['criterion']

        max_features = properties['max_features']
        if max_features == 'demi_sqrt':
            max_features = math.floor(math.sqrt(feature_count) / 2)

        return RandomForestClassifier(
            n_estimators=n_estimators, criterion=criterion,
            max_features=max_features, n_jobs=-1)
