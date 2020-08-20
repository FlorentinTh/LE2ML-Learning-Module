import os
import dotenv
import urllib3
import requests
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
import pickle
from task import Task
from learning import Learning

dotenv.load_dotenv()

task = Task()
conf_file = task.get_conf_file()

learning = Learning(conf_file)
learning.buildModel()


# -------------------------------------

# # Make training data and labels
# data_file_path = os.path.join(
#     os.path.abspath(os.path.curdir),
#     'job', 'data.csv')
# data = pd.read_csv(data_file_path, skiprows=1, header=None)
# num_col = len(data.columns)

# data_numeric = data.apply(pd.to_numeric, errors='coerce')
# X = data_numeric.loc[:, :num_col - 2].values

# y = data.loc[:, num_col - 1].values

# # set unique lables
# labels = np.unique(y)
# labels_one = np.insert(labels, 0, '', axis=0)

# stdsc = StandardScaler()
# X_std = stdsc.fit_transform(X)


# # Train classifier
# classifier = RandomForestClassifier(
#     n_estimators=300, criterion='entropy', max_features='log2', n_jobs=-1)

# # classifier = KNeighborsClassifier(
# #     n_neighbors=1, metric='manhattan', algorithm='brute')

# classifier.fit(X_std, y)


# # save model
# model_file_path = os.path.join(
#     os.path.abspath(os.path.curdir),
#     'job', 'test.model')
# model_save = pickle.dump(classifier, open(model_file_path, 'wb'))


# # run test
# test_file_path = os.path.join(
#     os.path.abspath(os.path.curdir),
#     'job', 'test.csv')
# test_data = pd.read_csv(test_file_path, skiprows=1, header=None)
# test_data_numeric = test_data.apply(pd.to_numeric, errors='coerce')
# X_test = test_data_numeric.values
# saved_model = pickle.load(open(model_file_path, 'rb'))
# pred = saved_model.predict(X_test)
# print(pred)


# # run cross-validation
# kfold = KFold(n_splits=10, random_state=100, shuffle=True)
# cross_val = cross_val_predict(classifier, X_std, y, cv=kfold)

# # output metrics
# accuracy = metrics.accuracy_score(y, cross_val)
# f1_score = metrics.f1_score(y, cross_val, average='weighted')
# kappa = metrics.cohen_kappa_score(y, cross_val)
# print("Accuracy: %.2f%%" % (accuracy*100.0))
# print("F1-Score: %.2f%%" % (f1_score*100.0))
# print("Kappa: %.2f%%" % (kappa*100.0))
# print('-------------')
# conf_mat = metrics.confusion_matrix(y, cross_val)
# conf_mat_join = np.column_stack((labels, conf_mat))
# conf_mat_array = pd.DataFrame(conf_mat_join).to_csv(os.path.join(
#     os.path.abspath(os.path.curdir),
#     'job', 'output.csv'), header=list(labels_one), index=False)
