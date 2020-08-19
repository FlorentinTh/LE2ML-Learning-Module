import os
import dotenv
import requests
import urllib3
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
import pickle

# Read a constant from .env file
dotenv.load_dotenv()
print(os.getenv('TEST_ENV_VAR'))

# Request API
urllib3.disable_warnings()
request = requests.get('https://localhost:3000/api/v1/hello', verify=False)
result = json.loads(request.text)
print(result["message"])

# Open conf file and read a property value;
conf_file_path = os.path.join(
    os.path.abspath(os.path.curdir),
    'job', 'conf.json')
with open(conf_file_path) as conf_file:
    algo_name = json.load(conf_file)["algorithm"]["name"]
print(algo_name)

# Read CSV with Panda
data_file_path = os.path.join(
    os.path.abspath(os.path.curdir),
    'job', 'data.csv')
data = pd.read_csv(data_file_path, skiprows=1, header=None)
num_col = len(data.columns)

data_numeric = data.apply(pd.to_numeric, errors='coerce')
X = data_numeric.loc[:, :num_col - 2].values

y = data.loc[:, num_col - 1].values

# le = LabelEncoder()
# y = le.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=0.66, random_state=1)

stdsc = StandardScaler()
# X_train_std = stdsc.fit_transform(X_train)
# X_test_std = stdsc.transform(X_test)

X_std = stdsc.fit_transform(X)

classifier = RandomForestClassifier(
    n_estimators=300, criterion='entropy', max_features='log2', n_jobs=-1)

# classifier = KNeighborsClassifier(
#     n_neighbors=1, metric='manhattan', algorithm='brute')

classifier.fit(X_std, y)
model_file_path = os.path.join(
    os.path.abspath(os.path.curdir),
    'job', 'test.model')
model_save = pickle.dump(classifier, open(model_file_path, 'wb'))

test_file_path = os.path.join(
    os.path.abspath(os.path.curdir),
    'job', 'test.csv')
test_data = pd.read_csv(test_file_path, skiprows=1, header=None)
test_data_numeric = test_data.apply(pd.to_numeric, errors='coerce')
X_test = test_data_numeric.values
saved_model = pickle.load(open(model_file_path, 'rb'))
pred = saved_model.predict(X_test)
print(pred)
# print('-------------')
# print("Accuracy: %.2f%%" % (accuracy*100.0))
# print('-------------')


# classifier.fit(X_train_std, y_train)
# model = classifier.predict(X_test_std)

# accuracy = metrics.accuracy_score(y_test, model)
# kappa = metrics.cohen_kappa_score(y_test, model)

kfold = KFold(n_splits=10, random_state=100, shuffle=True)
# cross_val = cross_val_score(classifier, X_std, y, cv=kfold)
cross_val = cross_val_predict(classifier, X_std, y, cv=kfold)
accuracy = metrics.accuracy_score(y, cross_val)
f1_score = metrics.f1_score(y, cross_val, average='weighted')
kappa = metrics.cohen_kappa_score(y, cross_val)
print("Accuracy: %.2f%%" % (accuracy*100.0))
print("F1-Score: %.2f%%" % (f1_score*100.0))
print("Kappa: %.2f%%" % (kappa*100.0))
print('-------------')
conf_mat = metrics.confusion_matrix(y, cross_val)
conf_mat_array = pd.DataFrame(conf_mat).to_csv(os.path.join(
    os.path.abspath(os.path.curdir),
    'job', 'output.csv'), header=None, index=None)

# print(results_kfold)
# print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
