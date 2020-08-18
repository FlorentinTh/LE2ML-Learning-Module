import os
import dotenv
import requests
import urllib3
import json
import pandas
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics

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
data = pandas.read_csv(data_file_path, skiprows=1, header=None)
num_col = len(data.columns)

data_numeric = data.apply(pandas.to_numeric, errors='coerce')
X = data_numeric.loc[:, :num_col - 2].values

y = data.loc[:, num_col - 1].values

le = LabelEncoder()
y = le.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=0.66, random_state=1)

stdsc = StandardScaler()
# X_train_std = stdsc.fit_transform(X_train)
# X_test_std = stdsc.transform(X_test)

X_std = stdsc.fit_transform(X)

classifier = KNeighborsClassifier(
    n_neighbors=1, metric='manhattan', algorithm='brute')

# classifier = RandomForestClassifier(
#     n_estimators=300, criterion='entropy', max_features='log2', n_jobs=-1)

# classifier.fit(X_train_std, y_train)
# model = classifier.predict(X_test_std)

# accuracy = metrics.accuracy_score(y_test, model)
# kappa = metrics.cohen_kappa_score(y_test, model)

kfold = KFold(n_splits=10, random_state=100, shuffle=True)
results_kfold = cross_val_score(classifier, X_std, y, cv=kfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
