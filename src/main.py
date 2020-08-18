import os
import dotenv
import requests
import urllib3
import json
import pandas

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
data = pandas.read_csv(data_file_path)
