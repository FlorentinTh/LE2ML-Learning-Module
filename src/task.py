import os
import json


class Task:
    def __init__(self):
        self.__name = 'learning'
        self.__state = 'started'
        self.__user = os.getenv('DATA_USER_ID')
        self.__job = os.getenv('DATA_JOB_ID')

    def start():
        pass

    def get_conf_file(self):
        conf_file_path = os.path.join(
            os.getenv('DATA_BASE_PATH'),
            self.__user,
            'jobs',
            self.__job,
            'conf.json')

        with open(conf_file_path) as conf_file:
            json_file = json.load(conf_file)

        return json_file

#    def success():
#        pass
#         # urllib3.disable_warnings()
#         # certs_path = os.path.join(
#         #     os.path.abspath(os.path.curdir),
#         #     '.certs')
#         # request = requests.get(
#         #     'https://localhost:3000/api/v1/hello',
#         #     verify=os.path.join(certs_path, 'cert.pem'))
#         # result = json.loads(request.text)
#         # print(result["message"])


#     def error():
#         pass
