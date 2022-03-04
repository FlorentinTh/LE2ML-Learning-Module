import os
import json
import requests


class Task:
    def __init__(self):
        self._user = os.getenv('DATA_USER_ID')
        self._job = os.getenv('DATA_JOB_ID')
        self.__name = 'learning'
        self.__state = 'started'
        self._container_name = 'core-py-sk'
        self.__certs_path = os.path.join(
            os.path.abspath(os.path.curdir),
            '.certs')
        self.__base_api_url = os.getenv(
            'API_URL') + '/v' + os.getenv('API_VERSION')

    def get_conf_file(self):
        conf_file_path = os.path.join(
            os.getenv('DATA_BASE_PATH'),
            'conf.json')

        try:
            with open(conf_file_path) as conf_file:
                json_file = json.load(conf_file)
        except Exception as error:
            raise Exception(error)

        return json_file

    def success(self, results=None):
        url = self.__base_api_url + '/jobs/' + self._job + '/tasks/complete'
        headers = {'app-key': os.getenv('API_APP_KEY')}
        body = {
            "task": "learning",
            "state": "completed",
            "token": os.getenv('DATA_TOKEN'),
        }

        if results:
            body['results'] = results

        try:
            # deepcode ignore TLSCertVerificationDisabled:
            # self-signed SSL certificates...
            request = requests.post(
                url, headers=headers, data=body,
                verify=False)

            if request.status_code == 200:
                print(
                    '[API] Info: Learning task started by user: ' + self._user
                    + ' for job: ' + self._job +
                    ' successfully updated (STATUS: COMPLETED).')
            else:
                self.error()

        except requests.exceptions.RequestException as error:
            print('[API] :' + str(error))

    def error(self):
        url = self.__base_api_url + '/jobs/' + self._job + '/tasks/error?task=' + self.__name
        headers = {'app-key': os.getenv('API_APP_KEY')}

        try:
            # deepcode ignore TLSCertVerificationDisabled:
            # self-signed SSL certificates...
            request = requests.post(
                url, headers=headers,
                verify=False)

            if request.status_code == 200:
                print(
                    '[API] Info: Learning task started by user: ' + self._user
                    + ' for job: ' + self._job +
                    ' successfully updated (STATUS: FAILED).')

        except requests.exceptions.RequestException as error:
            print('[API] :' + str(error))
