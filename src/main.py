import dotenv
import urllib3
from task import Task
from learning import Learning

dotenv.load_dotenv()
urllib3.disable_warnings()

task = Task()

try:
    print('[Container] Info: user - ' + task._user +
          ' starts learning task for job - ' + task._job)

    conf_file = task.get_conf_file()
    learning = Learning(conf_file, task._container_name)

    if conf_file['process'] == 'train':
        learning.buildModel()
        learning.trainModel()

        if conf_file['cross-validation']:
            cross_validation = learning.crossValidate()
            learning.writeConfusionMatrixToFile(cross_validation)
            results = learning.getMetrics(cross_validation)
            task.success(results=results)
        else:
            task.success()
            pass
    elif conf_file['process'] == 'test':
        predictions = learning.predict()
        learning.writePredictionsToFile(predictions)
        task.success()
    else:
        print('[Container] : Process property value unexpected.')

except Exception as error:
    print('[Container] : ' + str(error))
    task.error()
