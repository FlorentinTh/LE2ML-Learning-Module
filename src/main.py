import dotenv
from task import Task
from learning import Learning

dotenv.load_dotenv()

task = Task()
conf_file = task.get_conf_file()

learning = Learning(conf_file)
learning.buildModel()
learning.trainModel()
cross_validation = learning.crossValidate()
print(learning.getMetrics(cross_validation))
learning.writeConfusionMatrixToFile(cross_validation)
