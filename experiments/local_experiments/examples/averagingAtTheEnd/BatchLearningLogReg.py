import sys

sys.path.append("../../../../../dlapplication")
sys.path.append("../../../../../dlplatform")

from environments.local_environment import Experiment
from environments.datasources.standardDataSourceFactories import SVMLightDataSourceFactory
from DLplatform.aggregating import Average
from DLplatform.aggregating import RadonPoint
from DLplatform.synchronizing.aggAtTheEnd import AggregationAtTheEnd
from DLplatform.learning.factories.sklearnBatchLearnerFactory import SklearnBatchLearnerFactory
from DLplatform.learning.batch.sklearnClassifiers import LogisticRegression
from DLplatform.stopping import MaxAmountExamples
import multiprocessing
import time

if __name__ == "__main__":
    messengerHost = 'localhost'
    messengerPort = 5672
    numberOfNodes = 10

    regParam = 0.01
    dim = 4  # skin_segmentation has 4 attributes
    learnerFactory = SklearnBatchLearnerFactory(LogisticRegression, {'regParam': regParam, 'dim': dim})

    dsFactory = SVMLightDataSourceFactory("../../../../data/classification/skin_segmentation.dat", numberOfNodes,
                                          indices='roundRobin', shuffle=False)
    stoppingCriterion = MaxAmountExamples(20000)

    aggregator = Average()
    sync = AggregationAtTheEnd()

    exp = Experiment(executionMode='cpu', messengerHost=messengerHost, messengerPort=messengerPort,
                     numberOfNodes=numberOfNodes, sync=sync,
                     aggregator=aggregator, learnerFactory=learnerFactory,
                     dataSourceFactory=dsFactory, stoppingCriterion=stoppingCriterion, sleepTime=0)

    exp.run("Log_reg_radon_point")

    # # Start exp.run as a process
    # p = multiprocessing.Process(target=exp.run, args=("Log_reg_radon_point",))
    # p.start()
    # timeout_sec = 300
    # # Wait or until process finishes
    # p.join(timeout_sec)
    #
    # # If thread is still active
    # if p.is_alive():
    #     print("Process killed after timeout of", timeout_sec, "seconds.")
    #     # Terminate
    #     p.terminate()
    #     p.join()
