import sys

sys.path.append("../../../../../dlapplication")
sys.path.append("../../../../../dlplatform")

from environments.local_environment import Experiment
from environments.datasources.standardDataSourceFactories import FileDataSourceFactory, SVMLightDataSourceFactory
from environments.datasources.dataDecoders.otherDataDecoders import CSVDecoder
from DLplatform.aggregating import Average
from DLplatform.aggregating import RadonPoint
from DLplatform.synchronizing.aggAtTheEnd import AggregationAtTheEnd
from DLplatform.learning.factories.sklearnBatchLearnerFactory import SklearnBatchLearnerFactory
from DLplatform.learning.batch.sklearnClassifiers import LogisticRegression
from DLplatform.stopping import MaxAmountExamples
# import multiprocessing
# import time

if __name__ == "__main__":
    messengerHost = 'localhost'
    messengerPort = 5672
    numberOfNodes = 6

    regParam = 0.01
    # dim = 4  # skin_segmentation has 4 attributes
    dim = 18  # SUSY has 18 features
    learnerFactory = SklearnBatchLearnerFactory(LogisticRegression, {'regParam': regParam, 'dim': dim})

    # dsFactory = SVMLightDataSourceFactory("../../../../data/classification/skin_segmentation.dat", numberOfNodes,
    #                                       indices='roundRobin', shuffle=False)

    dsFactory = FileDataSourceFactory(
        filename="../../../../data/SUSY/SUSY.csv",
        decoder=CSVDecoder(delimiter=',', labelCol=0), numberOfNodes=numberOfNodes, indices='roundRobin',
        shuffle=False, cache=False)

    stoppingCriterion = MaxAmountExamples(5000)

    aggregator = Average()
    sync = AggregationAtTheEnd()

    exp = Experiment(executionMode='cpu', messengerHost=messengerHost, messengerPort=messengerPort,
                     numberOfNodes=numberOfNodes, sync=sync,
                     aggregator=aggregator, learnerFactory=learnerFactory,
                     dataSourceFactory=dsFactory, stoppingCriterion=stoppingCriterion, sleepTime=0)

    exp.run("Log_reg" + "_" + aggregator.__str__())

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
