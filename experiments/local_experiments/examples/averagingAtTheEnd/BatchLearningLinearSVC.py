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
from DLplatform.learning.batch.sklearnClassifiers import LinearSVC
from DLplatform.stopping import MaxAmountExamples
from DLplatform.dataprovisioning import BatchDataScheduler, IntervalDataScheduler

import tracemalloc

MEM_TRACE = False
CPU_TRACE = False
LOG_CONSOLE = True

if __name__ == "__main__":

    try:

        messengerHost = 'localhost'
        messengerPort = 5672
        numberOfNodes = 21

        regParam = 0.01
        # dim = 4 #skin_segmentation has 4 attributes
        dim = 18  # SUSY has 18 features
        # dim = 28  # HIGGS has 28 features
        learnerFactory = SklearnBatchLearnerFactory(LinearSVC, {'regParam': regParam, 'dim': dim})

        # dsFactory = SVMLightDataSourceFactory("../../../../data/classification/skin_segmentation.dat", numberOfNodes,
        # indices = 'roundRobin', shuffle = False)

        dsFactory = FileDataSourceFactory(
            filename="../../../../data/SUSY/SUSY.csv",
            decoder=CSVDecoder(delimiter=',', labelCol=0), numberOfNodes=numberOfNodes, indices='roundRobin',
            shuffle=False,
            cache=False)

        # dsFactory = FileDataSourceFactory(
        #     filename="../../../../data/HIGGS/HIGGS.csv",
        #     decoder=CSVDecoder(delimiter=',', labelCol=0), numberOfNodes=numberOfNodes, indices='roundRobin',
        #     shuffle=False,
        #     cache=False)

        stoppingCriterion = MaxAmountExamples(2000)
        aggregator = RadonPoint()  # RadonPoint()
        sync = AggregationAtTheEnd()

        exp = Experiment(executionMode='cpu', messengerHost=messengerHost, messengerPort=messengerPort,
                         numberOfNodes=numberOfNodes, sync=sync, aggregator=aggregator, learnerFactory=learnerFactory,
                         dataSourceFactory=dsFactory, stoppingCriterion=stoppingCriterion, sleepTime=0,
                         dataScheduler=BatchDataScheduler)

        exp.run("Linear_SVC" + "_" + aggregator.__str__())

    finally:
        # Set console output to file at src. Below code will copy and rename the file
        # to include a timestamp
        if LOG_CONSOLE:
            import os
            import shutil
            import datetime

            src = '../../../../../Console Logs/console_logs.txt'
            dst = '../../../../../Console Logs/console_logs_' + str(datetime.datetime.now()).replace(':', '_') + '.txt'
            shutil.copyfile(src, dst)
            # if os.path.isfile(path=src) and os.path.isfile(path=dst):
            #     os.remove(path=src)
