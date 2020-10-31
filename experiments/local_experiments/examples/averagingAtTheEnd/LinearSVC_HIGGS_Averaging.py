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
from DLplatform.learning.batch.sklearnClassifiers import LogisticRegression, LinearSVC, LinearSVCRandomFF, \
    LinearSVCNystrom
from DLplatform.stopping import MaxAmountExamples
from DLplatform.dataprovisioning import BatchDataScheduler, IntervalDataScheduler
from itertools import product
from datetime import datetime
import numpy as np

LOG_CONSOLE = True

if __name__ == "__main__":

    messengerHost = 'localhost'
    messengerPort = 5672

    dim = 28  # HIGGS has 28 features

    numberOfNodes = 205
    n_components = 200
    coord_sleep_time = numberOfNodes/15
    learner = LinearSVCRandomFF
    regParam = 243
    max_example_value = 11000

    aggregator = RadonPoint()
    sync = AggregationAtTheEnd()
    dsFactory = FileDataSourceFactory(filename="../../../../data/HIGGS/HIGGS.csv",
                                      decoder=CSVDecoder(delimiter=',', labelCol=0),
                                      numberOfNodes=numberOfNodes, indices='roundRobin', shuffle=False, cache=False)

    if learner.__name__ == LinearSVCRandomFF.__name__:
        learnerFactory = SklearnBatchLearnerFactory(learner, {'regParam': regParam, 'dim': dim, 'gamma': 0.001371742,
                                                              'n_components': n_components})
    else:
        learnerFactory = SklearnBatchLearnerFactory(learner, {'regParam': regParam, 'dim': dim})

    stoppingCriterion = MaxAmountExamples(max_example_value)
    try:
        exp = Experiment(executionMode='cpu', messengerHost=messengerHost, messengerPort=messengerPort,
                         numberOfNodes=numberOfNodes, sync=sync, aggregator=aggregator,
                         learnerFactory=learnerFactory, dataSourceFactory=dsFactory,
                         stoppingCriterion=stoppingCriterion, sleepTime=0.2, dataScheduler=BatchDataScheduler,
                         minStartNodes=numberOfNodes, minStopNodes=numberOfNodes,
                         coordinatorSleepTime=coord_sleep_time)

        exp.run(learner.__name__ + '_' + aggregator.__str__())

    finally:
        # Set console output to file at src. Below code will copy and rename the file
        # to include a timestamp
        if LOG_CONSOLE:
            import os
            import shutil
            import datetime
            # TODO change code to create separate files
            src = '../../../../../Console Logs/console_logs.txt'
            dst = './Results/' + 'console_logs_' + str(datetime.datetime.now()).replace(':', '_') + '.txt'
            shutil.copyfile(src, dst)
            # if os.path.isfile(path=src) and os.path.isfile(path=dst):
            #     os.remove(path=src)
