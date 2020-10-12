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

LOG_CONSOLE = True

if __name__ == "__main__":

    messengerHost = 'localhost'
    messengerPort = 5672

    # node_counts = [x for x in range(20, 81, 10)]
    # learners = [LinearSVC, LinearSVCRandomFF]
    # regParams = [0.1, 0.01, 0.001, 0.0001]
    # aggregators = [RadonPoint(), Average()]
    # max_example_values = [x for x in range(10000, 100001, 10000)]

    node_counts = [40]
    learners = [LinearSVC]
    regParams = [0.01]
    aggregators = [RadonPoint()]
    max_example_values = [10000]

    # dim = 4 #skin_segmentation has 4 attributes
    dim = 18  # SUSY has 18 features
    # dim = 28  # HIGGS has 28 features

    sync = AggregationAtTheEnd()
    for (node_count, learner, regParam, aggregator, max_example_value) in product(node_counts, learners, regParams,
                                                                                  aggregators, max_example_values):

        numberOfNodes = node_count

        # dsFactory = SVMLightDataSourceFactory("../../../../data/classification/skin_segmentation.dat", numberOfNodes,
        # indices = 'roundRobin', shuffle = False)

        dsFactory = FileDataSourceFactory(
            filename="../../../../data/SUSY/SUSY.csv",
            decoder=CSVDecoder(delimiter=',', labelCol=0), numberOfNodes=numberOfNodes, indices='roundRobin',
            shuffle=False, cache=False)

        # dsFactory = FileDataSourceFactory(
        #     filename="../../../../data/HIGGS/HIGGS.csv",
        #     decoder=CSVDecoder(delimiter=',', labelCol=0), numberOfNodes=numberOfNodes, indices='roundRobin',
        #     shuffle=False, cache=False)

        regParam = regParam
        learnerFactory = SklearnBatchLearnerFactory(learner, {'regParam': regParam, 'dim': dim})

        stoppingCriterion = MaxAmountExamples(max_example_value)
        aggregator = aggregator

        try:

            exp = Experiment(executionMode='cpu', messengerHost=messengerHost, messengerPort=messengerPort,
                             numberOfNodes=numberOfNodes, sync=sync, aggregator=aggregator,
                             learnerFactory=learnerFactory,
                             dataSourceFactory=dsFactory, stoppingCriterion=stoppingCriterion, sleepTime=0.2,
                             dataScheduler=BatchDataScheduler, minStartNodes=numberOfNodes, minStopNodes=numberOfNodes)

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
