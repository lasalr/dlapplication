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

    # dim = 4 #skin_segmentation has 4 attributes
    dim = 18  # SUSY has 18 features
    # dim = 28  # HIGGS has 28 features

    # # Ensure node_counts is multiple of radon number of models =
    # # dim(dimensionality of each support vector) + 1 (intercept) + 2
    # node_counts = [x for x in range(20, 201, dim+2+1)]
    # coord_sleep_times = [x/25 for x in node_counts]
    # learners = [LinearSVC, LinearSVCRandomFF]
    # regParams = [0.01, 0.001, 0.0001]
    # aggregators = [RadonPoint()]
    # max_example_values = [x for x in range(10000, 100001, 10000)]

    # Default experiment parameters (to be commented out)
    node_counts = [21]
    coord_sleep_times = [node_counts[0]/25]
    learners = [LinearSVCRandomFF]
    regParams = [4]
    aggregators = [RadonPoint()]
    max_example_values = [20000]

    sync = AggregationAtTheEnd()

    # Get total experiment count
    total_exp_count = len(list(product(zip(node_counts, coord_sleep_times), learners, regParams, aggregators, max_example_values)))
    print('total_exp_count={}'.format(total_exp_count))
    try:
        exp_count = 0
        for ((node_count, coordinator_sleep_time), learner, regParam, aggregator, max_example_value)\
                in product(zip(node_counts, coord_sleep_times), learners, regParams, aggregators, max_example_values):
            exp_count += 1
            numberOfNodes = node_count
            # Create list of n_components values ranging from all data points to 0.1% of data points
            # rff_n_components = list(set(np.linspace(max_example_value * node_count,
            #                                         max_example_value * node_count * 0.001, 30).astype(int)))

            # Default experiment (to be commented out)
            rff_n_components = [500]

            total_sub_exp_count = len(rff_n_components)
            sub_exp_count = 0

            for n_components in rff_n_components:
                sub_exp_count += 1
                print('Experiment {} of {}\nSub experiment {} of {}\n'.format(exp_count, total_exp_count, sub_exp_count,
                                                                            total_sub_exp_count),
                      'node_count =', node_count, 'coordinator_sleep_time =', coordinator_sleep_time, 'learner =',
                      learner, 'regParam =', regParam, 'aggregator =', aggregator, 'max_example_value =',
                      max_example_value, 'rff_components =', n_components)

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
                if learner.__name__ == LinearSVCRandomFF.__name__:
                    learnerFactory = SklearnBatchLearnerFactory(learner, {'regParam': regParam, 'dim': dim, 'gamma': 0.0078125, 'n_components': n_components}
                else:
                    learnerFactory = SklearnBatchLearnerFactory(learner, {'regParam': regParam, 'dim': dim})

                stoppingCriterion = MaxAmountExamples(max_example_value)
                aggregator = aggregator

                exp = Experiment(executionMode='cpu', messengerHost=messengerHost, messengerPort=messengerPort,
                                 numberOfNodes=numberOfNodes, sync=sync, aggregator=aggregator,
                                 learnerFactory=learnerFactory, dataSourceFactory=dsFactory,
                                 stoppingCriterion=stoppingCriterion, sleepTime=0.2, dataScheduler=BatchDataScheduler,
                                 minStartNodes=numberOfNodes, minStopNodes=numberOfNodes,
                                 coordinatorSleepTime=coordinator_sleep_time)

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
