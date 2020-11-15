from datetime import datetime
import sys

sys.path.append("../../../../../dlapplication")
sys.path.append("../../../../../dlplatform")

from environments.local_environment import Experiment
from environments.datasources.standardDataSourceFactories import FileDataSourceFactory
from environments.datasources.dataDecoders.otherDataDecoders import CSVDecoder
from DLplatform.aggregating import RadonPoint, Average
from DLplatform.synchronizing.aggAtTheEnd import AggregationAtTheEnd
from DLplatform.learning.factories.sklearnBatchLearnerFactory import SklearnBatchLearnerFactory
from DLplatform.learning.batch.sklearnClassifiers import LinearSVC, LinearSVCRandomFF, LogisticRegression
from DLplatform.stopping import MaxAmountExamples
from DLplatform.dataprovisioning import BatchDataScheduler
import gc

LOG_CONSOLE = True

if __name__ == "__main__":

    messengerHost = 'localhost'
    messengerPort = 5672

    aggregator_list = ['RadonPoint', 'Average']
    max_example_list = [2000, 1000, 500, 200, 100, 50, 25]

    dim = 5  # SYNTHETIC1 has 5 features
    numberOfNodes = 205
    n_components = 202
    coord_sleep_time = 0.1  # 7
    learner = LinearSVCRandomFF
    regParam = 512
    gamma = 0.0078125
    exp_sleep_time = 0.1  # 2

    exp_count = 0
    for aggregator_name in aggregator_list:
        exp_count += 1

        for max_example_value in max_example_list:
            if aggregator_name == 'RadonPoint':
                aggregator = RadonPoint()
            elif aggregator_name == 'Average':
                aggregator = Average()
            else:
                print('Incorrect aggregator! Continuing to next experiment')
                continue

            exp_name = learner.__name__ + '_' + aggregator.__str__() + '_' + str(max_example_value)

            f_name = './Results/{}.txt'.format(exp_name)
            with open(f_name, 'w') as fw:
                fw.write('Starting experiment {} at {}'.format(exp_count, datetime.now()))

            sync = AggregationAtTheEnd()
            stoppingCriterion = MaxAmountExamples(max_example_value)
            dsFactory = FileDataSourceFactory(filename="../../../../data/SYNTHETIC1/split/TRAIN_SYNTHETIC_DATA.csv",
                                              decoder=CSVDecoder(delimiter=',', labelCol=0),
                                              numberOfNodes=numberOfNodes, indices='roundRobin', shuffle=False,
                                              cache=False,
                                              stoppingCriterion=stoppingCriterion)

            if learner.__name__ == LinearSVCRandomFF.__name__:
                print('Using {} random fourier features with gamma of {}'.format(n_components, gamma))
                learnerFactory = SklearnBatchLearnerFactory(learner, {'regParam': regParam, 'dim': dim, 'gamma': gamma,
                                                                      'n_components': n_components})
            else:
                print('Not using RFF')
                learnerFactory = SklearnBatchLearnerFactory(learner, {'regParam': regParam, 'dim': dim})

            exp = Experiment(executionMode='cpu', messengerHost=messengerHost, messengerPort=messengerPort,
                             numberOfNodes=numberOfNodes, sync=sync, aggregator=aggregator,
                             learnerFactory=learnerFactory, dataSourceFactory=dsFactory,
                             stoppingCriterion=stoppingCriterion, sleepTime=exp_sleep_time,
                             dataScheduler=BatchDataScheduler,
                             minStartNodes=numberOfNodes, minStopNodes=0, coordinatorSleepTime=coord_sleep_time)

            print('Running experiment:{} with max_example_value={}'.format(exp_name, max_example_value))
            try:
                exp.run(exp_name)
            except Exception as ex:
                print(type(ex).__name__, 'in experiment {}'.format(exp_count))
                continue

            with open(f_name, 'a') as fw:
                fw.write('\nEnding experiment {} at {}\ndim={}\naggregator={}\nmax_example_value={}\n'.format(exp_count,
                                                                                                            datetime.now(),
                                                                                                            dim,
                                                                                                            aggregator_name,
                                                                                                            max_example_value))

                fw.write('regParam={}\ngamma={}\nlearner={}\nnumberOfNodes={}\nn_components={}\n'.format(regParam,
                                                                                                         gamma,
                                                                                                         learner.__name__,
                                                                                                         numberOfNodes,
                                                                                                         n_components))

                fw.write('coord_sleep_time={}\nexp_sleep_time={}\n'.format(coord_sleep_time, exp_sleep_time))

                gc.collect()