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

LOG_CONSOLE = True

if __name__ == "__main__":

    messengerHost = 'localhost'
    messengerPort = 5672

    aggregator_list = ['RadonPoint', 'Average']
    max_example_list = [2000, 1000, 500, 200, 100, 50, 25]

    dim = 5  # SYNTHETIC1 has 5 features
    numberOfNodes = 205
    n_components = 202
    coord_sleep_time = 0.01  # 7
    learner = LinearSVCRandomFF
    regParam = 512
    gamma = 0.0078125
    exp_sleep_time = 0.01  # 2

    for aggregator_name in aggregator_list:
        for max_example_value in max_example_list:
            if aggregator_name == 'RadonPoint':
                aggregator = RadonPoint()
            elif aggregator_name == 'Average':
                aggregator = Average()
            else:
                print('Incorrect aggregator! Continuing to next experiment')
                continue

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

            exp_name = learner.__name__ + '_' + aggregator.__str__()
            print('Running experiment:{} with max_example_value={}'.format(exp_name, max_example_value))
            exp.run(exp_name)
            with open('./Results/{}_{}.txt'.format(exp_name, max_example_value), 'w') as fw:
                fw.write(
                    'dim={}\naggregator={}\nmax_example_value={}\nregParam={}\ngamma={}\nlearner={}\nnumberOfNodes={}\nn_components={}\ncoord_sleep_time={}\nexp_sleep_time={}\n'.format(
                        dim, aggregator_name, max_example_value, regParam, gamma, learner.__name__, numberOfNodes,
                        n_components, coord_sleep_time, exp_sleep_time))
