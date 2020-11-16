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

    dim = 5  # SYNTHETIC1 has 5 features
    numberOfNodes = 196  # 205
    n_components = 11  # 202
    coord_sleep_time = 0.1 # 7
    learner = LinearSVCRandomFF
    regParam = 512
    gamma = 0.0078125
    max_example_value = 25 # 105000
    exp_sleep_time = 0.1  # 2

    aggregator = RadonPoint()
    sync = AggregationAtTheEnd()
    stoppingCriterion = MaxAmountExamples(max_example_value)

    dsFactory = FileDataSourceFactory(filename="../../../../data/SYNTHETIC1/SYNTHETIC_DATA.csv",
                                      decoder=CSVDecoder(delimiter=',', labelCol=0),
                                      numberOfNodes=numberOfNodes, indices='roundRobin', shuffle=False, cache=False,
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
                     stoppingCriterion=stoppingCriterion, sleepTime=exp_sleep_time, dataScheduler=BatchDataScheduler,
                     minStartNodes=numberOfNodes, minStopNodes=0, coordinatorSleepTime=coord_sleep_time)

    exp_name = learner.__name__ + '_' + aggregator.__str__()
    print('Running experiment:{}'.format(exp_name))
    exp.run(exp_name)
