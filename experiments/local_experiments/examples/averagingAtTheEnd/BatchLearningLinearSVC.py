import sys
sys.path.append("../../../../../dlapplication")
sys.path.append("../../../../../dlplatform")


from environments.local_environment import Experiment
from environments.datasources.standardDataSourceFactories import FileDataSourceFactory
from environments.datasources.dataDecoders.otherDataDecoders import HIGGSDecoder
from DLplatform.aggregating import Average
from DLplatform.aggregating import RadonPoint
from DLplatform.synchronizing.aggAtTheEnd import AggregationAtTheEnd
from DLplatform.learning.factories.sklearnBatchLearnerFactory import SklearnBatchLearnerFactory
from DLplatform.learning.batch.sklearnClassifiers import LogisticRegression
from DLplatform.learning.batch.sklearnClassifiers import LinearSVC
from DLplatform.stopping import MaxAmountExamples


if __name__ == "__main__":
  
    messengerHost = 'localhost'
    messengerPort = 5672
    numberOfNodes = 4
    
    regParam = 0.01
    dim = 4 #skin_segmentation has 4 attributes
    learnerFactory = SklearnBatchLearnerFactory(LinearSVC, {'regParam' : regParam, 'dim' : dim})
    
    # dsFactory = SVMLightDataSourceFactory("../../../../data/classification/skin_segmentation.dat", numberOfNodes,
    # indices = 'roundRobin', shuffle = False)

    dsFactory = FileDataSourceFactory(
        filename="../../../../data/HIGGS/HIGGS.csv",
        decoder=HIGGSDecoder(), numberOfNodes=numberOfNodes, indices='roundRobin', shuffle=False, cache=False)

    stoppingCriterion = MaxAmountExamples(6000)
        
    aggregator = Average()
    sync = AggregationAtTheEnd()
    
    exp = Experiment(executionMode = 'cpu', messengerHost = messengerHost, messengerPort = messengerPort, 
        numberOfNodes = numberOfNodes, sync = sync, 
        aggregator = aggregator, learnerFactory = learnerFactory, 
        dataSourceFactory = dsFactory, stoppingCriterion = stoppingCriterion, sleepTime=0)
    exp.run("Linear_SVC" + "_" + aggregator.__str__())

