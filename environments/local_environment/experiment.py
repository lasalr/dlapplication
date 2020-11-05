import random
from multiprocessing import Process
from DLplatform.coordinator import Coordinator, InitializationHandler
from DLplatform.worker import Worker
from DLplatform.communicating import Communicator, RabbitMQComm
from DLplatform.dataprovisioning import IntervalDataScheduler
from DLplatform.learningLogger import LearningLogger
from DLplatform.learning import LearnerFactory
import time
import os
import pickle
import numpy as np
import math
import subprocess

class Experiment():
    def __init__(self, executionMode, messengerHost, messengerPort, numberOfNodes, sync, aggregator, learnerFactory,
                 dataSourceFactory, stoppingCriterion, initHandler = InitializationHandler(),
                 dataScheduler = IntervalDataScheduler, minStartNodes=0, minStopNodes=0, sleepTime = 5, coordinatorSleepTime = 4):
        self.executionMode = executionMode
        if executionMode == 'cpu':
            self.devices = None
            self.modelsPer = None
        else:
            self.devices = []
            if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
                gpuIds = range(str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID'))
            else:
                gpuIds = os.environ.get('CUDA_VISIBLE_DEVICES').split(',')
            for taskid in gpuIds:
                self.devices.append('cuda:' + str(taskid))
            self.modelsPer = math.ceil(numberOfNodes * 1.0 / len(self.devices))
            print(self.modelsPer, "models per gpu on", ','.join(self.devices))

        self.messengerHost = messengerHost
        self.messengerPort = messengerPort
        self.numberOfNodes = numberOfNodes
        self.sync = sync
        self.aggregator = aggregator
        self.learnerFactory = learnerFactory
        self.dataSourceFactory = dataSourceFactory
        self.stoppingCriterion = stoppingCriterion
        self.initHandler = initHandler
        self.dataScheduler = dataScheduler
        self._uniqueId = str(os.getpid())
        self.sleepTime = sleepTime
        self.minStartNodes = minStartNodes
        self.minStopNodes = minStopNodes
        self.start_time = 0
        self.end_time = 0
        self.coordinatorSleepTime = coordinatorSleepTime

    def run(self, name):
        self.start_time = time.time()
        # exp_path = name + "_" + self.getTimestamp()
        exp_path = './Results/' + name + "_" + self.getTimestamp()
        os.mkdir(exp_path)
        self.writeExperimentSummary(exp_path, name)
        t = Process(target = self.createCoordinator, args=(exp_path, self.minStartNodes, self.minStopNodes),
                    name = 'coordinator')
        #t.daemon = True
        t.start()
        jobs = [t]
        time.sleep(self.sleepTime)
        # Create each worker
        for taskid in range(self.numberOfNodes):
            t = Process(target=self.createWorker,
                        args=(taskid, exp_path, self.executionMode, self.devices, self.modelsPer,),
                        name="worker_" + str(taskid))
            # t.daemon = True
            # print("Running t.start() for job:", t)
            t.start()
            jobs.append(t)
            # print('Sleeping for {}s to let worker {} run...'.format(self.sleepTime, taskid))
            # Sleep time reduces as more worker threads are started
            # time.sleep(5 + ((self.sleepTime / self.numberOfNodes) * (self.numberOfNodes - len(jobs))))
            time.sleep(self.sleepTime)
        for job in jobs:
            # print("Running job.join() for:", job)
            job.join()

        self.end_time = time.time()
        print('experiment done.')

    def createCoordinator(self, exp_path, minStartNodes, minStopNodes):
        print("create coordinator with minStart", minStartNodes, "and minStop", minStopNodes)
        coordinator = Coordinator(minStartNodes, minStopNodes, self.coordinatorSleepTime)
        coordinator.setInitHandler(self.initHandler)
        print('Creating RabbitMQComm within createCoordinator()...')
        comm = RabbitMQComm(hostname=self.messengerHost, port=self.messengerPort, user='guest', password='guest',
                            uniqueId=self._uniqueId)
        os.mkdir(os.path.join(exp_path, 'coordinator'))
        commLogger = LearningLogger(path=os.path.join(exp_path, 'coordinator'), id="communication", level='INFO')
        comm.setLearningLogger(commLogger)
        coordinator.setCommunicator(comm)
        self.sync.setAggregator(self.aggregator)
        coordinator.setSynchronizer(self.sync)
        logger = LearningLogger(path=exp_path, id="coordinator", level='INFO')
        coordinator.setLearningLogger(logger)
        print("Starting coordinator...\n")
        coordinator.run()

    def createWorker(self, id, exp_path, executionMode, devices, modelsPer):

        print("start creating worker" + str(id))
        if executionMode == 'cpu':
            device = None
        else:
            print("device for node", id, "is gpu", id // modelsPer)
            device = devices[id // modelsPer]
        nodeId = str(id)
        w = Worker(nodeId)
        # print('Sleeping before getting datasource within createWorker()')
        # time.sleep(random.uniform(1, 8))
        dataScheduler = self.dataScheduler()
        print('Getting datasource within createWorker()')
        dataSource = self.dataSourceFactory.getDataSource(nodeId=id)
        dataScheduler.setDataSource(source=dataSource)
        print('Finished getting and setting datasource within createWorker()')
        #       'Now sleeping Worker with PID={} for {}s'.format(self._uniqueId, self.sleepTime))
        # time.sleep(random.uniform(2, 6))
        w.setDataScheduler(dataScheduler)
        print('Creating RabbitMQComm within createWorker()...')
        comm = RabbitMQComm(hostname=self.messengerHost, port=self.messengerPort, user='guest', password='guest',
                            uniqueId=self._uniqueId)
        os.mkdir(os.path.join(exp_path, "worker" + str(id)))
        commLogger = LearningLogger(path=os.path.join(exp_path, "worker" + str(id)), id="communication", level='INFO')
        comm.setLearningLogger(commLogger)
        w.setCommunicator(comm)
        logger = LearningLogger(path=exp_path, id="worker" + str(id), level='INFO')
        learner = self.learnerFactory.getLearnerOnDevice(executionMode, device)
        learner.setLearningLogger(logger)
        learner.setStoppingCriterion(self.stoppingCriterion)
        self.sync.setAggregator(self.aggregator)
        learner.setSynchronizer(self.sync)
        w.setLearner(learner)
        print("created worker " + nodeId + "\n")
        w.run()

    def getTimestamp(self):
        return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

    def writeExperimentSummary(self, path, name):
        outString = "Experiment " + name + " Summary:\n\n"
        outString += "Start:\t" + str(self.start_time) + "\n"
        outString += "Number of Nodes:\t" + str(self.numberOfNodes) + "\n"
        outString += "Data source:\t\t" + str(self.dataSourceFactory) + "\n"
        outString += "Learner Factory:\t\t\t" + str(self.learnerFactory) + "\n"
        outString += "Learner:\t\t\t" + str(self.learnerFactory.getLearner()._name) + "\n"
        outString += "Learner Params:\t\t\t" + str(self.learnerFactory.sklearnParams) + "\n"
        outString += "Sync:\t\t\t" + str(self.sync) + "\n"
        outString += "Aggregator:\t\t" + str(self.aggregator) + "\n"
        outString += "Stopping criterion:\t" + str(self.stoppingCriterion) + "\n"
        outString += "Messenger Host:\t\t" + str(self.messengerHost) + "\n"
        outString += "Messenger Port:\t\t" + str(self.messengerPort) + "\n"
        outString += "Run Time:\t\t" +str(self.start_time - self.end_time) + "\n"
        outString += "Coordinator Sleep Time (s):\t\t" +str(self.coordinatorSleepTime) + "\n"

        summaryFile = os.path.join(path, "summary.txt")
        f = open(summaryFile, 'w')
        f.write(outString)
        f.close()
