from environments.datasources.standardDataSources import FileDataSource, SVMLightDataSource
import os

from DLplatform.dataprovisioning.dataSourceFactory import DataSourceFactory
from DLplatform.stopping import MaxAmountExamples


class FileDataSourceFactory(DataSourceFactory):
    def __init__(self, filename, decoder, numberOfNodes, indices = 'roundRobin', shuffle = False, cache = False,
                 stoppingCriterion = None):
        self.filename       = filename
        self.decoder        = decoder
        self.numberOfNodes  = numberOfNodes
        self.indices        = indices
        self.shuffle        = shuffle
        self.cache          = cache
        self.stoppingCriterion = stoppingCriterion
        self.maxAmount = None

    def getDataSource(self, nodeId):
        # print('Getting datasource for node {}'.format(nodeId))
        name = os.path.basename(self.filename).split('.')[0]
        # print('name={}'.format(name))

        # print('type(self.stoppingCriterion)={}'.format(type(self.stoppingCriterion)))
        # print('MaxAmountExamples={}'.format(MaxAmountExamples))
        if isinstance(self.stoppingCriterion, MaxAmountExamples):
            # print('isinstance-TRUE')
            self.maxAmount = self.stoppingCriterion.maxAmount

        dataSource = FileDataSource(filename = self.filename, decodeLine = self.decoder, name = name,
                                    indices = self.indices, nodeId=nodeId, numberOfNodes=self.numberOfNodes, 
                                    shuffle = self.shuffle, cache = self.cache, maxAmount=self.maxAmount)
        # print('Finished getting datasource for node {}'.format(nodeId))
        return dataSource

    def __str__(self):
        return "File DataSource, filename " + self.filename + ", decoder " + str(self.decoder) + ", data distribution " + self.indices + ", shuffle " + str(self.shuffle) + ", cache " + str(self.cache)

class SVMLightDataSourceFactory(DataSourceFactory):
    def __init__(self, filename, numberOfNodes, indices = 'roundRobin', shuffle = False):
        self.filename       = filename
        self.numberOfNodes  = numberOfNodes
        self.indices        = indices
        self.shuffle        = shuffle
                
    def getDataSource(self, nodeId):
        name = os.path.basename(self.filename).split('.')[0]
        dataSource = SVMLightDataSource(filename = self.filename, indices = self.indices, nodeId = nodeId, numberOfNodes = self.numberOfNodes, name = name, shuffle = self.shuffle)
        return dataSource 

    def __str__(self):
        return "SVMLight DataSource, filename " + self.filename + ", data distribution " + self.indices + ", shuffle " + str(self.shuffle)

