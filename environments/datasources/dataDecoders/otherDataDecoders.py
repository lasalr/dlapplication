import numpy as np
from environments.datasources.dataDecoders import DataDecoder, CSVDecoder


class HIGGSDecoder(CSVDecoder):
    """
    Decoder for the HIGGS dataset. More info about dataset at https://archive.ics.uci.edu/ml/datasets/HIGGS
    """
    def __init__(self, delimiter=',', labelCol=0):
        CSVDecoder.__init__(self, delimiter, labelCol)

    def __call__(self, line):
        parsed_line = [float(c) for c in line.split(self._delimiter)]
        features = np.asarray(parsed_line[1:], dtype='float32')
        label = int(parsed_line[self._labelCol])
        return features, label

    def __str__(self):
        return "HIGGS from csv file"

class SUSYDecoder(CSVDecoder):
    """
    Decoder for the SUSY dataset. More info about dataset at https://archive.ics.uci.edu/ml/datasets/SUSY
    """
    def __init__(self, delimiter=',', labelCol=0):
        CSVDecoder.__init__(self, delimiter, labelCol)

    def __call__(self, line):
        parsed_line = [float(c) for c in line.split(self._delimiter)]
        features = np.asarray(parsed_line[1:], dtype='float32')
        label = int(parsed_line[self._labelCol])
        return features, label

    def __str__(self):
        return "HIGGS from csv file"