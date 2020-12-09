import os
import shutil
import sys
from datetime import datetime
import re

sys.path.append("../../../..")
sys.path.append("../../../../../dlplatform")

from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.ParameterTuner import ParameterTuner
from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.CentralLearningExperimenter import \
    CentralLearningExperimenter
from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.DataGenerator import DataGenerator

RANDOM_STATE = 123
TIME_START = str(datetime.now())[:19]
RESULTS_FOLDER = os.path.join('./Results/', 'Exp_' + re.sub(r'[\s]', '__', re.sub(r'[\:-]', '_', TIME_START)))
# DATA_FOLDER = '/projects/nx11/resProj/dlapplication/data/HIGGS'
DATA_FOLDER = 'C:/Users/lasal/Documents/resProj/dlapplication/data/HIGGS'

# DATASET_NAME = 'SYN' + re.sub(r'[\s]', '.', re.sub(r'[\:-]', '', TIME_START))
DATASET_NAME = 'HIGGS'
DATASET_SIZE = 11_000_000
DIM = 28
# POLY_DEG = 3
DATA_LABEL_COL = 0
TUNE_DATA_FRACTION = 1000 / (DATASET_SIZE * 0.1)  # Tune using 2500 data points
TEST_DATA_FRACTION = 10000 / (DATASET_SIZE * 0.2)  # 0.003 # Test aggregated models using 3000 data points
CHECK_METRIC = False

if __name__ == '__main__':
    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)
    if not os.path.exists(os.path.join(RESULTS_FOLDER, 'scripts')):
        os.mkdir(os.path.join(RESULTS_FOLDER, 'scripts'))

    # copy scripts
    shutil.copy(__file__, os.path.join(RESULTS_FOLDER, 'scripts', os.path.basename(__file__)))
    C_list = [2 ** x for x in range(-12, 12)]  # [2 ** x for x in range(-12, 12)]
    # n_jobs = -1
    n_jobs = 5
    rff_gamma_list = [2 ** x for x in range(-12, 12)]  # [2 ** x for x in range(-12, 12)]
    n_components_list = [x for x in range(2, 1100, 100)]

    print('Running data generator in dir: {}'.format(DATA_FOLDER))

    data_generator = DataGenerator(size=DATASET_SIZE, dim=DIM, data_folder=DATA_FOLDER,
                                   data_name=DATASET_NAME, results_folder=RESULTS_FOLDER, method='existing')

    data_saved_path = data_generator()
    val_data_path = os.path.join(os.path.dirname(data_saved_path), 'split', 'VAL_' +
                                 os.path.basename(data_saved_path))
    train_data_path = os.path.join(os.path.dirname(data_saved_path), 'split', 'TRAIN_' +
                                   os.path.basename(data_saved_path))
    test_data_path = os.path.join(os.path.dirname(data_saved_path), 'split', 'TEST_' +
                                  os.path.basename(data_saved_path))

    print('Generated dataset {}'.format(data_saved_path))

    # max_samples_list = list([100_000, 150_000, 200_000, 300_000, 500_000])
    max_samples_list = list([50_000])

    learning_experimenter_kernel_svc = CentralLearningExperimenter(rff_sampler_gamma=None,
                                                                   reg_param=512,
                                                                   train_data_path=train_data_path,
                                                                   test_data_path=test_data_path,
                                                                   dim=DIM, data_label_col=DATA_LABEL_COL,
                                                                   dataset_name=DATASET_NAME,
                                                                   model_type='KernelSVC',
                                                                   test_fraction=TEST_DATA_FRACTION,
                                                                   results_folder_path=RESULTS_FOLDER,
                                                                   n_components_list=None,
                                                                   max_samples_list=max_samples_list,
                                                                   kernel_gamma=0.0078125)
    learning_experimenter_kernel_svc()
