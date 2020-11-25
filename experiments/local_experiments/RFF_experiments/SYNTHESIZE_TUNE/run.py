import os
import sys
from datetime import datetime

sys.path.append("../../../..")
sys.path.append("../../../../../dlplatform")
from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.DataGenerator import DataGenerator
from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.ParameterTuner import ParameterTuner
from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.LearningExperimenter import LearningExperimenter

RANDOM_STATE = 123
RESULTS_FOLDER = './Results/'
DATA_FILE_PATH = 'SYNTHETIC_DATA_' + str(datetime.now()).replace(':', '_').replace(' ', '_')[:19] + '.csv'
VAL_FILE_PATH = os.path.join(os.path.dirname(DATA_FILE_PATH), 'split', 'VAL_' + os.path.basename(DATA_FILE_PATH))
TRAIN_DATA_PATH = os.path.join(os.path.dirname(DATA_FILE_PATH), 'split', 'TRAIN_' + os.path.basename(DATA_FILE_PATH))
TEST_DATA_PATH = os.path.join(os.path.dirname(DATA_FILE_PATH), 'split', 'TEST_' + os.path.basename(DATA_FILE_PATH))
DATASET_NAME = 'SYNTHETIC-AUTO1'
DATASET_SIZE = 50_000
DIM = 5
POLY_DEG = 3
DATA_LABEL_COL = 0
TUNE_DATA_FRACTION = 0.01
TEST_DATA_FRACTION = 0.1

if __name__ == '__main__':
    C_list = [2 ** x for x in range(-12, 14)]
    n_jobs = 4
    rff_gamma_list = [2 ** x for x in range(-12, 12)]
    n_components_list = [x for x in range(2, 1100, 100)]

    data_generator = DataGenerator(poly_deg=POLY_DEG, size=DATASET_SIZE, dim=DIM)
    data_saved_path = data_generator()
    print('Generated dataset {}'.format(data_saved_path))
    param_tuner = ParameterTuner(C_list=C_list, rff_gamma_list=rff_gamma_list, n_components_list=n_components_list,
                                 n_jobs=4, dataset_name=DATASET_NAME, data_file_path=DATA_FILE_PATH,
                                 results_folder_path=RESULTS_FOLDER, tune_data_fraction=TUNE_DATA_FRACTION, dim=DIM,
                                 data_label_col=DATA_LABEL_COL, score_method='roc_auc')

    svc_best_params, svc_rff_best_params = param_tuner()
    print('Tuned parameters\nsvc_best_params={}\nsvc_rff_best_params={}'.format(svc_best_params, svc_rff_best_params))
    learning_experimenter = LearningExperimenter(rff_sampler_gamma=svc_rff_best_params['rff__gamma'],
                                                 reg_param=svc_rff_best_params['svc__C'],
                                                 train_data_path=TRAIN_DATA_PATH, test_data_path=TEST_DATA_PATH,
                                                 dim=DIM, data_label_col=DATA_LABEL_COL, dataset_name=DATASET_NAME,
                                                 model_type='LinearSVCRFF',
                                                 test_fraction=TEST_DATA_FRACTION, results_folder_path=RESULTS_FOLDER)
