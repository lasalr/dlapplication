import os
import sys

sys.path.append("../../../..")
sys.path.append("../../../../../dlplatform")
from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.DataGenerator import DataGenerator
from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.ParameterTuner import ParameterTuner

from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.LearningExperimenter import LearningExperimenter

RANDOM_STATE = 123
DATA_FOLDER = './Data/'
RESULTS_FOLDER = './Results/'

DATASET_NAME = 'SYNTHETIC-AUTO1'
DATASET_SIZE = 5_000_000
DIM = 5
POLY_DEG = 3
DATA_LABEL_COL = 0
TUNE_DATA_FRACTION = 0.005
TEST_DATA_FRACTION = 0.05

if __name__ == '__main__':
    # C_list = [2 ** x for x in range(-12, 14)]
    # n_jobs = 4
    # rff_gamma_list = [2 ** x for x in range(-12, 12)]
    # n_components_list = [x for x in range(2, 1100, 100)]
    C_list = [2 ** x for x in range(-3, 10)]
    n_jobs = -1
    rff_gamma_list = [2 ** x for x in range(-3, 10)]
    n_components_list = [x for x in range(2, 1100, 100)]

    print('Generating dataset in dir: {}'.format(DATA_FOLDER))
    data_generator = DataGenerator(poly_deg=POLY_DEG, size=DATASET_SIZE, dim=DIM, data_folder=DATA_FOLDER,
                                   xy_noise_scale=[0.15, 0.15], x_range=[-10, 10], bias_range=[-10, 10])
    data_saved_path = data_generator()

    val_data_path = os.path.join(os.path.dirname(data_saved_path), 'split', 'VAL_' +
                                 os.path.basename(data_saved_path))
    train_data_path = os.path.join(os.path.dirname(data_saved_path), 'split', 'TRAIN_' +
                                   os.path.basename(data_saved_path))
    test_data_path = os.path.join(os.path.dirname(data_saved_path), 'split', 'TEST_' +
                                  os.path.basename(data_saved_path))

    print('Generated dataset {}'.format(data_saved_path))
    param_tuner = ParameterTuner(C_list=C_list, rff_gamma_list=rff_gamma_list, n_components_list=n_components_list,
                                 n_jobs=n_jobs, dataset_name=DATASET_NAME, val_file_path=val_data_path,
                                 results_folder_path=RESULTS_FOLDER, tune_data_fraction=TUNE_DATA_FRACTION, dim=DIM,
                                 data_label_col=DATA_LABEL_COL, score_method='roc_auc')

    svc_best_params, svc_rff_best_params = param_tuner()
    print('Tuned parameters\nsvc_best_params={}\nsvc_rff_best_params={}'.format(svc_best_params, svc_rff_best_params))

    n_comps1 = list(reversed([i for i in range(2, 1100, 500)]))
    # n_comps2 = list(reversed([i for i in range(2, 200, 80)]))
    # n_nodes_list = [(x + 3) ** 2 for x in n_comps2] + [(x + 3) for x in n_comps1]
    n_nodes_list =[(x + 3) for x in n_comps1]
    # n_components_list = n_comps2 + n_comps1
    n_components_list = n_comps1

    learning_experimenter = LearningExperimenter(rff_sampler_gamma=svc_rff_best_params['rff__gamma'],
                                                 reg_param=svc_rff_best_params['svc__C'],
                                                 train_data_path=train_data_path, test_data_path=test_data_path,
                                                 dim=DIM, data_label_col=DATA_LABEL_COL, dataset_name=DATASET_NAME,
                                                 model_type='LinearSVCRFF',
                                                 test_fraction=TEST_DATA_FRACTION, results_folder_path=RESULTS_FOLDER,
                                                 n_nodes_list=n_nodes_list, n_components_list=n_components_list)

    learning_experimenter()
