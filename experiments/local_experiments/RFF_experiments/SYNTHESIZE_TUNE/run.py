import os
import sys

sys.path.append("../../../..")
sys.path.append("../../../../../dlplatform")
from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.DataGenerator import DataGenerator
from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.ParameterTuner import ParameterTuner

from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.LearningExperimenter import LearningExperimenter
from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE import RadonAggregation

RANDOM_STATE = 123
DATA_FOLDER = './Data/'
RESULTS_FOLDER = './Results/'

DATASET_NAME = 'SYNTHETIC-AUTO1'
DATASET_SIZE = 50000
DIM = 5
POLY_DEG = 3
DATA_LABEL_COL = 0
TUNE_DATA_FRACTION = 1000 / (DATASET_SIZE * 0.1)  # Tune using 2500 data points
TEST_DATA_FRACTION = 3000 / (DATASET_SIZE * 0.2)  # 0.003 # Test aggregated models using 3000 data points

if __name__ == '__main__':
    # C_list = [2 ** x for x in range(-12, 14)]
    # n_jobs = 4
    # rff_gamma_list = [2 ** x for x in range(-12, 12)]
    # n_components_list = [x for x in range(2, 1100, 100)]
    C_list = [2 ** x for x in range(-12, 12)]
    n_jobs = -1
    rff_gamma_list = [2 ** x for x in range(-12, 12)]
    n_components_list = [x for x in range(2, 1100, 100)]

    print('Generating dataset in dir: {}'.format(DATA_FOLDER))
    # xy_noises = [[0.1, 0.01], [0.1, 0.001], [0.1, 0.0001], [0.1, 0.00001], [0.05, 0], [0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0]]
    # xy_noises = [[0.4, 0.0]]
    # idx = 0
    # for xy in xy_noises:
    # idx += 1
    # print('Starting tuning experiment {} of {}'.format(idx, len(xy_noises)))
    data_generator = DataGenerator(poly_deg=POLY_DEG, size=DATASET_SIZE, dim=DIM, data_folder=DATA_FOLDER,
                                   xy_noise_scale=[0.05, 0.05], x_range=[0.95, 1.5], bias_range=[-1, 1])
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
    gs_model_svc, gs_model_rff_svc = param_tuner()
    svc_best_params = gs_model_svc.best_params_
    svc_best_score = gs_model_svc.best_score_

    svc_rff_best_params = gs_model_rff_svc.best_params_
    svc_rff_best_score = gs_model_rff_svc.best_score_

    print('Tuned parameters\nsvc_best_params={} score={}\nsvc_rff_best_params={} score={}'.format(svc_best_params,
                                                                                                  svc_best_score,
                                                                                                  svc_rff_best_params,
                                                                                                  svc_rff_best_score))

    n_comps1 = list(reversed([i for i in range(2, 1100, 200)]))
    n_comps2 = list(reversed([i for i in range(2, 160, 40)]))
    n_nodes_list = [(x + 3) ** 2 for x in n_comps2] + [(x + 3) for x in n_comps1]
    # n_nodes_list = [(x + 3) for x in n_comps1]
    # n_components_list = n_comps2 + n_comps1
    n_components_list = n_comps1
    max_samples_list = list(reversed([25, 50, 100, 200, 500]))

    learning_experimenter = LearningExperimenter(rff_sampler_gamma=svc_rff_best_params['rff__gamma'],
                                                 reg_param=svc_rff_best_params['svc__C'],
                                                 train_data_path=train_data_path, test_data_path=test_data_path,
                                                 dim=DIM, data_label_col=DATA_LABEL_COL, dataset_name=DATASET_NAME,
                                                 model_type='LinearSVCRFF',
                                                 test_fraction=TEST_DATA_FRACTION, results_folder_path=RESULTS_FOLDER,
                                                 n_nodes_list=n_nodes_list, n_components_list=n_components_list,
                                                 max_node_samples_list=max_samples_list)

    learning_experimenter()
