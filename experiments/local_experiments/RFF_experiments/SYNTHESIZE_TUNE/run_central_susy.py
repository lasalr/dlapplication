import os
import shutil
import sys
from datetime import datetime
import re

from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.ParameterTuner import ParameterTuner

sys.path.append("../../../..")
sys.path.append("../../../../../dlplatform")

from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.CentralLearningExperimenter import \
    CentralLearningExperimenter
from experiments.local_experiments.RFF_experiments.SYNTHESIZE_TUNE.DataGenerator import DataGenerator

RANDOM_STATE = 123
TIME_START = str(datetime.now())[:19]
RESULTS_FOLDER = os.path.join('./Results/', 'Exp_' + re.sub(r'[\s]', '__', re.sub(r'[\:-]', '_', TIME_START)))
# DATA_FOLDER = '/projects/nx11/resProj/dlapplication/data/HIGGS'
DATA_FOLDER = 'C:/Users/lasal/Documents/resProj/dlapplication/data/SUSY'

# DATASET_NAME = 'SYN' + re.sub(r'[\s]', '.', re.sub(r'[\:-]', '', TIME_START))
DATASET_NAME = 'SUSY'
DATASET_SIZE = 5_000_000
DIM = 18
# POLY_DEG = 3
DATA_LABEL_COL = 0
TUNE_DATA_FRACTION = 2500 / (DATASET_SIZE * 0.1)  # Tune using 2500 data points
TEST_DATA_FRACTION = 2000 / (DATASET_SIZE * 0.2)  # 0.003 # Test aggregated models using 3000 data points
CHECK_METRIC = False

if __name__ == '__main__':
    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)
    if not os.path.exists(os.path.join(RESULTS_FOLDER, 'scripts')):
        os.mkdir(os.path.join(RESULTS_FOLDER, 'scripts'))

    # copy scripts
    shutil.copy(__file__, os.path.join(RESULTS_FOLDER, 'scripts', os.path.basename(__file__)))
    C_list = [2 ** x for x in range(-14, 14)]  # [2 ** x for x in range(-12, 12)]
    n_jobs = -1
    rff_gamma_list = [2 ** x for x in range(-14, 14)]  # [2 ** x for x in range(-12, 12)]
    n_components_list = [x for x in range(2, 1100, 100)]

    print('Running data generator in dir: {}'.format(DATA_FOLDER))
    # xy_noises = [[0.1, 0.01], [0.1, 0.001], [0.1, 0.0001], [0.1, 0.00001], [0.05, 0], [0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0]]
    # xy_noises = [[0.4, 0.0]]
    # idx = 0
    # for xy in xy_noises:
    # idx += 1
    # print('Starting tuning experiment {} of {}'.format(idx, len(xy_noises)))

    # data_generator = DataGenerator(poly_deg=POLY_DEG, size=DATASET_SIZE, dim=DIM, data_folder=DATA_FOLDER,
    #                                data_name=DATASET_NAME, xy_noise_scale=[0.2, 0.15], x_range=[0.95, 1.5],
    #                                bias_range=[-10, 150], method='custom')

    # data_generator = DataGenerator(poly_deg=POLY_DEG, size=DATASET_SIZE, dim=DIM, data_folder=DATA_FOLDER,
    #                                xy_noise_scale=[None, 0.05], method='sklearn')

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
    param_tuner = ParameterTuner(C_list=C_list, rff_gamma_list=rff_gamma_list, n_components_list=n_components_list,
                                 n_jobs=n_jobs, dataset_name=DATASET_NAME, val_file_path=val_data_path,
                                 results_folder_path=RESULTS_FOLDER, tune_data_fraction=TUNE_DATA_FRACTION, dim=DIM,
                                 data_label_col=DATA_LABEL_COL, score_method='roc_auc')
    gs_model_svc, gs_model_rff_svc = param_tuner()
    param_results = 'Tuned parameters\nsvc_best_params={} score={}\n'.format(gs_model_svc.best_params_,
                                                                             gs_model_svc.best_score_)
    param_results += 'svc_rff_best_params={} score={}'.format(gs_model_rff_svc.best_params_,
                                                              gs_model_rff_svc.best_score_)
    print(param_results)
    with open(os.path.join(RESULTS_FOLDER, 'param_results.txt'), 'w') as fw:
        fw.write(param_results)

    if (gs_model_rff_svc.best_score_ - gs_model_svc.best_score_ < 0.05):
        if not CHECK_METRIC:
            print('svc_rff_best_score={} does not exceed svc_best_score={} by at least 0.05 but continuing with experiment')
        else:
            print('svc_rff_best_score={} does not exceed svc_best_score={} by at least 0.05! Stopping experiment')
            print('Deleting data file {}'.format(data_saved_path))
            os.remove(path=data_saved_path)
            print('Deleting val data file {}'.format(val_data_path))
            os.remove(path=val_data_path)
            print('Deleting train data file {}'.format(train_data_path))
            os.remove(path=train_data_path)
            print('Deleting test data file {}'.format(test_data_path))
            os.remove(path=test_data_path)
            sys.exit()

    n_comps1 = list(reversed([i for i in range(2, 1100, 200)]))
    n_comps2 = list(reversed([i for i in range(2, 160, 40)]))
    # n_nodes_list = [(x + 3) ** 2 for x in n_comps2] + [(x + 3) for x in n_comps1]
    n_components_list = n_comps2 + n_comps1
    max_samples_list = list(reversed([25, 50, 100, 200, 500, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 6000]))

    # n_comps1 = list([i for i in range(2, 1100, 200)])
    # n_nodes_list = [(x + 3) for x in n_comps1]
    # n_components_list = n_comps1
    # max_samples_list = list([25, 50, 100, 200, 500])

    learning_experimenter_svc = CentralLearningExperimenter(rff_sampler_gamma=None,
                                                            reg_param=gs_model_svc.best_params_['C'],
                                                            train_data_path=train_data_path,
                                                            test_data_path=test_data_path,
                                                            dim=DIM, data_label_col=DATA_LABEL_COL,
                                                            dataset_name=DATASET_NAME,
                                                            model_type='LinearSVC',
                                                            test_fraction=TEST_DATA_FRACTION,
                                                            results_folder_path=RESULTS_FOLDER,
                                                            n_components_list=None,
                                                            max_samples_list=max_samples_list)

    learning_experimenter_svc()

    learning_experimenter_svc_rff = CentralLearningExperimenter(rff_sampler_gamma=gs_model_rff_svc.best_params_['rff__gamma'],
                                                                reg_param=gs_model_rff_svc.best_params_['svc__C'],
                                                                train_data_path=train_data_path,
                                                                test_data_path=test_data_path,
                                                                dim=DIM, data_label_col=DATA_LABEL_COL,
                                                                dataset_name=DATASET_NAME,
                                                                model_type='LinearSVCRFF',
                                                                test_fraction=TEST_DATA_FRACTION,
                                                                results_folder_path=RESULTS_FOLDER,
                                                                n_components_list=n_components_list,
                                                                max_samples_list=max_samples_list)

    learning_experimenter_svc_rff()
