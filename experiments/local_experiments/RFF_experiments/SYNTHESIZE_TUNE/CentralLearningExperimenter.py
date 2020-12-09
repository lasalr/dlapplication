import os
import shutil
import sys
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

sys.path.append("../../../..")
sys.path.append("../../../../../dlplatform")

from experiments.local_experiments.RFF_experiments.data_handling import load_data


class CentralLearningExperimenter:
    RANDOM_STATE = 123

    def __init__(self, rff_sampler_gamma, reg_param, train_data_path, test_data_path, dim, data_label_col, dataset_name,
                 results_folder_path, n_components_list, max_samples_list, model_type, kernel_gamma=None,
                 test_fraction=0.1):

        self.n_components_list = n_components_list
        self.max_samples_list = max_samples_list
        self.test_fraction = test_fraction
        self.rff_sampler_gamma = rff_sampler_gamma
        self.kernel_gamma = kernel_gamma
        self.reg_param = reg_param
        self.model_type = model_type
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data_label_col = data_label_col
        self.dim = dim
        self.dataset_name = dataset_name
        self.results_folder_path = results_folder_path

    def __call__(self):

        # Copy script to results folder
        shutil.copy(__file__, os.path.join(self.results_folder_path, 'scripts', os.path.basename(__file__)))

        # Load full train data to dataframe
        full_train_data_df = self.data_to_memory()

        # Load test data
        X, y = load_data(path=self.test_data_path, label_col=self.data_label_col, d=self.dim)
        _, X_test, _, y_test = train_test_split(X, y, test_size=self.test_fraction,
                                                random_state=CentralLearningExperimenter.RANDOM_STATE)
        if not os.path.exists(self.results_folder_path):
            print('Creating folder {}'.format(str(self.results_folder_path)))
            os.mkdir(path=self.results_folder_path)

        all_results_dict = {}
        exp_indx = 0

        total_exp_count = 0
        if self.model_type == 'LinearSVCRFF':
            for c in self.n_components_list:
                for m in self.max_samples_list:
                    total_exp_count += 1

        elif self.model_type == 'LinearSVC':
            for m in self.max_samples_list:
                total_exp_count += 1

        elif self.model_type == 'KernelSVC':
            for m in self.max_samples_list:
                total_exp_count += 1

        if self.model_type == 'KernelSVC':
            for max_smp in self.max_samples_list:
                # Save to dictionary of models
                svc_model = self.train_single_model(n_components=None, full_df=full_train_data_df,
                                                    max_samples=max_smp)

                print('Running experiment {} of {}'.format(exp_indx, total_exp_count))
                details = {}
                details['MODEL_TYPE'] = self.model_type
                details['DATASET_NAME'] = self.dataset_name
                details['N_NODES'] = 1
                details['N_COMPONENTS'] = '-'
                details['MAX_NODE_SAMPLES'] = max_smp

                all_results_dict[exp_indx] = self.get_calc_metrics(X_test=X_test, y_test=y_test,
                                                                   experiment_info_dict=details,
                                                                   model=svc_model)
                all_results_dict[exp_indx]['aggregator'] = 'None'
                exp_indx += 1
                # Log interim df
                pd.DataFrame(all_results_dict).transpose().to_csv(
                    os.path.join(self.results_folder_path, 'interim_results.csv'))
                print('Logged interim results after completion of experiment {}'.format(exp_indx))

        elif self.model_type == 'LinearSVCRFF':
            for n_c in self.n_components_list:
                # Create RFF sampler
                sampler = RBFSampler(gamma=self.rff_sampler_gamma, n_components=n_c,
                                     random_state=CentralLearningExperimenter.RANDOM_STATE)
                # Generating RFF features from test data
                X_test_sampled = sampler.fit_transform(X_test)

                for max_smp in self.max_samples_list:
                    # Save to dictionary of models
                    svc_model = self.train_single_model(n_components=n_c, full_df=full_train_data_df,
                                                        max_samples=max_smp)

                    print('Running experiment {} of {}'.format(exp_indx, total_exp_count))
                    details = {}
                    details['MODEL_TYPE'] = self.model_type
                    details['DATASET_NAME'] = self.dataset_name
                    details['N_NODES'] = 1
                    details['N_COMPONENTS'] = n_c
                    details['MAX_NODE_SAMPLES'] = max_smp
                    all_results_dict[exp_indx] = self.get_calc_metrics(X_test=X_test_sampled, y_test=y_test,
                                                                       experiment_info_dict=details,
                                                                       model=svc_model)
                    all_results_dict[exp_indx]['aggregator'] = 'None'
                    exp_indx += 1
                    # Log interim df
                    pd.DataFrame(all_results_dict).transpose().to_csv(
                        os.path.join(self.results_folder_path, 'interim_results.csv'))
                    print('Logged interim results after completion of experiment {}'.format(exp_indx))

        elif self.model_type == 'LinearSVC':

            for max_smp in self.max_samples_list:
                # Save to dictionary of models
                svc_model = self.train_single_model(n_components=None, full_df=full_train_data_df,
                                                    max_samples=max_smp)

                print('Running experiment {} of {}'.format(exp_indx, total_exp_count))
                details = {}
                details['MODEL_TYPE'] = self.model_type
                details['DATASET_NAME'] = self.dataset_name
                details['N_NODES'] = 1
                details['N_COMPONENTS'] = '-'
                details['MAX_NODE_SAMPLES'] = max_smp

                all_results_dict[exp_indx] = self.get_calc_metrics(X_test=X_test, y_test=y_test,
                                                                   experiment_info_dict=details,
                                                                   model=svc_model)
                all_results_dict[exp_indx]['aggregator'] = 'None'
                exp_indx += 1
                # Log interim df
                pd.DataFrame(all_results_dict).transpose().to_csv(
                    os.path.join(self.results_folder_path, 'interim_results.csv'))
                print('Logged interim results after completion of experiment {}'.format(exp_indx))

        # Log final df
        pd.DataFrame(all_results_dict).transpose().to_csv(os.path.join(self.results_folder_path, 'final_results.csv'))
        print('Logged final results')

    def get_calc_metrics(self, X_test, y_test, experiment_info_dict, model):
        # Calculating metrics and storing in dict
        det_dict = experiment_info_dict.copy()
        det_dict['accuracy'] = model.score(X=X_test, y=y_test)
        y_test_df_score = model.decision_function(X_test)
        det_dict['rocauc'] = roc_auc_score(y_true=y_test, y_score=y_test_df_score)

        return det_dict

    def data_to_memory(self):
        """
        Loads entire dataset to memory
        """
        header_names = ['f' + str(x) if x != self.data_label_col else 'label' for x in range(self.dim + 1)]
        full_df = pd.read_csv(filepath_or_buffer=self.train_data_path, names=header_names,
                              dtype={h: float for h in header_names})
        return full_df

    def load_df_memory(self, max_samples, df_large):
        """
        Loads required amount of data to dataframe
        Assumes original dataset is shuffled
        """
        # Get every n_nodes row starting from
        return df_large.head(max_samples)

    def train_and_get_model(self, X, y, model=None, sampler=None):
        if isinstance(model, LinearSVC) and isinstance(sampler, RBFSampler):
            sampler = sampler
            X_train = sampler.fit_transform(X)
        elif isinstance(model, LinearSVC) and (sampler is None):
            X_train = X
        elif isinstance(model, SVC) and (sampler is None):
            X_train = X
        else:
            raise ValueError('Incorrect model type')

        model.fit(X_train, y)

        return model

    def score_model(self, X, y, n_components, model=None, sampler=None, ):
        if isinstance(model, LinearSVC) and (sampler is not None):
            sampler = RBFSampler(gamma=self.rff_sampler_gamma, n_components=n_components,
                                 random_state=CentralLearningExperimenter.RANDOM_STATE)
            X_test = sampler.fit_transform(X)
        elif isinstance(model, LinearSVC) and (sampler is None):
            X_test = X
        else:
            raise ValueError('Incorrect model type')

        acc = model.score(X_test, y)
        y_df_score = model.decision_function(X_test)
        rocauc = roc_auc_score(y_true=y, y_score=y_df_score)

        return acc, rocauc

    def train_single_model(self, n_components, full_df, max_samples):
        """
        Trains a single model
        """
        # Loading relevant data
        train_df = self.load_df_memory(max_samples=max_samples, df_large=full_df)
        print('Finished reading {} rows of data'.format(train_df.shape[0]))

        # Splitting into features and label
        train_features = train_df.drop('label', axis=1)
        train_label = train_df['label']

        # Creating sampler if required
        if self.model_type == 'LinearSVCRFF':
            rff_sampler = RBFSampler(gamma=self.rff_sampler_gamma, n_components=n_components,
                                     random_state=CentralLearningExperimenter.RANDOM_STATE)
            # Creating LinearSVC model
            svc_model = LinearSVC(C=self.reg_param, loss='squared_hinge', dual=False, max_iter=2000,
                                  random_state=CentralLearningExperimenter.RANDOM_STATE)
            # Training model
            trained_model = self.train_and_get_model(X=train_features, y=train_label, model=svc_model,
                                                     sampler=rff_sampler)
        elif self.model_type == 'LinearSVC':
            rff_sampler = None
            # Creating LinearSVC model
            svc_model = LinearSVC(C=self.reg_param, loss='squared_hinge', dual=False, max_iter=2000,
                                  random_state=CentralLearningExperimenter.RANDOM_STATE)
            # Training model
            trained_model = self.train_and_get_model(X=train_features, y=train_label, model=svc_model,
                                                     sampler=rff_sampler)
        elif self.model_type == 'KernelSVC':
            svc_model = SVC(C=self.reg_param, gamma=self.kernel_gamma, cache_size=8192,
                            random_state=CentralLearningExperimenter.RANDOM_STATE)
            trained_model = self.train_and_get_model(X=train_features, y=train_label, model=svc_model, sampler=None)
        else:
            raise ValueError('Incorrect model type given.')

        print('Trained model')
        return trained_model
