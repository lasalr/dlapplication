import os
import re
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

sys.path.append("../../../..")
sys.path.append("../../../../../dlplatform")

from experiments.local_experiments.RFF_experiments.data_handling import load_data
from DLplatform.parameters.vectorParameters import VectorParameter
from DLplatform.aggregating import RadonPoint, Average


class LearningExperimenter:

    RANDOM_STATE = 123

    def __init__(self, rff_sampler_gamma, reg_param, train_data_path, test_data_path, dim, data_label_col, dataset_name,
                 results_folder_path, model_type='LinearSVCRFF', test_fraction=0.1):

        n_comps1 = list(reversed([i for i in range(2, 1100, 50)]))
        n_comps2 = list(reversed([i for i in range(2, 200, 50)]))
        self.n_nodes_list = [(x + 3) ** 2 for x in n_comps2] + [(x + 3) for x in n_comps1]
        self.n_components_list = n_comps2 + n_comps1
        self.max_node_samples_list = list(reversed([25, 50, 100, 200, 500]))
        self.agg_types_list = ['Radon point', 'Averaging']
        self.test_fraction = test_fraction
        self.rff_sampler_gamma = rff_sampler_gamma
        self.reg_param = reg_param
        self.model_type = model_type
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data_label_col = data_label_col
        self.dim = dim
        self.dataset_name = dataset_name
        self.results_folder_path = results_folder_path

    def __call__(self):

        # Load full train data to dataframe
        full_train_data_df = self.data_to_memory(path=self.train_data_path, dim=self.dim, data_label_col=self.data_label_col)

        # Load test data
        X, y = load_data(path=self.test_data_path, label_col=self.data_label_col, d=self.dim)
        _, X_test, _, y_test = train_test_split(X, y, test_size=self.test_fraction, random_state=LearningExperimenter.RANDOM_STATE)

        # Create outer folder for all experiments
        folder_ts = 'Exp_' + re.sub(r'[\s]', '__', re.sub(r'[\:-]', '_', str(datetime.now())[:19]))
        sub_root_folder = os.path.join(self.results_folder_path, folder_ts)

        if not os.path.exists(sub_root_folder):
            print('Creating folder {}'.format(str(sub_root_folder)))
            os.mkdir(path=sub_root_folder)

        all_results_dict = {}
        exp_indx = 0

        total_exp_count = 0
        for c, n in zip(self.n_components_list, self.n_nodes_list):
            for m in self.max_node_samples_list:
                for a in self.agg_types_list:
                    total_exp_count += 1

        for n_c, n_nd in zip(self.n_components_list, self.n_nodes_list):
            print('Running experiment {} of {}'.format(exp_indx, total_exp_count))
            # Create RFF sampler
            sampler = RBFSampler(gamma=self.rff_sampler_gamma, n_components=n_c, random_state=LearningExperimenter.RANDOM_STATE)
            # Generating RFF features from test data
            X_test_sampled = sampler.fit_transform(X_test)

            for max_smp in self.max_node_samples_list:
                # Save to dictionary of models
                trained_models = self.train_for_all_nodes(n_nodes=n_nd, n_components=n_c, full_df=full_train_data_df,
                                                          max_samples=max_smp, print_every=300,
                                                          gamma=self.rff_sampler_gamma, reg_param=self.reg_param,
                                                          model_type=self.model_type)

                for agg_type in self.agg_types_list:
                    details = {}
                    details['MODEL_TYPE'] = self.model_type
                    details['DATASET_NAME'] = DATASET_NAME
                    details['N_NODES'] = n_nd
                    details['N_COMPONENTS'] = n_c
                    details['MAX_NODE_SAMPLES'] = max_smp

                    agg_model = self.aggregate_models_memory(model_dict=trained_models, agg_type=agg_type)
                    svc_model = self.set_model(agg_model)

                    if self.model_type == 'LinearSVCRFF':
                        all_results_dict[exp_indx] = self.get_calc_metrics(X_test=X_test_sampled, y_test=y_test,
                                                                           experiment_info_dict=details,
                                                                           aggregation_type=agg_type, model=svc_model)
                    elif self.model_type == 'LinearSVC':
                        all_results_dict[exp_indx] = self.get_calc_metrics(X_test=X_test, y_test=y_test,
                                                                           experiment_info_dict=details,
                                                                           aggregation_type=agg_type, model=svc_model)
                    exp_indx += 1

                # Log interim df
                pd.DataFrame(all_results_dict).transpose().to_csv(os.path.join(sub_root_folder, 'interim_results.csv'))

        # Log final df
        pd.DataFrame(all_results_dict).transpose().to_csv(os.path.join(sub_root_folder, 'final_results.csv'))

    def get_calc_metrics(self, X_test, y_test, experiment_info_dict, aggregation_type, model):
        # Calculating metrics and storing in dict
        experiment_info_dict['accuracy'] = model.score(X=X_test, y=y_test)
        y_test_df_score = model.decision_function(X_test)
        experiment_info_dict['rocauc'] = roc_auc_score(y_true=y_test, y_score=y_test_df_score)
        experiment_info_dict['aggregator'] = aggregation_type
        return experiment_info_dict

    def aggregate_models_memory(self, model_dict, agg_type):
        # Creating aggregator
        if agg_type == 'Radon point':
            aggtr = RadonPoint()
        elif agg_type == 'Averaging':
            aggtr = Average()
        else:
            print('Incorrect aggregator type!')
            raise TypeError

        return aggtr(self.create_vector_param_list(model_dict.values()))

    def data_to_memory(self):
        """
        Loads entire dataset to memory
        """
        header_names = ['f' + str(x) if x != self.data_label_col else 'label' for x in range(self.dim + 1)]
        full_df = pd.read_csv(filepath_or_buffer=self.train_data_path, names=header_names)
        return full_df

    def load_df_round_robin_memory(self, node_id, n_nodes, max_node_samples, df_large):
        """
        Takes relevant rows from in memory dataframe for given node
        Assumes original dataset is shuffled
        """
        # Get every n_nodes row starting from
        node_df = None
        # Check number of iterations needed to resample dataframe (if replacement is needed)
        seq = ((n_nodes * max_node_samples) // df_large.shape[0]) + 1
        for i in range(seq):
            if i > 0:
                node_df = pd.concat([node_df, df_large.iloc[node_id + i:: n_nodes]]).head(max_node_samples)
            else:
                node_df = df_large.iloc[node_id::n_nodes].head(max_node_samples)

        return node_df

    def train_and_get_model(self, X, y, model=None, sampler=None):
        if isinstance(model, LinearSVC) and isinstance(sampler, RBFSampler):
            sampler = sampler
            X_train = sampler.fit_transform(X)
        elif isinstance(model, LinearSVC) and (sampler is None):
            X_train = X
        else:
            raise ValueError('Incorrect model type')

        model.fit(X_train, y)

        return model

    def score_model(self, X, y, n_components, model=None, sampler=None, ):
        if isinstance(model, LinearSVC) and (sampler is not None):
            sampler = RBFSampler(gamma=self.rff_sampler_gamma, n_components=n_components, random_state=RANDOM_STATE)
            X_test = sampler.fit_transform(X)
        elif isinstance(model, LinearSVC) and (sampler is None):
            X_test = X
        else:
            raise ValueError('Incorrect model type')

        acc = model.score(X_test, y)
        y_df_score = model.decision_function(X_test)
        rocauc = roc_auc_score(y_true=y, y_score=y_df_score)

        return acc, rocauc

    def train_for_all_nodes(self, n_nodes, n_components, full_df, max_samples, print_every=10):
        """
        Trains a model for each node and
        returns a dict with structure
        {node_id: model}
        """

        models_dict = {}
        for i in range(n_nodes):
            # Loading relevant data
            train_df = self.load_df_round_robin_memory(node_id=i, n_nodes=n_nodes, max_node_samples=max_samples,
                                                       df_large=full_df)

            if i % print_every == 0:
                print('Finished reading {} rows of data for node {}'.format(train_df.shape[0], i))

            # Splitting into features and label
            train_features = train_df.drop('label', axis=1)
            train_label = train_df['label']

            # Creating sampler if required
            if self.model_type == 'LinearSVCRFF':
                rff_sampler = RBFSampler(gamma=self.rff_sampler_gamma, n_components=n_components,
                                         random_state=LearningExperimenter.RANDOM_STATE)
            elif self.model_type == 'LinearSVC':
                rff_sampler = None
            else:
                raise ValueError('Incorrect model type given.')
            # Creating LinearSVC model
            svc_model = LinearSVC(C=self.reg_param, loss='squared_hinge', dual=False, max_iter=2000,
                                  random_state=LearningExperimenter.RANDOM_STATE)
            # Training model
            trained_model = self.train_and_get_model(X=train_features, y=train_label, model=svc_model,
                                                     sampler=rff_sampler)
            if i % print_every == 0:
                print('Trained model {}'.format(i))
            models_dict[i] = trained_model

        return models_dict

    def create_get_ts_folder(self, folder_indx, n_nodes, n_components, max_node_samples):
        """
        Creates directory with timestamp and returns path
        Also saves summary file within created folder
        """

        ts = str(datetime.now())[:19]  # Strip date and time upto seconds
        ts = re.sub(r'[\:-]', '_', ts)  # Replace unwanted chars
        ts = re.sub(r'[\s]', '__', ts)  # Replace space
        fname = '_'.join(['modSet', str(folder_indx)])
        ts_directory = os.path.join(self.results_folder_path, fname)

        if not os.path.exists(ts_directory):
            print('Creating folder {}'.format(str(ts_directory)))
            os.mkdir(path=ts_directory)

        with open(os.path.join(ts_directory, 'details.txt'), 'w') as fw:
            fw.write(
                'MODEL_TYPE={}\nDATASET_NAME={}\nN_NODES={}\nN_COMPONENTS={}\nMAX_NODE_SAMPLES={}'.format(
                    self.model_type,
                    self.dataset_name,
                    n_nodes,
                    n_components,
                    max_node_samples))

        return ts_directory

    @staticmethod
    def create_vector_param_list(models):
        vp_list = []
        for mod in models:
            wb = np.concatenate((mod.coef_.flatten(), mod.intercept_))
            vp_list.append(VectorParameter(wb))
        return vp_list

    def set_model(self, model):
        # Creating LinearSVC
        svc_model = LinearSVC(C=self.reg_param, loss='squared_hinge', dual=False, max_iter=2000,
                              random_state=LearningExperimenter.RANDOM_STATE)
        # Setting parameters
        w = model.get().tolist()
        b = w[-1]
        del w[-1]
        svc_model.coef_ = np.array(w)
        svc_model.coef_ = svc_model.coef_.reshape(1, svc_model.coef_.shape[0])
        svc_model.intercept_ = np.array([b])
        svc_model.classes_ = np.asarray([-1, 1])
        return svc_model