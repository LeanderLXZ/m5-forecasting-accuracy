import sys
import re
import time
import copy
import numpy as np
from models import utils
from models.cross_validation import CrossValidation
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from config import cfg


class ModelBase(object):
    """
        Base Model Class of Models in scikit-learn Module
    """
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te,
                 x_va=None, y_va=None, w_va=None, e_va=None, use_multi_group=False):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te

        self.x_global_valid = x_va
        self.y_global_valid = y_va
        self.w_global_valid = w_va
        self.e_global_valid = e_va

        self.importance = np.array([])
        self.indices = np.array([])
        self.std = np.array([])
        self.model_name = ''
        self.num_boost_round = 0
        self.use_multi_group = use_multi_group
        self.use_global_valid = False
        self.use_custom_obj = False
        self.postscale = False
        self.postscale_rate = None

        if cfg.group_list is None:
            if use_multi_group:
                raise ValueError("Groups not found! 'use_multi_group' should be False!")

    @staticmethod
    def get_reg(parameters):

        print('This Is Base Model!')
        reg = DecisionTreeClassifier()

        return reg

    def print_start_info(self):

        print('------------------------------------------------------')
        print('This Is Base Model!')

        self.model_name = 'base'

    @staticmethod
    def select_category_variable(x_train, x_g_train, x_valid, x_g_valid, x_test, x_g_test):

        return x_train, x_valid, x_test

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        # Get Classifier
        reg = self.get_reg(parameters)

        # Training Model
        reg.fit(x_train, y_train, sample_weight=w_train)

        return reg

    def get_pattern(self):
        return None

    def fit_with_round_log(self, boost_round_log_path, cv_count, x_train, y_train,
                           w_train, x_valid, y_valid, w_valid, parameters,
                           param_name_list, param_value_list, append_info=''):

        boost_round_log_path, _ = utils.get_boost_round_log_path(boost_round_log_path, self.model_name,
                                                                 param_name_list, param_value_list, append_info)
        boost_round_log_path += 'cv_cache/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + '_cv_{}_log.txt'.format(cv_count)

        print('Saving Outputs to:', boost_round_log_path)
        print('------------------------------------------------------')

        open(boost_round_log_path, 'w+').close()

        with open(boost_round_log_path, 'a') as f:
            __console__ = sys.stdout
            sys.stdout = f
            reg = self.fit(x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters)
            sys.stdout = __console__

        with open(boost_round_log_path) as f:
            lines = f.readlines()
            idx_round_cv = []
            train_loss_round_cv = []
            valid_loss_round_cv = []
            global_valid_loss_round_cv = []
            pattern = self.get_pattern()
            for line in lines:
                if pattern.match(line) is not None:
                    idx_round_cv.append(int(pattern.match(line).group(1)))
                    train_loss_round_cv.append(float(pattern.match(line).group(2)))
                    valid_loss_round_cv.append(float(pattern.match(line).group(3)))
                    if self.use_global_valid:
                        global_valid_loss_round_cv.append(float(pattern.match(line).group(4)))

        if self.use_global_valid:
            return reg, idx_round_cv, train_loss_round_cv, valid_loss_round_cv, global_valid_loss_round_cv
        else:
            return reg, idx_round_cv, train_loss_round_cv, valid_loss_round_cv

    def save_boost_round_log(self, boost_round_log_path, idx_round, train_loss_round_mean,
                             valid_loss_round_mean, train_seed, cv_seed, csv_idx, parameters,
                             param_name_list, param_value_list, append_info='',
                             global_valid_loss_round_mean=None, profit=None):

        boost_round_log_upper_path = \
            utils.get_boost_round_log_upper_path(
                boost_round_log_path, self.model_name, param_name_list, append_info)
        boost_round_log_path, param_name = \
            utils.get_boost_round_log_path(
                boost_round_log_path, self.model_name,
                param_name_list, param_value_list, append_info)
        utils.save_boost_round_log_to_csv(
            self.model_name, boost_round_log_path, boost_round_log_upper_path, csv_idx,
            idx_round, valid_loss_round_mean, train_loss_round_mean, train_seed, cv_seed,
            parameters, param_name_list, param_value_list, param_name, profit=profit)
        if self.use_global_valid:
            utils.save_boost_round_log_gl_to_csv(
                self.model_name, boost_round_log_path, boost_round_log_upper_path,
                csv_idx, idx_round, valid_loss_round_mean, train_loss_round_mean,
                global_valid_loss_round_mean, train_seed, cv_seed, parameters,
                param_name_list, param_value_list, param_name, profit=profit)

        boost_round_log_path += 'final_logs/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + '_' + str(csv_idx) + '_t-' \
            + str(train_seed) + '_c-' + str(cv_seed) + '_log.csv'

        if self.use_global_valid:
            utils.save_final_boost_round_gl_log(
                boost_round_log_path, idx_round, train_loss_round_mean,
                valid_loss_round_mean, global_valid_loss_round_mean, profit=profit)
        else:
            utils.save_final_boost_round_log(
                boost_round_log_path, idx_round, train_loss_round_mean,
                valid_loss_round_mean, profit=profit)

    def get_importance(self, reg):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = reg.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (
                f + 1, self.indices[f], self.importance[self.indices[f]]))

    def predict(self, reg, x_test, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Test Result...')

        pred_test = np.array(reg.predict(x_test))

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, pred_test)

        return pred_test

    def get_pred_train(self, reg, x_train, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Train Probability...')

        pred_train = np.array(reg.predict(x_train))[:, 1]

        if pred_path is not None:
            utils.save_pred_train_to_csv(pred_path, pred_train, self.y_train)

        return pred_train

    def save_csv_log(self, mode, csv_log_path, param_name_list, param_value_list, csv_idx,
                     loss_train_w_mean, loss_valid_w_mean, acc_train, train_seed, cv_seed,
                     n_valid, n_cv, parameters, boost_round_log_path=None,
                     file_name_params=None, append_info='', loss_global_valid=None,
                     acc_global_valid=None, profit=None):

        if mode == 'auto_grid_search':

            csv_log_path, param_name, param_info = \
                utils.get_grid_search_log_path(csv_log_path, self.model_name,
                                               param_name_list, param_value_list, append_info)
            if self.use_global_valid:
                utils.save_grid_search_log_with_glv_to_csv(
                    csv_idx, csv_log_path + param_name + '_',
                    loss_train_w_mean, loss_valid_w_mean, acc_train, train_seed,
                    loss_global_valid, acc_global_valid, cv_seed, n_valid, n_cv,
                    parameters, param_name_list, param_value_list, profit=profit)
                csv_log_path += str(param_info) + '_'
                utils.save_grid_search_log_with_glv_to_csv(
                    csv_idx, csv_log_path, loss_train_w_mean, loss_valid_w_mean,
                    acc_train, train_seed, loss_global_valid, acc_global_valid,
                    cv_seed, n_valid, n_cv, parameters, param_name_list,
                    param_value_list, profit=profit)
            else:
                utils.save_grid_search_log_to_csv(
                    csv_idx, csv_log_path + param_name + '_', loss_train_w_mean,
                    loss_valid_w_mean, acc_train, train_seed, cv_seed, n_valid,
                    n_cv, parameters, param_name_list, param_value_list, profit=profit)
                csv_log_path += str(param_info) + '_'
                utils.save_grid_search_log_to_csv(
                    csv_idx, csv_log_path, loss_train_w_mean, loss_valid_w_mean,
                    acc_train, train_seed, cv_seed, n_valid, n_cv, parameters,
                    param_name_list, param_value_list, profit=profit)

        elif mode == 'auto_train_boost_round':

            boost_round_log_path, _ = \
                utils.get_boost_round_log_path(
                    boost_round_log_path, self.model_name, param_name_list, param_value_list, append_info)
            boost_round_log_path += self.model_name + '_' + append_info + '_'
            if self.use_global_valid:
                utils.save_grid_search_log_to_csv(
                    csv_idx, boost_round_log_path, loss_train_w_mean, loss_valid_w_mean,
                    acc_train, train_seed, cv_seed, n_valid, n_cv, parameters,
                    param_name_list, param_value_list, profit=profit)
            else:
                utils.save_final_loss_log_to_csv(
                    csv_idx, boost_round_log_path, loss_train_w_mean,
                    loss_valid_w_mean, acc_train, train_seed, cv_seed,
                    n_valid, n_cv, parameters, profit=profit)

        elif mode == 'auto_train':

            csv_log_path += self.model_name + '/'
            utils.check_dir([csv_log_path])
            csv_log_path += self.model_name + '_' + append_info + '/'
            utils.check_dir([csv_log_path])
            csv_log_path += self.model_name + '_'
            if file_name_params is not None:
                for p_name in file_name_params:
                    csv_log_path += str(parameters[p_name]) + '_'
            else:
                for p_name, p_value in parameters.items():
                    csv_log_path += str(p_value) + '_'

            if self.use_global_valid:
                utils.save_log_with_glv_to_csv(
                    csv_idx, csv_log_path, loss_train_w_mean,
                    loss_valid_w_mean, acc_train, train_seed,
                    loss_global_valid, acc_global_valid, cv_seed,
                    n_valid, n_cv, parameters, profit=profit)
            else:
                utils.save_final_loss_log_to_csv(
                    csv_idx, csv_log_path, loss_train_w_mean,
                    loss_valid_w_mean, acc_train, train_seed,
                    cv_seed, n_valid, n_cv, parameters, profit=profit)

        else:

            csv_log_path += self.model_name + '_' + append_info + '_'
            if self.use_global_valid:
                utils.save_log_with_glv_to_csv(csv_idx, csv_log_path, loss_train_w_mean,
                                               loss_valid_w_mean, acc_train, train_seed,
                                               loss_global_valid, acc_global_valid, cv_seed,
                                               n_valid, n_cv, parameters, profit=profit)
            else:
                utils.save_final_loss_log_to_csv(csv_idx, csv_log_path, loss_train_w_mean,
                                                 loss_valid_w_mean, acc_train, train_seed,
                                                 cv_seed, n_valid, n_cv, parameters, profit=profit)

    def save_final_pred(self, mode, pred_test_mean, pred_path, parameters, csv_idx,
                        train_seed, cv_seed, boost_round_log_path=None, param_name_list=None,
                        param_value_list=None, file_name_params=None, append_info=''):

        params = '_'
        if file_name_params is not None:
            for p_name in file_name_params:
                params += utils.get_simple_param_name(p_name) + \
                          '-' + str(parameters[p_name]) + '_'
        else:
            for p_name, p_value in parameters.items():
                params += utils.get_simple_param_name(p_name) + '-' + str(p_value) + '_'

        if mode == 'auto_train':

            pred_path += self.model_name + '/'
            utils.check_dir([pred_path])
            pred_path += self.model_name + '_' + append_info + '/'
            utils.check_dir([pred_path])
            pred_path += self.model_name + params + 'results/'
            utils.check_dir([pred_path])
            pred_path += self.model_name + '_' + str(csv_idx) + \
                '_t-' + str(train_seed) + '_c-' + str(cv_seed) + '_'
            utils.save_pred_to_csv(pred_path, self.id_test, pred_test_mean)

        elif mode == 'auto_train_boost_round':

            boost_round_log_path, _ = \
                utils.get_boost_round_log_path(
                    boost_round_log_path, self.model_name, param_name_list, param_value_list, append_info)
            pred_path = boost_round_log_path + 'final_results/'
            utils.check_dir([pred_path])
            pred_path += self.model_name + '_' + str(csv_idx) + \
                '_t-' + str(train_seed) + '_c-' + str(cv_seed) + '_'
            utils.save_pred_to_csv(pred_path, self.id_test, pred_test_mean)

        else:
            pred_path += 'final_results/'
            utils.check_dir([pred_path])
            pred_path += self.model_name + '_' + append_info + '/'
            utils.check_dir([pred_path])
            pred_path += self.model_name + '_t-' + str(train_seed) + '_c-' + str(cv_seed) + params
            utils.save_pred_to_csv(pred_path, self.id_test, pred_test_mean)

    @staticmethod
    def get_postscale_rate(y):

        positive = 0
        for y_ in y:
            if y_ == 1:
                positive += 1

        positive_rate = positive / len(y)
        postscale_rate = len(y) / (2*positive)

        return positive_rate, postscale_rate

    @staticmethod
    def prescale(x_train, y_train, w_train, e_train):

        print('[W] PreScaling Train Set...')

        positive_idx = []
        negative_idx = []
        for i, y in enumerate(y_train):
            if y == 1:
                positive_idx.append(i)
            else:
                negative_idx.append(i)
        n_positive = len(positive_idx)
        n_negative = len(negative_idx)
        print('Number of Positive Labels: {}'.format(n_positive))
        print('Number of Negative Labels: {}'.format(n_negative))

        if n_positive > n_negative:
            positive_idx = list(np.random.choice(positive_idx, len(negative_idx), replace=False))
        elif n_negative > n_positive:
            negative_idx = list(np.random.choice(negative_idx, n_positive, replace=False))

        # Checking
        if len(positive_idx) != len(negative_idx):
            raise ValueError('PreScaling Failed! len(positive_idx) != len(negative_idx)!')
        else:
            print('Number of PreScaled Labels: {}'.format(len(positive_idx)))

        prescale_idx = list(np.sort(positive_idx + negative_idx))
        x_train = x_train[prescale_idx]
        y_train = y_train[prescale_idx]
        w_train = w_train[prescale_idx]
        e_train = e_train[prescale_idx]
        print('------------------------------------------------------')

        return x_train, y_train, w_train, e_train

    def lgb_postscale_feval(self, preds, train_data):

        pred = copy.deepcopy(preds)
        labels = train_data.get_label()
        weights = train_data.get_weight()
        pred *= self.postscale_rate
        loss = utils.log_loss_with_weight(pred, labels, weights)

        return 'binary_logloss', loss, False

    def xgb_postscale_feval(self, preds, train_data):

        pred = copy.deepcopy(preds)
        labels = train_data.get_label()
        weights = train_data.get_weight()
        pred *= self.postscale_rate
        loss = utils.log_loss_with_weight(pred, labels, weights)

        return 'logloss', loss

    def train(self, pred_path=None, loss_log_path=None, csv_log_path=None, boost_round_log_path=None,
              train_seed=None, cv_args=None, parameters=None, show_importance=False, show_accuracy=False,
              save_cv_pred=True, save_cv_pred_train=False, save_final_pred=True, save_final_pred_train=False,
              save_csv_log=True, csv_idx=None, prescale=False, postscale=False, use_global_valid=False,
              return_pred_test=False, mode=None, param_name_list=None, param_value_list=None,
              use_custom_obj=False, use_scale_pos_weight=False, file_name_params=None, append_info=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)
        utils.check_dir([pred_path, loss_log_path, csv_log_path, boost_round_log_path])

        # Global Validation
        self.use_global_valid = use_global_valid

        # Use Custom Objective Function
        self.use_custom_obj = use_custom_obj

        # Cross Validation Arguments
        cv_args_copy, n_valid, n_cv, n_era, cv_seed = utils.get_cv_args(cv_args, append_info)

        if csv_idx is None:
            csv_idx = self.model_name

        # Print Start Information and Get Model Name
        self.print_start_info()

        if use_global_valid:
            print('------------------------------------------------------')
            print('[W] Using Global Validation...')

        cv_count = 0
        pred_test_total = []
        pred_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []
        idx_round = []
        train_loss_round_total = []
        valid_loss_round_total = []
        global_valid_loss_round_total = []
        pred_global_valid_total = []
        loss_global_valid_total = []
        loss_global_valid_w_total = []

        # Get Cross Validation Generator
        if 'cv_generator' in cv_args_copy:
            cv_generator = cv_args_copy['cv_generator']
            if cv_generator is None:
                cv_generator = CrossValidation.era_k_fold
            cv_args_copy.pop('cv_generator')
        else:
            cv_generator = CrossValidation.era_k_fold
        print('------------------------------------------------------')
        print('[W] Using CV Generator: {}'.format(getattr(cv_generator, '__name__')))

        if 'era_list' in cv_args_copy:
            print('Era List: ', cv_args_copy['era_list'])
        if 'window_size' in cv_args_copy:
            print('Window Size: ', cv_args_copy['window_size'])
        if 'cv_weights' in cv_args_copy:
            cv_weights = cv_args_copy['cv_weights']
            cv_args_copy.pop('cv_weights')
            if cv_weights is not None:
                if len(cv_weights) != n_cv:
                    raise ValueError("The length of 'cv_weights'({}) should be equal to 'n_cv'({})!"
                                     .format(len(cv_weights), n_cv))
        else:
            cv_weights = None

        # Training on Cross Validation Sets
        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era \
                in cv_generator(x=self.x_train, y=self.y_train,
                                w=self.w_train, e=self.e_train, **cv_args_copy):

            # CV Start Time
            cv_start_time = time.time()

            cv_count += 1

            # Get Positive Rate of Train Set and postscale Rate
            positive_rate_train, postscale_rate = self.get_postscale_rate(y_train)
            positive_rate_valid, _ = self.get_postscale_rate(y_valid)

            # Remove Metric of Post Scale
            if postscale:
                self.postscale = True
                self.postscale_rate = postscale_rate
                if 'metric' in parameters.keys():
                    parameters.pop('metric')
                if 'eval_metric' in parameters.keys():
                    parameters.pop('eval_metric')

            if use_scale_pos_weight:
                if self.model_name == 'xgb':
                    parameters['scale_pos_weight'] = postscale_rate

            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            print('Number of Features: ', x_train.shape[1])
            print('------------------------------------------------------')
            print('Positive Rate of Train Set: {:.6f}'.format(positive_rate_train))
            print('Positive Rate of Valid Set: {:.6f}'.format(positive_rate_valid))
            print('------------------------------------------------------')

            # prescale
            if prescale:
                x_train, y_train, w_train, e_train = self.prescale(x_train, y_train, w_train, e_train)

            # Fitting and Training Model
            if mode == 'auto_train_boost_round':
                if use_global_valid:
                    reg, idx_round_cv, train_loss_round_cv, \
                        valid_loss_round_cv, global_valid_loss_round_cv = \
                        self.fit_with_round_log(
                            boost_round_log_path, cv_count, x_train, y_train, w_train, x_valid, y_valid,
                            w_valid, parameters, param_name_list, param_value_list, append_info=append_info)
                    global_valid_loss_round_total.append(global_valid_loss_round_cv)
                else:
                    reg, idx_round_cv, train_loss_round_cv, valid_loss_round_cv = \
                        self.fit_with_round_log(
                            boost_round_log_path, cv_count, x_train, y_train, w_train, x_valid, y_valid,
                            w_valid, parameters, param_name_list, param_value_list, append_info=append_info)

                idx_round = idx_round_cv
                train_loss_round_total.append(train_loss_round_cv)
                valid_loss_round_total.append(valid_loss_round_cv)
            else:
                reg = self.fit(x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters)

            # Feature Importance
            if show_importance:
                self.get_importance(reg)

            # Prediction
            if save_cv_pred:
                cv_pred_path = \
                    pred_path + 'cv_results/' + self.model_name + '_cv_{}_'.format(cv_count)
            else:
                cv_pred_path = None
            pred_test = self.predict(reg, self.x_test, pred_path=cv_pred_path)

            # Save Train Probabilities to CSV File
            if save_cv_pred_train:
                cv_pred_train_path = \
                    pred_path + 'cv_pred_train/' + self.model_name + '_cv_{}_'.format(cv_count)
            else:
                cv_pred_train_path = None
            pred_train = self.get_pred_train(reg, x_train, pred_path=cv_pred_train_path)
            pred_train_all = self.get_pred_train(reg, self.x_train, pred_path=cv_pred_train_path)

            # Predict Global Validation Set
            if use_global_valid:
                pred_global_valid = self.predict(reg, self.x_global_valid)
            else:
                pred_global_valid = np.array([])

            # Get Probabilities of Validation Set
            pred_valid = self.predict(reg, x_valid)

            # postscale
            if postscale:
                print('------------------------------------------------------')
                print('[W] PostScaling Results...')
                print('PostScale Rate: {:.6f}'.format(postscale_rate))
                pred_test *= postscale_rate
                pred_train *= postscale_rate
                pred_valid *= postscale_rate
                if use_global_valid:
                    pred_global_valid *= postscale_rate

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss(pred_train, y_train, w_train, pred_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(pred_train, y_train, e_train,
                                             pred_valid, y_valid, e_valid, show_accuracy)

            # Print Loss and Accuracy of Global Validation Set
            if use_global_valid:
                loss_global_valid, loss_global_valid_w, acc_global_valid = \
                    utils.print_global_valid_loss_and_acc(
                        pred_global_valid, self.y_global_valid, self.w_global_valid)
                pred_global_valid_total.append(pred_global_valid)
                loss_global_valid_total.append(loss_global_valid)
                loss_global_valid_w_total.append(loss_global_valid_w)

            # Save Losses to File
            utils.save_loss_log(
                loss_log_path + self.model_name + '_', cv_count, parameters, n_valid, n_cv,
                valid_era, loss_train, loss_valid, loss_train_w, loss_valid_w, train_seed,
                cv_seed, acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era)

            pred_test_total.append(pred_test)
            pred_train_total.append(pred_train_all)
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

            # CV End Time
            print('------------------------------------------------------')
            print('CV Done! Using Time: {}s'.format(time.time() - cv_start_time))

        print('======================================================')
        print('Calculating Final Result...')

        # Calculate Means of pred and losses
        pred_test_mean, pred_train_mean, loss_train_mean, \
            loss_valid_mean, loss_train_w_mean, loss_valid_w_mean = \
            utils.calculate_means(pred_test_total, pred_train_total, loss_train_total, loss_valid_total,
                                  loss_train_w_total, loss_valid_w_total, weights=cv_weights)

        # Save 'num_boost_round'
        if self.model_name in ['xgb', 'lgb']:
            parameters['num_boost_round'] = self.num_boost_round

        # Calculate Profit
        profit = 0

        # Save Logs of num_boost_round
        if mode == 'auto_train_boost_round':
            if use_global_valid:
                train_loss_round_mean, valid_loss_round_mean, global_valid_loss_round_mean = \
                    utils.calculate_boost_round_means(
                        train_loss_round_total, valid_loss_round_total, weights=cv_weights,
                        global_valid_loss_round_total=global_valid_loss_round_total)
                self.save_boost_round_log(
                    boost_round_log_path, idx_round, train_loss_round_mean,
                    valid_loss_round_mean, train_seed, cv_seed, csv_idx,
                    parameters, param_name_list, param_value_list, append_info=append_info,
                    global_valid_loss_round_mean=global_valid_loss_round_mean, profit=profit)
            else:
                train_loss_round_mean, valid_loss_round_mean = \
                    utils.calculate_boost_round_means(
                        train_loss_round_total, valid_loss_round_total, weights=cv_weights)
                self.save_boost_round_log(
                    boost_round_log_path, idx_round, train_loss_round_mean,
                    valid_loss_round_mean, train_seed, cv_seed, csv_idx, parameters,
                    param_name_list, param_value_list, append_info=append_info, profit=profit)

        # Save Final Result
        if save_final_pred:
            self.save_final_pred(
                mode, pred_test_mean, pred_path, parameters, csv_idx, train_seed,
                cv_seed, boost_round_log_path, param_name_list, param_value_list,
                file_name_params=file_name_params, append_info=append_info)

        # Save Final pred_train
        if save_final_pred_train:
            utils.save_pred_train_to_csv(pred_path + 'final_pred_train/' + self.model_name + '_',
                                         pred_train_mean, self.y_train)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean,
                               loss_valid_w_mean, profit=profit)

        # Print and Get Accuracies of CV of All Train Set
        acc_train, acc_train_era = \
            utils.print_and_get_train_accuracy(pred_train_mean, self.y_train, self.e_train, show_accuracy)

        # Save Final Losses to File
        utils.save_final_loss_log(
            loss_log_path + self.model_name + '_', parameters, n_valid, n_cv,
            loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean,
            train_seed, cv_seed, acc_train, acc_train_era)

        # Print Global Validation Information and Save
        if use_global_valid:
            # Calculate Means of Probabilities and Losses
            pred_global_valid_mean, loss_global_valid_mean, loss_global_valid_w_mean = \
                utils.calculate_global_valid_means(pred_global_valid_total, loss_global_valid_total,
                                                   loss_global_valid_w_total, weights=cv_weights)
            # Print Loss and Accuracy
            acc_total_global_valid = \
                utils.print_total_global_valid_loss_and_acc(
                    pred_global_valid_mean, self.y_global_valid,
                    loss_global_valid_mean, loss_global_valid_w_mean)
            # Save csv log
            if save_csv_log:
                self.save_csv_log(
                    mode, csv_log_path, param_name_list, param_value_list, csv_idx, loss_train_w_mean,
                    loss_valid_w_mean, acc_train, train_seed, cv_seed, n_valid, n_cv, parameters,
                    boost_round_log_path=boost_round_log_path, file_name_params=file_name_params,
                    append_info=append_info, loss_global_valid=loss_global_valid_w_mean,
                    acc_global_valid=acc_total_global_valid, profit=profit)

        # Save Loss Log to csv File
        if save_csv_log:
            if not use_global_valid:
                self.save_csv_log(
                    mode, csv_log_path, param_name_list, param_value_list, csv_idx, loss_train_w_mean,
                    loss_valid_w_mean, acc_train, train_seed, cv_seed, n_valid, n_cv, parameters,
                    boost_round_log_path=boost_round_log_path, file_name_params=file_name_params,
                    append_info=append_info, profit=profit)

        # Remove 'num_boost_round' of parameters
        if 'num_boost_round' in parameters:
            parameters.pop('num_boost_round')

        # Return Final Result
        if return_pred_test:
            return pred_test_mean

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        # Select Group Variable
        x_train, x_valid, x_test = self.select_category_variable(x_train, x_g_train, x_valid,
                                                                 x_g_valid, x_test, x_g_test)

        # Print Start Information and Get Model Name
        self.print_start_info()
        print('Number of Features: ', x_train.shape[1])
        print('------------------------------------------------------')

        # Fitting and Training Model
        reg = self.fit(x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters)

        # Feature Importance
        if show_importance:
            self.get_importance(reg)

        # Prediction
        pred_train = self.predict(reg, x_train)
        pred_valid = self.predict(reg, x_valid)
        pred_test = self.predict(reg, x_test)

        # Print LogLoss
        loss_train, loss_valid, loss_train_w, loss_valid_w = \
            utils.print_loss(pred_train, y_train, w_train, pred_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        return pred_valid, pred_test, losses


class LRegression(ModelBase):
    """
        Logistic Regression
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = LogisticRegression(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training Logistic Regression...')

        self.model_name = 'lr'

    def get_importance(self, reg):

        print('------------------------------------------------------')
        print('Feature Importance')
        self.importance = np.abs(reg.coef_)[0]
        indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d | feature %d | %f" % (f + 1, indices[f], self.importance[indices[f]]))


class KNearestNeighbor(ModelBase):
    """
        k-Nearest Neighbor Classifier
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = KNeighborsClassifier(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training k-Nearest Neighbor Classifier...')

        self.model_name = 'knn'


class SupportVectorClustering(ModelBase):
    """
        SVM - Support Vector Clustering
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = SVC(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training Support Vector Clustering...')

        self.model_name = 'svc'


class Gaussian(ModelBase):
    """
        Gaussian NB
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = GaussianNB(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training Gaussian...')

        self.model_name = 'gs'


class DecisionTree(ModelBase):
    """
        Decision Tree
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = DecisionTreeClassifier(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training Decision Tree...')

        self.model_name = 'dt'


class RandomForest(ModelBase):
    """
        Random forecast
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = RandomForestRegressor(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training Random forecast...')

        self.model_name = 'rf'


class ExtraTrees(ModelBase):
    """
        Extra Trees
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = ExtraTreesClassifier(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training Extra Trees...')

        self.model_name = 'et'


class AdaBoost(ModelBase):
    """
        AdaBoost
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = AdaBoostClassifier(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training AdaBoost...')

        self.model_name = 'ab'


class GradientBoosting(ModelBase):
    """
        Gradient Boosting
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = GradientBoostingClassifier(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training Gradient Boosting...')

        self.model_name = 'gb'


class XGBoost(ModelBase):
    """
        XGBoost
    """
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te,
                 x_va=None, y_va=None, w_va=None, e_va=None,
                 num_boost_round=None, use_multi_group=False):

        super(XGBoost, self).__init__(x_tr, y_tr, w_tr, e_tr, x_te, id_te,
                                      x_va, y_va, w_va, e_va, use_multi_group)

        self.num_boost_round = num_boost_round

    def print_start_info(self):

        print('======================================================')
        print('Training XGBoost...')

        self.model_name = 'xgb'

    @staticmethod
    def logloss_obj(pred, d_train):

        y = d_train.get_label()

        grad = (pred - y) / ((1.0 - pred) * pred)
        hess = (pred * pred - 2.0 * pred * y + y) / ((1.0 - pred) * (1.0 - pred) * pred * pred)

        return grad, hess

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        d_train = xgb.DMatrix(x_train, label=y_train, weight=w_train)
        d_valid = xgb.DMatrix(x_valid, label=y_valid, weight=w_valid)

        # Booster
        if self.use_global_valid:
            d_gl_valid = xgb.DMatrix(self.x_global_valid, label=self.y_global_valid, weight=self.w_global_valid)
            eval_list = [(d_train, 'Train'), (d_valid, 'Valid'), (d_gl_valid, 'Global_Valid')]
        else:
            eval_list = [(d_train, 'Train'), (d_valid, 'Valid')]

        if self.postscale:
            if self.use_custom_obj:
                bst = xgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                                evals=eval_list, obj=self.logloss_obj, feval=self.xgb_postscale_feval)
            else:
                bst = xgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                                evals=eval_list, feval=self.xgb_postscale_feval)
        else:
            if self.use_custom_obj:
                bst = xgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                                obj=self.logloss_obj, evals=eval_list)
            else:
                bst = xgb.train(parameters, d_train, num_boost_round=self.num_boost_round, evals=eval_list)

        return bst

    def get_pattern(self):

        if self.use_global_valid:
            if self.postscale:
                return re.compile(r'\[(\d*)\].*\tTrain-logloss:(.*)\tValid-logloss:(.*)\tGlobal_Valid-logloss:(.*)')
            else:
                return re.compile(r'\[(\d*)\]\tTrain-logloss:(.*)\tValid-logloss:(.*)\tGlobal_Valid-logloss:(.*)')
        else:
            if self.postscale:
                return re.compile(r'\[(\d*)\].*\tTrain-logloss:(.*)\tValid-logloss:(.*)')
            else:
                return re.compile(r'\[(\d*)\]\tTrain-logloss:(.*)\tValid-logloss:(.*)')

    def get_importance(self, model):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = model.get_fscore()
        sorted_importance = sorted(self.importance.items(), key=lambda d: d[1], reverse=True)

        feature_num = len(self.importance)

        for i in range(feature_num):
            print('{} | feature {} | {}'.format(i + 1, sorted_importance[i][0], sorted_importance[i][1]))

    def predict(self, model, x_test, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Test Probability...')

        pred_test = model.predict(xgb.DMatrix(x_test))

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, pred_test)

        return pred_test

    def get_pred_train(self, model, x_train, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Train Probability...')

        pred_train = model.predict(xgb.DMatrix(x_train))

        if pred_path is not None:
            utils.save_pred_train_to_csv(pred_path, pred_train, self.y_train)

        return pred_train


class SKLearnXGBoost(ModelBase):
    """
        XGBoost using sklearn module
    """
    @staticmethod
    def get_reg(parameters=None):

        print('Initialize Model...')
        reg = XGBClassifier(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training XGBoost(sklearn)...')

        self.model_name = 'xgb_sk'

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        # Get Classifier
        reg = self.get_reg(parameters)

        # Training Model
        reg.fit(x_train, y_train, sample_weight=w_train,
                eval_set=[(x_train, y_train), (x_valid, y_valid)],
                early_stopping_rounds=100, eval_metric='logloss', verbose=True)

        return reg

    def get_importance(self, reg):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = reg.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %f" % (f + 1, self.indices[f], self.importance[self.indices[f]]))


class LightGBM(ModelBase):
    """
        LightGBM
    """
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te,
                 x_va=None, y_va=None, w_va=None, e_va=None,
                 num_boost_round=None, use_multi_group=False):

        super(LightGBM, self).__init__(x_tr, y_tr, w_tr, e_tr, x_te, id_te,
                                       x_va, y_va, w_va, e_va, use_multi_group)

        self.num_boost_round = num_boost_round

    def print_start_info(self):

        print('======================================================')
        print('Training LightGBM...')

        self.model_name = 'lgb'

    @staticmethod
    def select_category_variable(x_train, x_g_train, x_valid, x_g_valid, x_test, x_g_test):

        return x_g_train, x_g_valid, x_g_test

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        # Get Category Feature's Index
        idx_category = utils.get_idx_category(x_train, self.use_multi_group)

        d_train = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=idx_category)
        d_valid = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, categorical_feature=idx_category)

        # Booster
        if self.use_global_valid:
            d_gl_valid = lgb.Dataset(self.x_global_valid, label=self.y_global_valid,
                                     weight=self.w_global_valid, categorical_feature=idx_category)
            if self.postscale:
                bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                                valid_sets=[d_valid, d_gl_valid, d_train],
                                valid_names=['Valid', 'Global_Valid', 'Train'], feval=self.lgb_postscale_feval)
            else:
                bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                                valid_sets=[d_valid, d_gl_valid, d_train],
                                valid_names=['Valid', 'Global_Valid', 'Train'])
        else:
            if self.postscale:
                bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                                valid_sets=[d_valid, d_train], valid_names=['Valid', 'Train'],
                                feval=self.lgb_postscale_feval)
            else:
                bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                                valid_sets=[d_valid, d_train], valid_names=['Valid', 'Train'])

        return bst

    def get_pattern(self):

        if self.use_global_valid:
            return re.compile(r"\[(\d*)\]\tTrain\'s binary_logloss: (.*)\tValid\'s binary_logloss:(.*)\tGlobal_Valid\'s binary_logloss:(.*)")
        else:
            return re.compile(r"\[(\d*)\]\tTrain\'s binary_logloss: (.*)\tValid\'s binary_logloss:(.*)")

    @staticmethod
    def logloss_obj(y, pred):

        grad = (pred - y) / ((1 - pred) * pred)
        hess = (pred * pred - 2 * pred * y + y) / ((1 - pred) * (1 - pred) * pred * pred)

        return grad, hess

    def get_importance(self, bst):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = bst.feature_importance()
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

        print('\n')

    def predict(self, bst, x_test, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Test Probability...')

        pred_test = bst.predict(x_test)

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, pred_test)

        return pred_test

    def get_pred_train(self, bst, x_train, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Train Probability...')

        pred_train = bst.predict(x_train)

        if pred_path is not None:
            utils.save_pred_train_to_csv(pred_path, pred_train, self.y_train)

        return pred_train


class SKLearnLightGBM(ModelBase):
    """
        LightGBM using sklearn module
    """
    @staticmethod
    def get_reg(parameters=None):

        print('Initialize Model...')
        reg = LGBMClassifier(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training LightGBM(sklearn)...')

        self.model_name = 'lgb_sk'

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        # Get Classifier
        reg = self.get_reg(parameters)

        # Get Category Feature's Index
        idx_category = utils.get_idx_category(x_train, self.use_multi_group)

        # Fitting and Training Model
        reg.fit(x_train, y_train, sample_weight=w_train, categorical_feature=idx_category,
                eval_set=[(x_train, y_train), (x_valid, y_valid)], eval_names=['train', 'eval'],
                early_stopping_rounds=100, eval_sample_weight=[w_train, w_valid],
                eval_metric='logloss', verbose=True)

        return reg


class CatBoost(ModelBase):
    """
        CatBoost
    """
    @staticmethod
    def get_reg(parameters=None):

        reg = CatBoostClassifier(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training CatBoost...')

        self.model_name = 'cb'

    @staticmethod
    def select_category_variable(x_train, x_g_train, x_valid, x_g_valid, x_test, x_g_test):

        return x_g_train, x_g_valid, x_g_test

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        # Get Classifier
        reg = self.get_reg(parameters)

        # Get Category Feature's Index
        idx_category = utils.get_idx_category(x_train, self.use_multi_group)

        # Convert Zeros in Weights to Small Positive Numbers
        w_train = [0.001 if w == 0 else w for w in w_train]

        # Fitting and Training Model
        reg.fit(X=x_train, y=y_train, cat_features=idx_category, sample_weight=w_train,
                baseline=None, use_best_model=None, eval_set=(x_valid, y_valid), verbose=True, plot=False)

        return reg

    def get_pattern(self):

        return re.compile(r'(\d*):\tlearn (.*)\ttest (.*)\tbestTest')

    def get_importance(self, reg):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = reg.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))
