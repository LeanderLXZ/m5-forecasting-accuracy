import numpy as np
from models import utils
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor




class ModelBase(object):
    """
        Base Model Class of Models in scikit-learn Module
    """
    def __init__(self, x, y, pred_head, forecast_num, idx=None, mode=None):
        self.x = x
        self.y = y
        self.pred_head = pred_head
        self.forecast_num = forecast_num
        self.idx = idx
        self.mode = mode

    @staticmethod
    def get_reg(parameters):
        return DecisionTreeRegressor()

    def fit(self, parameters=None, early_stopping_rounds=None):
        # Get Classifier
        reg = self.get_reg(parameters)
        # Training Model
        reg.fit(self.x, self.y)
        return reg

    def predict(self,
                reg,
                head,
                forecast_num,
                use_month_features=False,
                time_features=None):
        """
        :param reg: regressor
        :param head: np.array
        :param forecast_num: forecast
        :param use_month_features: use month_features or not
        :param time_features: time features in past like (30, 50, 100)
        :return: prediction series
        """
        pred_list = []
        n_tf = len(time_features)
        for i in range(forecast_num):
            pred_i = reg.predict(head.reshape(1, -1))[0]
            pred_list.append(pred_i)
            for f_i, feature_num_i in enumerate(time_features):
                if feature_num_i > i+1:
                    n_y_pre = feature_num_i - (i+1)
                    y_pre = self.y[-n_y_pre:]
                    y = np.append(y_pre, pred_list)
                else:
                    y = pred_list[-feature_num_i:]
                assert len(y) == feature_num_i
                feature_i = np.mean(y)
                if use_month_features:
                    head[f_i+1] = feature_i
                else:
                    head[f_i] = feature_i
            if use_month_features:
                head[n_tf+1:-1], head[-1] = head[n_tf+2:], pred_i
            else:
                head[n_tf:-1], head[-1] = head[n_tf+1:], pred_i
            # print(i, pred_i, '\n', head)
        return np.array(pred_list)

    def get_importance(self, reg):
        # # print('------------------------------------------------------')
        # # print('Feature Importance')
        # importance = reg.feature_importances_
        # indices = np.argsort(importance)[::-1]
        # feature_num = len(importance)
        # for f in range(feature_num):
        #     print("%d | feature %d | %d" % (
        #         f + 1, indices[f], importance[indices[f]]))

        importance = reg.feature_importances_
        indices = np.argsort(importance)[::-1]
        feature_num = len(importance)
        features_list = []
        importance_list = []
        for f in range(feature_num):
            if importance[indices[f]] != 0:
                features_list.append('feature_{}'.format(indices[f]))
                importance_list.append(importance[indices[f]])

        utils.save_importance(self.mode,
                              self.idx,
                              features_list,
                              importance_list)

    def train(self,
              parameters=None,
              time_features=None,
              use_month_features=False,
              early_stopping_rounds=None):

        # Fitting and Training Model
        reg = self.fit(parameters,
                       early_stopping_rounds=early_stopping_rounds)
        # Feature Importance
        self.get_importance(reg)
        # Predicting
        pred = self.predict(
            reg, self.pred_head, self.forecast_num,
            use_month_features=use_month_features,
            time_features=time_features)

        return pred


class KNearestNeighbor(ModelBase):
    """
        k-Nearest Neighbor Regressor
    """
    @staticmethod
    def get_reg(parameters):
        return KNeighborsRegressor(**parameters)


class SupportVectorMachine(ModelBase):
    """
        SVM - Support Vector Machine
    """
    @staticmethod
    def get_reg(parameters):
        return SVR(**parameters)


class DecisionTree(ModelBase):
    """
        Decision Tree
    """
    @staticmethod
    def get_reg(parameters):
        return DecisionTreeRegressor(**parameters)

class RandomForest(ModelBase):
    """
        Random forecast
    """
    @staticmethod
    def get_reg(parameters):
        return RandomForestRegressor(**parameters)


class ExtraTrees(ModelBase):
    """
        Extra Trees
    """
    @staticmethod
    def get_reg(parameters):
        return ExtraTreesRegressor(**parameters)


class AdaBoost(ModelBase):
    """
        AdaBoost
    """
    @staticmethod
    def get_reg(parameters):
        return AdaBoostRegressor(**parameters)


class GradientBoosting(ModelBase):
    """
        Gradient Boosting
    """
    @staticmethod
    def get_reg(parameters):
        return GradientBoostingRegressor(**parameters)

class XGBoost(ModelBase):
    """
        XGBoost using sklearn module
    """
    @staticmethod
    def get_reg(parameters=None):
        return XGBRegressor(**parameters)

    def fit(self, parameters=None, early_stopping_rounds=1000):
        reg = self.get_reg(parameters)
        reg.fit(self.x, self.y,
                eval_set=[(self.x, self.y)],
                early_stopping_rounds=early_stopping_rounds,
                eval_metric='mae', verbose=False)
        return reg


class LightGBM(ModelBase):
    """
        LightGBM using sklearn module
    """
    @staticmethod
    def get_reg(parameters=None):
        return LGBMRegressor(**parameters)

    def fit(self, parameters=None, early_stopping_rounds=1000):
        reg = self.get_reg(parameters)
        reg.fit(self.x, self.y,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False)
        return reg


class MachineLearningModel(object):

    def __init__(self, x, y, pred_head, forecast_num):
        self.x = x
        self.y = y
        self.pred_head = pred_head
        self.forecast_num = forecast_num

    def predict(self,
                model_name,
                time_features=None,
                use_month_features=False,
                model_parameters=None,
                idx=None,
                mode=None):

        train_seed = 95
        early_stopping_rounds = None

        if model_parameters:
            if 'early_stopping_rounds' in model_parameters.keys():
                early_stopping_rounds = \
                    model_parameters['early_stopping_rounds']
                model_parameters.pop('early_stopping_rounds')

        if model_name == 'knn':
            parameters ={'n_neighbors': 5,
                         'weights': 'uniform',
                         'algorithm': 'auto',
                         'leaf_size': 30,
                         'p': 2,
                         'metric': 'minkowski',
                         'metric_params': None,
                         'n_jobs': -1}
            if model_parameters:
                for model_param in model_parameters.keys():
                    parameters[model_param] = \
                        model_parameters[model_param]
            pred = KNearestNeighbor(
                self.x, self.y, self.pred_head,
                self.forecast_num, idx=idx, mode=mode
            ).train(
                parameters, time_features=time_features,
                use_month_features=use_month_features
            )

        elif model_name == 'svm':
            parameters = {'kernel': 'rbf',
                          'degree': 3,
                          'gamma': 'auto',
                          'coef0': 0.0,
                          'tol': 0.001,
                          'C': 1.0,
                          'epsilon': 0.1,
                          'shrinking': True,
                          'cache_size': 200,
                          'verbose': False,
                          'max_iter': -1}
            if model_parameters:
                for model_param in model_parameters.keys():
                    parameters[model_param] = \
                        model_parameters[model_param]
            pred = SupportVectorMachine(
                self.x, self.y, self.pred_head,
                self.forecast_num, idx=idx, mode=mode
            ).train(
                parameters, time_features=time_features,
                use_month_features=use_month_features
            )

        elif model_name == 'dt':
            parameters = {'criterion': 'mae',
                          'splitter': 'best',
                          'max_depth': None,
                          'min_samples_split': 2,
                          'min_samples_leaf': 1,
                          'min_weight_fraction_leaf': 0.0,
                          'max_features': None,
                          'random_state': train_seed,
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0.0,
                          'min_impurity_split': None,
                          'presort': False}
            if model_parameters:
                for model_param in model_parameters.keys():
                    parameters[model_param] = \
                        model_parameters[model_param]
            pred = DecisionTree(
                self.x, self.y, self.pred_head,
                self.forecast_num, idx=idx, mode=mode
            ).train(
                parameters, time_features=time_features,
                use_month_features=use_month_features
            )

        elif model_name == 'rf':
            parameters = {'bootstrap': True,
                          'criterion': 'mae',
                          'max_depth': None,
                          'max_features': 'auto',
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0.0,
                          'min_samples_leaf': 1,
                          'min_samples_split': 2,
                          'min_weight_fraction_leaf': 0.0,
                          'n_estimators': 20,
                          'n_jobs': -1,
                          'oob_score': True,
                          'random_state': train_seed,
                          'verbose': 0,
                          'warm_start': False}
            if model_parameters:
                for model_param in model_parameters.keys():
                    parameters[model_param] = \
                        model_parameters[model_param]
            pred = RandomForest(
                self.x, self.y, self.pred_head,
                self.forecast_num, idx=idx, mode=mode
            ).train(
                parameters, time_features=time_features,
                use_month_features=use_month_features
            )

        elif model_name == 'et':
            parameters = {'n_estimators': 10,
                          'criterion': 'mae',
                          'max_depth': None,
                          'min_samples_split': 2,
                          'min_samples_leaf': 1,
                          'min_weight_fraction_leaf': 0.0,
                          'max_features': 'auto',
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0.0,
                          'min_impurity_split': None,
                          'bootstrap': False,
                          'oob_score': False,
                          'n_jobs': -1,
                          'random_state': train_seed,
                          'verbose': 0,
                          'warm_start': False}
            if model_parameters:
                for model_param in model_parameters.keys():
                    parameters[model_param] = \
                        model_parameters[model_param]
            pred = ExtraTrees(
                self.x, self.y, self.pred_head,
                self.forecast_num, idx=idx, mode=mode
            ).train(
                parameters, time_features=time_features,
                use_month_features=use_month_features
            )

        elif model_name == 'ab':
            et_parameters = {'n_estimators': 10,
                             'criterion': 'mae',
                             'max_depth': None,
                             'min_samples_split': 2,
                             'min_samples_leaf': 1,
                             'min_weight_fraction_leaf': 0.0,
                             'max_features': 'auto',
                             'max_leaf_nodes': None,
                             'min_impurity_decrease': 0.0,
                             'min_impurity_split': None,
                             'bootstrap': False,
                             'oob_score': False,
                             'n_jobs': -1,
                             'random_state': train_seed,
                             'verbose': 0,
                             'warm_start': False}
            reg_et = ExtraTreesRegressor(**et_parameters)
            parameters = {'loss': 'linear',
                          'base_estimator': reg_et,
                          'learning_rate': 1.0,
                          'n_estimators': 50,
                          'random_state': train_seed}
            if model_parameters:
                for model_param in model_parameters.keys():
                    parameters[model_param] = \
                        model_parameters[model_param]
            pred = AdaBoost(
                self.x, self.y, self.pred_head,
                self.forecast_num, idx=idx, mode=mode
            ).train(
                parameters, time_features=time_features,
                use_month_features=use_month_features
            )

        elif model_name == 'gb':
            parameters = {'loss': 'ls',
                          'learning_rate': 0.1,
                          'n_estimators': 100,
                          'subsample': 1.0,
                          'criterion': 'mae',
                          'min_samples_split': 2,
                          'min_samples_leaf': 1,
                          'min_weight_fraction_leaf': 0.0,
                          'max_depth': 3,
                          'min_impurity_decrease': 0.0,
                          'min_impurity_split': None,
                          'init': None,
                          'random_state': train_seed,
                          'max_features': None,
                          'alpha': 0.9,
                          'verbose': 0,
                          'max_leaf_nodes': None,
                          'warm_start': False,
                          'presort': 'auto'}
            if model_parameters:
                for model_param in model_parameters.keys():
                    parameters[model_param] = \
                        model_parameters[model_param]
            pred = GradientBoosting(
                self.x, self.y, self.pred_head,
                self.forecast_num, idx=idx, mode=mode
            ).train(
                parameters, time_features=time_features,
                use_month_features=use_month_features
            )

        elif model_name == 'xgb':
            parameters = {'max_depth': 3,
                          'learning_rate': 0.1,
                          'n_estimators': 100,
                          'silent': True,
                          'objective': 'reg:linear',
                          'booster': 'gbtree',
                          'n_jobs': -1,
                          'gamma': 0,
                          'min_child_weight': 1,
                          'max_delta_step': 0,
                          'subsample': 1,
                          'colsample_bytree': 1,
                          'colsample_bylevel': 1,
                          'reg_alpha': 0,
                          'reg_lambda': 1,
                          'scale_pos_weight': 1,
                          'base_score': 0.5,
                          'random_state': train_seed,
                          'seed': train_seed,
                          'missing': None,
                          'importance_type': 'gain'}
            if model_parameters:
                for model_param in model_parameters.keys():
                    parameters[model_param] = \
                        model_parameters[model_param]
            pred = XGBoost(
                self.x, self.y, self.pred_head,
                self.forecast_num, idx=idx, mode=mode
            ).train(
                parameters, time_features=time_features,
                early_stopping_rounds=early_stopping_rounds,
                use_month_features=use_month_features
            )

        elif model_name == 'lgb':
            parameters = {'boosting_type': 'gbdt',
                          'num_leaves': 31,
                          'max_depth': -1,
                          'learning_rate': 0.1,
                          'n_estimators': 100,
                          'subsample_for_bin': 200000,
                          'objective': 'regression',
                          'min_split_gain': 0.0,
                          'min_child_weight': 0.001,
                          'min_child_samples': 20,
                          'subsample': 1.0,
                          'subsample_freq': 0,
                          'colsample_bytree': 1.0,
                          'reg_alpha': 0.0,
                          'reg_lambda': 0.0,
                          'random_state': train_seed,
                          'n_jobs': -1,
                          'silent': True}
            if model_parameters:
                for model_param in model_parameters.keys():
                    parameters[model_param] = \
                        model_parameters[model_param]
            pred = LightGBM(
                self.x, self.y, self.pred_head,
                self.forecast_num, idx=idx, mode=mode
            ).train(
                parameters, time_features=time_features,
                early_stopping_rounds=early_stopping_rounds,
                use_month_features=use_month_features
            )

        else:
            raise Exception("Wrong Model Name!")

        return pred
