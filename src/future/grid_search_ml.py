import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
from models import utils
from main_ml import Training
from config import cfg


class GridSearch(object):

    def __init__(self,
                 sample_mode,
                 select_col):

        self.T = {'no': Training(sample_mode, select_col),
                  'w_ff': Training(sample_mode, select_col, 'w_ff'),
                  'w_bf': Training(sample_mode, select_col, 'w_bf'),
                  'w_avg': Training(sample_mode, select_col, 'w_avg'),
                  'w_line': Training(sample_mode, select_col, 'w_line'),
                  'a_ff': Training(sample_mode, select_col, 'a_ff'),
                  'a_bf': Training(sample_mode, select_col, 'a_bf'),
                  'a_avg': Training(sample_mode, select_col, 'a_avg'),
                  'a_line': Training(sample_mode, select_col, 'a_line')}
        self.sample_mode = sample_mode

    @staticmethod
    def _generate_grid_combinations(param_grid):

        n_param = len(param_grid)
        n_value = 1
        param_name = []
        for i in range(n_param):
            param_name.append(param_grid[i][0])
            n_value *= len(param_grid[i][1])

        param_value: np.ndarray = np.zeros((n_param, n_value)).tolist()
        global value_list
        global value_col
        value_list = []
        value_col = 0

        def generate_value_matrix_(idx_param):
            idx_param_next = idx_param + 1
            for value in param_grid[idx_param][1]:
                global value_list
                value_list.append(value)
                if idx_param_next < n_param:
                    generate_value_matrix_(idx_param_next)
                else:
                    global value_col
                    for i_row, row in enumerate(param_value):
                        row[value_col] = value_list[i_row]
                    value_col += 1
                value_list.pop()
        generate_value_matrix_(0)

        grid_combs = []
        for i_param_value in range(n_value):
            grid_search_tuple_dict = {}
            for i_param in range(n_param):
                param_name_i = param_name[i_param]
                param_value_i = param_value[i_param][i_param_value]
                grid_search_tuple_dict[param_name_i] = param_value_i
            grid_combs.append(grid_search_tuple_dict)

        return grid_combs

    def grid_search(self,
                    param_grid,
                    save_every_result=False,
                    save_shifted_result=False,
                    append_info=None):

        start_time = time.time()

        df_total = pd.read_csv(
            join(cfg.source_path, 'z_hack_submit_new_with_cost.csv'),
            index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
        forecast_num = len(df_total) - 1

        df_valid = pd.DataFrame(index=range(35))

        idx = 0

        for i, grid_i in enumerate(param_grid):

            task_time = time.time()

            utils.thick_line()
            print('Grid Searching Task {}...'.format(i))

            grid_combs = self._generate_grid_combinations(grid_i)

            for grid_search_tuple_dict in tqdm(grid_combs,
                                               total=len(grid_combs),
                                               ncols=100,
                                               unit=' comb'):
                model_name = grid_search_tuple_dict['model_name']
                start_year = grid_search_tuple_dict['start_year']
                valid_range = grid_search_tuple_dict['valid_range']
                feature_num = grid_search_tuple_dict['feature_num']
                fill_mode = grid_search_tuple_dict['fill_mode']
                time_features = grid_search_tuple_dict['time_features']
                use_month_features = \
                    grid_search_tuple_dict['use_month_features']
                train_start = {2009: '2009-01-05',
                               2010: '2010-01-04',
                               2011: '2011-01-04',
                               2012: '2010-01-04',
                               2013: '2010-01-04'}
                data_range = {'train_start': train_start[start_year],
                              'valid_start': valid_range[0],
                              'valid_end': valid_range[1]}

                if append_info is None:
                    append_info = ''

                pred_final, cost, pred_valid = self.T[fill_mode].train(
                    model_name=model_name,
                    feature_num=feature_num,
                    forecast_num=forecast_num,
                    time_features=time_features,
                    use_month_features=use_month_features,
                    data_range=data_range,
                    save_result=save_every_result,
                    save_shifted_result=save_shifted_result,
                    append_info='_' + str(idx) + append_info,
                    idx=idx
                )

                utils.save_log_to_csv(
                    log_path=cfg.log_path,
                    grid_search_tuple_dict=grid_search_tuple_dict,
                    cost=cost,
                    idx=idx,
                    append_info='_' + self.sample_mode + append_info)

                pred_final = np.append(pred_final, cost)
                df_total[str(idx)] = pred_final

                if len(pred_valid) < 34:
                    pred_valid = \
                        np.append(pred_valid, np.zeros(34 - len(pred_valid)))
                pred_cost_valid = np.append(pred_valid, cost)
                df_valid[str(idx)] = pred_cost_valid

                idx += 1

            utils.thin_line()
            print('Task {} Done! Using {:.2f}s...'.format(
                i, time.time() - task_time))
            utils.thick_line()

        utils.check_dir([cfg.result_path])
        df_total = df_total.stack().unstack(0)
        df_total.to_csv(join(
            cfg.log_path,
            'all_results_{}_{}.csv'.format(
                self.sample_mode, append_info)))
        df_valid.to_csv(join(
            cfg.log_path,
            'all_valid_{}_{}.csv'.format(
                self.sample_mode, append_info)))

        utils.thick_line()
        print('All Task Done! Using {:.2f}s...'.format(
            time.time() - start_time))
        utils.thick_line()

    def grid_search_model_params(self,
                                 param_grid,
                                 save_every_result=False,
                                 save_shifted_result=False,
                                 append_info=None):

        start_time = time.time()

        df_total = pd.read_csv(
            join(cfg.source_path, 'z_hack_submit_new_with_cost.csv'),
            index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
        df_total_sf = df_total.copy()
        forecast_num = len(df_total) - 1

        df_valid = pd.DataFrame(index=range(35))

        idx = 0

        for i, grid_i in enumerate(param_grid):

            task_time = time.time()

            utils.thick_line()
            print('Grid Searching Task {}...'.format(i))

            grid_combs = self._generate_grid_combinations(grid_i)

            for grid_search_tuple_dict in tqdm(grid_combs,
                                               total=len(grid_combs),
                                               ncols=100,
                                               unit=' comb'):
                grid_search_tuple_dict_copy = grid_search_tuple_dict.copy()
                model_name = grid_search_tuple_dict.pop('model_name')
                start_year = grid_search_tuple_dict.pop('start_year')
                valid_range = grid_search_tuple_dict.pop('valid_range')
                feature_num = grid_search_tuple_dict.pop('feature_num')
                fill_mode = grid_search_tuple_dict.pop('fill_mode')
                time_features = grid_search_tuple_dict.pop('time_features')
                use_month_features = \
                    grid_search_tuple_dict.pop('use_month_features')
                train_start = {2009: '2009-01-05',
                               2010: '2010-01-04',
                               2011: '2011-01-04',
                               2012: '2010-01-04',
                               2013: '2010-01-04'}
                data_range = {'train_start': train_start[start_year],
                              'valid_start': valid_range[0],
                              'valid_end': valid_range[1]}
                if append_info is None:
                    append_info = ''

                model_parameters = {}
                for model_param in grid_search_tuple_dict:
                    model_parameters[model_param] = \
                        grid_search_tuple_dict[model_param]

                pred_final, pred_final_sf, cost, pred_valid = self.T[fill_mode].train(
                    model_name=model_name,
                    model_parameters=model_parameters,
                    feature_num=feature_num,
                    forecast_num=forecast_num,
                    time_features=time_features,
                    use_month_features=use_month_features,
                    data_range=data_range,
                    save_result=save_every_result,
                    save_shifted_result=save_shifted_result,
                    append_info='_' + str(idx) + append_info,
                    idx=idx)

                utils.save_log_to_csv(
                    log_path=cfg.log_path,
                    grid_search_tuple_dict=grid_search_tuple_dict_copy,
                    cost=cost,
                    idx=idx,
                    append_info='_' + self.sample_mode + append_info)

                pred_final = np.append(pred_final, cost)
                df_total[str(idx)] = pred_final

                if (fill_mode != 'no') and save_shifted_result:
                    pred_final_sf = np.append(pred_final_sf, cost)
                    df_total_sf[str(idx)] = pred_final_sf

                if len(pred_valid) < 34:
                    pred_valid = \
                        np.append(pred_valid, np.zeros(34 - len(pred_valid)))
                pred_cost_valid = np.append(pred_valid, cost)
                df_valid[str(idx)] = pred_cost_valid

                idx += 1

            utils.thin_line()
            print('Task {} Done! Using {:.2f}s...'.format(
                i, time.time() - task_time))
            utils.thick_line()

        utils.check_dir([cfg.result_path])
        df_total = df_total.stack().unstack(0)
        df_total.to_csv(join(
            cfg.log_path,
            'all_results_{}{}.csv'.format(
                self.sample_mode, append_info)))
        df_valid.to_csv(join(
            cfg.log_path,
            'all_valid_{}{}.csv'.format(
                self.sample_mode, append_info)))

        if save_shifted_result:
            df_total_sf = df_total_sf.stack().unstack(0)
            df_total_sf.to_csv(join(
                cfg.log_path,
                'all_results_sf_{}{}.csv'.format(
                    self.sample_mode, append_info)))

        utils.thick_line()
        print('All Task Done! Using {:.2f}s...'.format(
            time.time() - start_time))
        utils.thick_line()


if __name__ == '__main__':

    GS = GridSearch(sample_mode='day', select_col='CONTPRICE')

    # parameter_grid = [
    #     [['fill_mode', ['no', 'w_ff', 'w_avg', 'w_line',
    #                     'a_ff', 'a_avg', 'a_line']],
    #      ['model_name', ['knn', 'svm', 'dt', 'rf', 'et',
    #                      # 'ab', 'gb',
    #                      'xgb', 'lgb']],
    #      ['start_year', [2009, 2010, 2011, 2012]],
    #      ['valid_range', [('2013-12-02', '2013-12-31'),
    #                       ('2013-01-04', '2013-01-31')]],
    #      ['feature_num', [21, 30, 50, 100]],
    #      ['time_features',  [(10, 20, 30, 40, 50)]],
    #      ['use_month_features', [False, True]]
    #      ]
    # ]
    # GS.grid_search(parameter_grid,
    #                save_every_result=True,
    #                save_shifted_result=True,
    #                append_info='_ml')


    # parameter_grid = [
    #     [
    #         ['fill_mode', ['a_avg']],
    #         ['model_name', ['xgb']],
    #         ['start_year', [2009,]],
    #         ['valid_range', [('2013-12-02', '2013-12-31')]],
    #         ['feature_num', [21]],
    #         ['time_features', [(5, 10, 20, 30, 60)]],
    #         ['use_month_features', [True]],
    #
    #         ['learning_rate', [0.01]],
    #         ['n_estimators', [100, 1000]],
    #         ['max_depth', [8]],
    #         ['min_child_weight', [3]],
    #         ['subsample', [0.90]],
    #         ['colsample_bytree', [0.9]],
    #         ['colsample_bylevel', [0.8]],
    #         ['early_stopping_rounds', [None]]
    #     ]
    # ]

    parameter_grid = [
        [
            ['fill_mode', ['no']],
            ['model_name', ['xgb']],
            ['start_year', [2009, 2011, 2012]],
            ['valid_range', [('2013-12-02', '2013-12-31')]],
            ['feature_num', [21, 30, 60]],
            ['time_features', [(5, 10, 20, 30, 60)]],
            ['use_month_features', [True]],

            ['learning_rate', [0.05, 0.1, 0.3]],
            ['n_estimators', [100, 500]],
            ['max_depth', [4, 6, 8]],
            ['min_child_weight', [1, 3, 5]],
            ['subsample', [0.85, 0.90, 0.95]],
            ['colsample_bytree', [0.85, 0.9, 0.95]],
            ['colsample_bylevel', [0.7, 0.8, 0.9]],
            ['early_stopping_rounds', [None]]
        ],
        [
            ['fill_mode', ['a_avg', 'a_line']],
            ['model_name', ['xgb']],
            ['start_year', [2009, 2011, 2012]],
            ['valid_range', [('2013-12-01', '2013-12-31')]],
            ['feature_num', [21, 30, 60]],
            ['time_features', [(5, 10, 20, 30, 60)]],
            ['use_month_features', [True]],

            ['learning_rate', [0.05, 0.1, 0.3]],
            ['n_estimators', [100, 500]],
            ['max_depth', [4, 6, 8]],
            ['min_child_weight', [1, 3, 5]],
            ['subsample', [0.85, 0.90, 0.95]],
            ['colsample_bytree', [0.85, 0.9, 0.95]],
            ['colsample_bylevel', [0.7, 0.8, 0.9]],
            ['early_stopping_rounds', [None]]
        ]
    ]

    GS.grid_search_model_params(parameter_grid,
                                save_every_result=True,
                                save_shifted_result=True,
                                append_info='_xgb')
