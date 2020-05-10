from models import utils
import pandas as pd
import numpy as np
from os.path import join
from models.time_series_models import TimeSeriesModel
from config import cfg


class Training(object):

    def __init__(self,
                 sample_mode,
                 select_col,
                 fill_mode=None):

        if fill_mode == 'no':
            fill_mode = None

        self.sample_mode = sample_mode
        self.select_col = select_col
        self.suffix = utils.get_suffix(sample_mode)
        self.fill_mode = fill_mode

        file_path = cfg.preprocessed_path
        if fill_mode:
            print('Loading {}...'.format(
                join(file_path, 'total_day_{}.csv'.format(fill_mode))))
            self.data = pd.read_csv(
                join(file_path, 'total_day_{}.csv'.format(fill_mode)),
                index_col='DATE',
                parse_dates=['DATE'])
        else:
            print('Loading {}...'.format(
                join(file_path, 'total_{}.csv'.format(sample_mode))))
            self.data = pd.read_csv(
                join(file_path, 'total_{}.csv'.format(sample_mode)),
                index_col='DATE',
                parse_dates=['DATE'])

    def _train_valid_split(self,
                           train_start,
                           valid_start,
                           valid_end):

        if valid_start:
            train_started = False
            valid_started = False
            valid_ended = False
            train_data = []
            valid_data = []
            final_data = []
            for row_iter in self.data.iterrows():
                row = row_iter[1]
                date_str = row_iter[0].to_pydatetime().strftime('%Y-%m-%d')
                if date_str == train_start:
                    train_started = True
                if train_started:
                    final_data.append(
                        row['{}_{}'.format(self.select_col, self.suffix)])
                if date_str == valid_start:
                    valid_started = True
                if train_started and not valid_started:
                    train_data.append(
                        row['{}_{}'.format(self.select_col, self.suffix)])
                if valid_started and not valid_ended:
                    valid_data.append(
                        row['{}_{}'.format(self.select_col, self.suffix)])
                if date_str == valid_end:
                    valid_ended = True
            return np.array(train_data), np.array(valid_data), \
                   np.array(final_data)
        else:
            train_started = False
            final_data = []
            for row_iter in self.data.iterrows():
                row = row_iter[1]
                date_str = row_iter[0].to_pydatetime().strftime('%Y-%m-%d')
                if date_str == train_start:
                    train_started = True
                if train_started:
                    final_data.append(
                        row['{}_{}'.format(self.select_col, self.suffix)])
            return np.array(final_data)

    @staticmethod
    def _calc_acc(y, pred):
        if pred.sum() < 1:
            return np.nan
        assert len(y) == len(pred), (len(y), len(pred))
        cost = []
        for y_i, pred_i in zip(y, pred):
            cost.append(abs(y_i - pred_i))
        return np.mean(cost)

    def save_result(self,
                    pred,
                    model_name,
                    append_info=None):

        utils.check_dir([cfg.result_path])
        df_pred = pd.read_csv(
            join(cfg.source_path, 'z_hack_submit_new.csv'),
            index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
        assert len(df_pred) == len(pred), (len(df_pred), len(pred))
        df_pred['FORECASTPRICE'] = pred
        if append_info is None:
            append_info = '_' + model_name + '_' + self.sample_mode
        df_pred.to_csv(
            join(cfg.result_path, 'result{}.csv'.format(append_info)),
            float_format='%.2f')

    def get_shifted_results(self,
                            model_name,
                            x_final,
                            freq,
                            seasonal,
                            append_info=None):

        # Get predictions for different lengths for different fill modes
        if self.fill_mode in ['w_ff', 'w_bf', 'w_avg', 'w_line']:
            pred_final_sf = TimeSeriesModel(
                x_final, freq, 22,
                seasonal=seasonal).predict(model_name)
            df_pred_sf = pd.read_csv(
                join(cfg.source_path, 'z_hack_submit_new_day_work.csv'),
                index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
        elif self.fill_mode in ['a_ff', 'a_bf', 'a_avg', 'a_line']:
            pred_final_sf = TimeSeriesModel(
                x_final, freq, 30,
                seasonal=seasonal).predict(model_name)
            df_pred_sf = pd.read_csv(
                join(cfg.source_path, 'z_hack_submit_new_day_all.csv'),
                index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
        else:
            raise ValueError('Wrong Fill Mode! Find: {}'.format(self.fill_mode))

        # Assign prediction to DataFrame
        assert len(df_pred_sf) == len(pred_final_sf), \
            (len(df_pred_sf), len(pred_final_sf))
        df_pred_sf['FORECASTPRICE'] = pred_final_sf

        # Reindex DataFrame to get final results
        df_pred = pd.read_csv(
            join(cfg.source_path, 'z_hack_submit_new.csv'),
            index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
        df_pred_sf = df_pred_sf.reindex(df_pred.index)

        # Save shifted results
        assert len(df_pred_sf) == len(df_pred), \
            (len(df_pred_sf), len(df_pred))
        utils.check_dir([cfg.result_path])
        shifted_result_path = join(cfg.result_path, 'shifted_results')
        utils.check_dir([shifted_result_path])
        df_pred_sf.to_csv(join(
            shifted_result_path, 'result_sf{}.csv'.format(append_info)),
            float_format='%.2f')


    def train(self,
              model_name,
              freq,
              forecast_num=21,
              seasonal='multiplicative',
              data_range=None,
              save_result=False,
              save_shifted_result=False,
              append_info=None):

        if data_range:
            if data_range['valid_start']:
                x_train, y, x_final = self._train_valid_split(
                    train_start=data_range['train_start'],
                    valid_start=data_range['valid_start'],
                    valid_end=data_range['valid_end'])
                # print(x_train, y, x_final)
                pred_valid = TimeSeriesModel(
                    x_train, freq, forecast_num=len(y),
                    seasonal=seasonal).predict(model_name)
                cost = self._calc_acc(y, pred_valid)
            else:
                x_final = self._train_valid_split(
                    train_start=data_range['train_start'],
                    valid_start=data_range['valid_start'],
                    valid_end=data_range['valid_end'])
                pred_valid = None
                cost = None
            pred_final = TimeSeriesModel(
                x_final, freq, forecast_num,
                seasonal=seasonal).predict(model_name)

            # Save shifted results for different fill modes
            if (self.fill_mode is not None) and save_shifted_result:
                self.get_shifted_results(
                    model_name, x_final,freq, seasonal, append_info)
        else:
            x_train = self.data[
                '{}_{}'.format(self.select_col, self.suffix)].values
            pred_final = TimeSeriesModel(
                x_train, freq, forecast_num,
                seasonal=seasonal).predict(model_name)
            pred_valid = None
            cost = None

        # Save final results
        if save_result:
            self.save_result(pred_final, model_name, append_info)

        return pred_final, cost, pred_valid


if __name__ == '__main__':

    utils.check_dir([cfg.log_path])

    df = pd.read_csv(join(cfg.source_path, 'z_hack_submit_new.csv'),
                     index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
    T = Training('day', 'CONTPRICE')

    range_1 = {'train_start': '2013-12-01',
               'valid_start': None,
               'valid_end': None}

    for f_mode in ['no', 'w_ff', 'w_avg', 'w_line']:
        df['arima_' + f_mode], _, _ = Training(
            'day', 'CONTPRICE', f_mode
        ).train(
            'arima', freq=5, forecast_num=21,
            data_range={'train_start': '2013-12-02',
                        'valid_start': None,
                        'valid_end': None},
            save_result=True,
            save_shifted_result=True,
            append_info='_arima_' + f_mode
        )

    for f_mode in ['a_ff', 'a_avg', 'a_line']:
        df['arima_' + f_mode], _, _ = Training(
            'day', 'CONTPRICE', f_mode
        ).train(
            'arima', freq=7, forecast_num=21,
            data_range={'train_start': '2013-12-01',
                        'valid_start': None,
                        'valid_end': None},
            save_result=True,
            save_shifted_result=True,
            append_info='_arima_' + f_mode
        )

    df.to_csv(join(cfg.log_path, 'result_day.csv'))
