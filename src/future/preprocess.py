import time
import pandas as pd
import numpy as np
from copy import deepcopy
from pandas import datetime as dt
from models import utils
from os.path import join
from config import cfg


class PreProcess(object):

    def __init__(self):
        pass

    @staticmethod
    def load_csv(file_path):
        df = pd.read_csv(file_path,
                              index_col='FDATE',
                              parse_dates=['FDATE'],
                              usecols=['FDATE', 'CONTNUM', 'CONTPRICE'])
        df['VOLUME'] = df['CONTNUM'] * df['CONTPRICE']
        return df

    @staticmethod
    def save_to_csv(df, sample_mode):
        utils.check_dir([cfg.preprocessed_path])
        df.to_csv(
            join(cfg.preprocessed_path, 'total_{}.csv'.format(sample_mode)))

    @staticmethod
    def preprocess(df, sample_mode):

        if sample_mode == 'day':
            df_grouped = df.groupby(df.index)
        elif sample_mode == 'month':
            df_grouped = df.groupby([lambda x: x.year, lambda x: x.month])
        elif sample_mode == 'year':
            df_grouped = df.groupby(lambda x: x.year)
        else:
            raise ValueError('Wrong Sample Mode Name!')

        suffix = utils.get_suffix(sample_mode)
        df_preprocessed = pd.DataFrame()
        df_preprocessed['CONTNUM_' + suffix] = df_grouped['CONTNUM'].sum()
        df_preprocessed['VOLUME_' + suffix] = df_grouped['VOLUME'].sum()
        df_preprocessed['CONTPRICE_' + suffix] = \
            df_preprocessed['VOLUME_' + suffix] / \
            df_preprocessed['CONTNUM_' + suffix]
        # df_preprocessed['CONTPRICE_' + suffix + '_AVG'] = \
        #     df_grouped['CONTPRICE'].mean()
        if sample_mode == 'day':
            df_preprocessed.index = df_preprocessed.index.rename('DATE')
        if sample_mode == 'month':
            df_preprocessed.index = df_preprocessed.index.rename(['YEAR', 'MONTH'])
        if sample_mode == 'year':
            df_preprocessed.index = df_preprocessed.index.rename('YEAR')

        return df_preprocessed

    @staticmethod
    def fill_nan(df, fill_mode):
        df_copy = deepcopy(df)
        for col_name in df.keys():
            nan_start_idx = None
            nan_rng = []
            values = df_copy[col_name].values
            for i, value in enumerate(values):
                if np.isnan(value) and not np.isnan(values[i + 1]) and i != 0:
                    nan_rng.append((nan_start_idx, i + 1))
                if i != len(values) - 1:
                    if not np.isnan(value) and np.isnan(values[i + 1]):
                        nan_start_idx = i

            for start, end in nan_rng:
                assert end - start >= 2
                if fill_mode == 'ff':
                    values[start + 1: end] = values[start]
                elif fill_mode == 'bf':
                    values[start + 1: end] = values[end]
                elif fill_mode == 'avg':
                    values[start + 1: end] = (values[start] + values[end]) / 2
                elif fill_mode == 'line':
                    n_rng = end - start
                    step = (values[end] - values[start]) / n_rng
                    values[start + 1: end] = \
                        [values[start] + step*(i+1) for i in range(n_rng-1)]
                else:
                    raise ValueError('Wrong Fill Mode!')
            df_copy[col_name] = values
        return df_copy

    def preprocess_years(self):

        df_day = []
        df_month = []
        df_year = []
        for year in ['2009', '2010', '2011', '2012', '2013']:
            print('Preprocessing data in {}...'.format(year))
            df_loaded = self.load_csv(
                join(cfg.source_path,
                     'z_hack_transaction_{}_new.csv'.format(year)))
            df_day.append(self.preprocess(df_loaded, 'day'))
            df_month.append(self.preprocess(df_loaded, 'month'))
            df_year.append(self.preprocess(df_loaded, 'year'))

        print('Concatenating data...')
        df_total_day: pd.DataFrame = pd.concat(df_day)
        df_total_month: pd.DataFrame = pd.concat(df_month)
        df_total_year: pd.DataFrame = pd.concat(df_year)

        print('Filling missing data...')
        start = dt(2009, 1, 4)
        end = dt(2013, 12, 31)

        df_day_work = df_total_day.reindex(
            pd.bdate_range(start, end).rename('DATE'))
        df_day_w_ff = self.fill_nan(df_day_work, 'ff')
        df_day_w_bf = self.fill_nan(df_day_work, 'bf')
        df_day_w_avg = self.fill_nan(df_day_work, 'avg')
        df_day_w_line = self.fill_nan(df_day_work, 'line')

        df_day_all = df_total_day.reindex(
            pd.date_range(start, end).rename('DATE'))
        df_day_a_ff = self.fill_nan(df_day_all, 'ff')
        df_day_a_bf = self.fill_nan(df_day_all, 'bf')
        df_day_a_avg = self.fill_nan(df_day_all, 'avg')
        df_day_a_line = self.fill_nan(df_day_all, 'line')

        print('Writing data to CSV...')
        self.save_to_csv(df_total_day, 'day')
        self.save_to_csv(df_total_month, 'month')
        self.save_to_csv(df_total_year, 'year')
        self.save_to_csv(df_day_work, 'day_work')
        self.save_to_csv(df_day_w_ff, 'day_w_ff')
        self.save_to_csv(df_day_w_bf, 'day_w_bf')
        self.save_to_csv(df_day_w_avg, 'day_w_avg')
        self.save_to_csv(df_day_w_line, 'day_w_line')
        self.save_to_csv(df_day_all, 'day_all')
        self.save_to_csv(df_day_a_ff, 'day_a_ff')
        self.save_to_csv(df_day_a_bf, 'day_a_bf')
        self.save_to_csv(df_day_a_avg, 'day_a_avg')
        self.save_to_csv(df_day_a_line, 'day_a_line')


if __name__ == '__main__':

    time_ = time.time()
    utils.thick_line()
    print('Start preprocessing...')

    PreProcess().preprocess_years()

    utils.thin_line()
    print('Finished! Using time: {:.2f}s'.format(time.time() - time_))
    utils.thick_line()
