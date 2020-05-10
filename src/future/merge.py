import pandas as pd
import numpy as np


def merge(name_rate_tuple, append_info=''):

    df_merge = pd.read_csv('./' + name_rate_tuple[0][0],
                           index_col=['FORECASTDATE'],
                           usecols=['FORECASTDATE'])
    pred_all = []
    for name_i, rate_i in name_rate_tuple:

        pred_i = pd.read_csv(
            './' + name_i, usecols=['FORECASTPRICE'])['FORECASTPRICE'].values
        pred_all.append(pred_i * rate_i)
    df_merge['FORECASTPRICE'] = np.sum(pred_all, axis=0)
    df_merge.to_csv('./merge{}.csv'.format(append_info), float_format='%.2f')

if __name__ == '__main__':

    merge_tuple = [('result_0.csv', 0.5),
                   ('result_1.csv', 0.5)]

    merge(merge_tuple, append_info='')
