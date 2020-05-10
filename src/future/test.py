import time
from models import utils
from os.path import join
import pandas as pd

source_path = '../data/source_data'
preprocessed_path = '../data/preprocessed_data'


# Load CSV Files Using Pandas
def load_csv(file_path):

    f = pd.read_csv(file_path)

    return f

# df = load_csv(join(source_path, 'z_hack_transaction_2013.csv'))
# print(df.loc[(df['FDATE'] == '2013-01-05') & df['FIRMID'].isin(['118422837', '12987'])])


# for year in ['2009', '2010', '2011', '2012', '2013']:
#     # print('Preprocessing data in {}...'.format(year))
#     df_loaded = load_csv(
#         join(source_path, 'z_hack_transaction_{}.csv'.format(year)))
#
#     # df_dup = df_loaded.duplicated(keep=False)
#     # df_dup = df_loaded.loc[df_dup[df_dup == True].index]
#     # df_dup.to_csv(join(preprocessed_path, 'dup_{}.csv'.format(year)))
#
#     df_dup = df_loaded.duplicated()
#     print('{}: {} pairs'.format(year, len(df_dup[df_dup == True].index)))

print(utils.get_month_features())