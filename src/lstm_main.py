import os
import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import plotly.express as px
import plotly.graph_objects as go

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from utils import *
from eval_func import *


df_sales = pd.read_csv('../data/sales_train_validation.csv', 
                       index_col='item_id')
df_prices = pd.read_csv('../data/sell_prices.csv')
df_calendar = pd.read_csv('../data/calendar.csv', 
                          index_col = 'date')
first_date = 'd_1'
last_date = 'd_1913'

dates = df_calendar.drop(['wm_yr_wk', 'wday'], axis = 1)
dates['Date'] = dates.index
dates.index = dates['d']
dates = dates.fillna(0)

my_labeler = LabelEncoder()
for i in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
    dates[i] = my_labeler.fit_transform(dates[i].astype('str'))

df_prices_stats = df_prices.loc[:,['item_id', 'sell_price']]
df_prices_stats = df_prices_stats.groupby(
    'item_id').sell_price.agg([min, max, 'mean'])
df_estados = df_sales.loc[:,'state_id':last_date]
df_estados = df_estados.groupby('state_id').sum()
df_estados_Q = pd.DataFrame(df_estados.sum(axis=1))

df_estados = df_estados.transpose()
df_estados= pd.merge(df_estados, dates, left_index= True, right_index = True)
df_estados.Date = pd.to_datetime(df_estados.Date)

df_sales_tot_ = pd.DataFrame(df_sales.loc[:, first_date:last_date].sum(axis=1))
df_sales_tot_a = df_sales_tot_.groupby(df_sales_tot_.index).sum()
df_sales_tot = pd.merge(df_sales_tot_a, 
                        df_prices_stats, 
                        right_index = True,
                         left_index=True)
df_sales_tot['Total'] = df_sales_tot.iloc[:,0]*df_sales_tot.loc[:,'mean']
df_sales_tot = df_sales_tot.Total.sort_values()


def features_(df, train = True, label = None):

    cols_to_remove = ['year',label]
    X = df.drop(cols_to_remove, axis = 1)

    if label:
        y = df[label]
        return X, y
    else:
        return X


s1 = [[None],
      ["state_id"],
      ["store_id"],
      ["cat_id"],
      ["dept_id"],
      ["state_id", "cat_id"],
      ["state_id", "dept_id"],
      ["store_id", "cat_id"],
      ["store_id", "dept_id"],
      ["item_id"],
      ["item_id", "state_id"],
      ["item_id", "store_id"]]

series = create_series(s1[1], df_sales)


def train(serie, split=100, lb=28):
    
    np.random.seed(36)
    
    scaler = MinMaxScaler(feature_range = (0,1))
    df = scaler.fit_transform(serie)
    train_size = int(len(df)-split)
    valid_size = len(df) - train_size
    train, valid = df[0:train_size,:], df[train_size:len(df), :]

    def create_dataset(dataset, look_back = 28):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i+look_back,0])
        return np.array(dataX), np.array(dataY)
    
    x_train,Y_train = create_dataset(train, lb)
    x_valid, Y_valid = create_dataset(valid, lb)

    # reshape input to be [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], 1, x_valid.shape[1]))
    
    # create and fit the lstm network
    model = Sequential()
    model.add(LSTM(4, input_shape = (1, lb)))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    model.fit(x_train, Y_train, epochs = 10, batch_size = 1, verbose = False)
    
    # make predictions
    pred_train = model.predict(x_train)
    pred_valid = model.predict(x_valid)
    
    #invert predictions
    pred_train = scaler.inverse_transform(pred_train)
    Y_train = scaler.inverse_transform([Y_train])
    pred_valid = scaler.inverse_transform(pred_valid)
    Y_valid = scaler.inverse_transform([Y_valid])
    
    # calculate root mean squared error
    print('-' * 100)
    train_error = math.sqrt(mean_squared_error(Y_train[0], pred_train[:,0]))
    print('Train Error: %.2f RMSE' % (train_error))
    valid_error = math.sqrt(mean_squared_error(Y_valid[0], pred_valid[:,0]))
    print('Valid Error: %.2f RMSE' % (valid_error))
    print('-' * 100)
    
    pred_train_plot = np.empty_like(df)
    pred_train_plot[:, :] = np.nan
    pred_train_plot[lb:len(pred_train)+lb, :] = pred_train
    pred_valid_plot = np.empty_like(df)
    pred_valid_plot[:, :] = np.nan
    pred_valid_plot[len(df)-len(pred_valid):, :] = pred_valid
    pred_valid_plot = pd.DataFrame(pred_valid_plot)
        
    return pred_train_plot, pred_valid_plot


for i in series.index:
    df_series = create_features(series.loc[i,:], dates)

    train_pred, valid_pred = train(pd.DataFrame(df_series.iloc[:,0]))

    n_days = 28

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_series.index[-n_days:], 
        y=df_series.iloc[-n_days:, 0], 
        name="y_true", 
        line_color='gray'))
    fig.add_trace(go.Scatter(
        x=df_series.index[-n_days:], 
        y=valid_pred.iloc[-n_days:,0], 
        name="y_pred",
        line_color='red'))
    fig.show()
