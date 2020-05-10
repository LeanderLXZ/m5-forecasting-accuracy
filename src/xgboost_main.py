import os
import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import plotly.express as px
import plotly.graph_objects as go

import xgboost as xgb
from xgboost import plot_importance, plot_tree

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


def train(df_product, params=None):
    error = list()
    y = df_product.iloc[:,0]
    split_date = df_product.index[-60]
    df_train = df_product.loc[df_product.index < split_date].copy()
    df_valid = df_product.loc[df_product.index >= split_date].copy()
    X_train, y_train = features_(df_train, 
                                 train=True, 
                                 label=df_product.columns[0])
    X_valid, y_valid = features_(df_valid, 
                               train=False, 
                               label=df_product.columns[0])

    model = xgb.XGBRegressor(n_estimators=250, n_jobs=-1)
    
    model.fit(X_train, 
              y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              early_stopping_rounds=10,
              verbose=True)

    _ = plot_importance(model, height=1)
    
    df_valid['PREDICTION'] = model.predict(X_valid)

    # train_error = eval_func(y_train, model.predict(X_train))
    train_error = math.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    print('-' * 100)
    print('Train Error: %.4f RMSE' % (train_error))

    # valid_error = eval_func(y_valid, df_valid['PREDICTION'])
    valid_error =  math.sqrt(mean_squared_error(y_valid, df_valid['PREDICTION']))
    print('Valid Error: %.4f RMSE' % (valid_error))
    print('-' * 100)
    
    df_final = pd.concat([df_train, df_valid], sort = False)
        
    return df_final

for i in series.index:
    df_series = create_features(series.loc[i,:], dates)

    pred_y = train(df_series)

    n_days = 28
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_series.index[-n_days:], 
        y=df_series.iloc[-n_days:, 0], 
        name="y_true", 
        line_color='gray'))
    fig.add_trace(go.Scatter(
        x=df_series.index[-n_days:], 
        y=pred_y.PREDICTION[-n_days:], 
        name="y_pred", 
        line_color='red'))
    fig.show()


global cv_results
cv_results = None

def train_grid_search(df_product, params=None):
    error = list()
    y = df_product.iloc[:,0]
    split_date = df_product.index[-60]
    df_train = df_product.loc[df_product.index < split_date].copy()
    df_valid = df_product.loc[df_product.index >= split_date].copy()
    X_train, y_train = features_(df_train, 
                                 train=True, 
                                 label=df_product.columns[0])
    X_valid, y_valid = features_(df_valid, 
                               train=False, 
                               label=df_product.columns[0])

    model = GridSearchCV(estimator=xgb.XGBRegressor(),
                         param_grid=params,
                         cv = 10,
                         n_jobs=-1,
                         verbose=False)

    model.fit(X_train, 
              y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              early_stopping_rounds=10,
              verbose=False)
    global cv_results 
    cv_results = pd.DataFrame(model.cv_results_)
    print(model.best_params_)
    print(model.best_score_)

    model = xgb.XGBRegressor(**model.best_params_)
    model.fit(X_train, 
              y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              early_stopping_rounds=10,
              verbose=True)

    _ = plot_importance(model, height=1)
    
    df_valid['PREDICTION'] = model.predict(X_valid)

    # train_error = eval_func(y_train, model.predict(X_train))
    train_error = math.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    print('-' * 100)
    print('Train Error: %.4f RMSE' % (train_error))

    # valid_error = eval_func(y_valid, df_valid['PREDICTION'])
    valid_error =  math.sqrt(mean_squared_error(y_valid, df_valid['PREDICTION']))
    print('Valid Error: %.4f RMSE' % (valid_error))
    print('-' * 100)

    df_final = pd.concat([df_train, df_valid], sort = False)
        
    return df_final


results = []

params = {
    "learning_rate" : [0.01, 0.1, 0.5],
    "max_depth" : [3, 5, 8],
    "n_estimators": [100, 200, 500]
    }

for i in series.index:
    df_series = create_features(series.loc[i,:], dates)

    pred_y = train_grid_search(df_series, params=params)

    n_days = 28

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_series.index[-n_days:], 
        y=df_series.iloc[-n_days:, 0], 
        name="y_true", 
        line_color='gray'))
    fig.add_trace(go.Scatter(
        x=df_series.index[-n_days:], 
        y=pred_y.PREDICTION[-n_days:], 
        name="y_pred", 
        line_color='red'))
    fig.show()

cv_results.to_csv('sv_results.csv')
