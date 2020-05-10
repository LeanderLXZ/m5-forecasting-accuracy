import pandas as pd

def create_features(series, dates):
    series_t = series.transpose()
    df_products = pd.merge(
        series_t, dates, left_index= True, right_index = True)
    df_products['Date'] = pd.to_datetime(df_products['Date'])
    df_products['quarter'] = df_products['Date'].dt.quarter.astype('uint8')
    df_products['Month'] = df_products['Date'].dt.month.astype('uint8')
    df_products['Year'] = df_products['Date'].dt.year.astype('uint8')
    df_products['dayofyear'] = df_products['Date'].dt.dayofyear.astype('uint8')
    df_products['dayofweek'] = df_products['Date'].dt.dayofweek.astype('uint8')
    df_products.index = df_products.Date
    df_products = df_products.drop(['Date', 'weekday', 'month', 'd'], axis= 1)
    return df_products

def create_series(data, df_sales):
    a = data[0]
    df = df_sales.copy()
    first_date = 'd_1'
    last_date = 'd_1969'
    if a:
        final_df = df.groupby(data).sum()
        lnn = list()
        try:
            for i in final_df.index:
                nn = '_'.join([i[0], i[1]])
                lnn.append(nn)
                final_df['final_name'] = lnn
                final_df.index = final_df['final_name']
                final_df = final_df.drop('final_name', axis =1)
        except:
            pass
        return final_df
    else:
        df = df.loc[:,first_date:last_date]
        final_df = pd.Series(df.sum(axis = 0))
        return final_df
