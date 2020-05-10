import numpy as np
from rpy2 import robjects as ro


class RModelBase(object):

    def __init__(self, data_series, freq, forecast_num):
        ro.r('library(forecast)')

        self.data_series = data_series
        self.frequency = freq
        self.forecast_num = forecast_num

    def data_process(self):
        data_series = ro.IntVector(self.data_series)
        result = ro.r['ts'](data_series, frequency=self.frequency)
        return result


class Arima(RModelBase):

    def predict(self):
        data = self.data_process()
        try:
            model = ro.r['auto.arima'](data)
            result = ro.r.forecast(model, self.forecast_num)
            result_np = np.array(result.rx(4)).reshape(-1)
            return result, result_np
        except Exception as e:
            print('[Error in Arima] ', e)
            return None, np.zeros(self.forecast_num)


class STL(RModelBase):

    def predict(self):
        data = self.data_process()
        try:
            result = ro.r.stlf(data, self.forecast_num)
            result_np = np.array(result.rx(2)).reshape(-1)
            return result, result_np
        except Exception as e:
            print('[Error in STL] ', e)
            return None, np.zeros(self.forecast_num)


class ETS(RModelBase):

    def predict(self):
        data = self.data_process()
        try:
            model = ro.r.ets(data)
            result = ro.r.predict(model, h=self.forecast_num)
            result_np = np.array(result.rx(2)).reshape(-1)
            return result, result_np
        except Exception as e:
            print('[Error in ETS] ', e)
            return None, np.zeros(self.forecast_num)


class HoltWinters(RModelBase):

    def predict(self, seasonal='multiplicative'):
        data = self.data_process()
        try:
            model = ro.r.hw(data)
            result = ro.r.predict(
                model, h=self.forecast_num, seasonal=seasonal)
            result_np = np.array(result.rx(2)).reshape(-1)
            return result, result_np
        except Exception as e:
            print('[Error in HoltWinters] ', e)
            return None, np.zeros(self.forecast_num)


class TimeSeriesModel(object):

    def __init__(self, data, freq, forecast_num, seasonal=None):
        self.data = data
        self.freq = freq
        self.forecast_num = forecast_num
        self.seasonal = seasonal

    def predict(self, model_name):

        if model_name == 'arima':
            _, result_np = Arima(self.data, self.freq, self.forecast_num).predict()
        elif model_name == 'stl':
            _, result_np = STL(self.data, self.freq, self.forecast_num).predict()
        elif model_name == 'ets':
            _, result_np = ETS(self.data, self.freq, self.forecast_num).predict()
        elif model_name == 'hw':
            _, result_np = HoltWinters(
                self.data, self.freq, self.forecast_num).predict(self.seasonal)
        else:
            raise Exception("Wrong Model Name!")

        return result_np
