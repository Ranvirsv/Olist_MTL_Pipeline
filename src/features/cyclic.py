import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CyclicEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            dt_series = pd.to_datetime(X.iloc[:, 0])
        else:
            dt_series = pd.to_datetime(X[:, 0])

        hour = dt_series.dt.hour
        month = dt_series.dt.month
        day_of_week = dt_series.dt.dayofweek

        HOUR_MAX = 24
        MONTH_MAX = 12
        DAY_OF_WEEK_MAX = 7

        hour_sin = np.sin(2 * np.pi * hour / HOUR_MAX)
        hour_cos = np.cos(2 * np.pi * hour / HOUR_MAX)
        month_sin = np.sin(2 * np.pi * month / MONTH_MAX)
        month_cos = np.cos(2 * np.pi * month / MONTH_MAX)
        day_of_week_sin = np.sin(2 * np.pi * day_of_week / DAY_OF_WEEK_MAX)
        day_of_week_cos = np.cos(2 * np.pi * day_of_week / DAY_OF_WEEK_MAX)

        return np.column_stack((
            hour_sin, hour_cos, 
            month_sin, month_cos, 
            day_of_week_sin, day_of_week_cos
        ))