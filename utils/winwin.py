import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
class Winsorizer(TransformerMixin, BaseEstimator):
    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, y, **kwargs):
        low, high = np.quantile(y, [self.lower_q, self.upper_q])
        self.low_  = low
        self.high_ = high
        return self

    def transform(self, y):
        return np.clip(y, self.low_, self.high_)

    def inverse_transform(self, y):
        return y