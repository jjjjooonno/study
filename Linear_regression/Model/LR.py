'''
Linear Regression
'''
import numpy as np
import pandas as pd


class GradientDescent():
    def __init__(self, learning_rate=0.01, threshold=0.01, max_iterations=1000):
        self._learning_rate = learning_rate
        self._threshold = threshold
        self._max_iterations = max_iterations
        self._W = None

    def fit(self, x_data, y_data):
        num_examples, num_features = np.shape(x_data)
        self._W = np.ones(num_features)
        x_data_transposed = x_data.transpose()

        for i in range(self._max_iterations):
            # 실제값과 예측값의 차이
            diff = np.dot(x_data, self._W) - y_data

            # diff를 이용하여 cost 생성 : 오차의 제곱합 / 2 * 데이터 개수
            cost = np.sum(diff ** 2) / (2 * num_examples)

            # transposed X * cost / n
            gradient = np.dot(x_data_transposed, diff) / num_examples

            # W벡터 업데이트
            self._W = self._W - self._learning_rate * gradient

            # 판정 임계값에 다다르면 학습 중단
            if cost < self._threshold:
                return self._W

        return self._W


class NotSatisfiedDataType(Exception):
    def __str__(self):
        return "Not satisfied data type Check your data type"


def check_df(x):
    if isinstance(x, pd.Series) | isinstance(x, pd.DataFrame) :
        x = x.values
    elif isinstance(x, np.ndarray):
        x = x
    else:
        raise NotSatisfiedDataType()
    return x


class LinearRegression:
    def __init__(self):
        self.bias = False
        self.coeff = None
        self.const = None

    def fit(self, X, Y, Gradient = False, learning_rate = None, threshold=None, max_iter = None):
        if Gradient:
            self.lr = learning_rate
            self.x = X
            self.y = Y
            X = check_df(X)
            Y = check_df(Y)
            if len(Y.shape) != 1:
                raise ValueError
            optimizer = GradientDescent(learning_rate = self.lr, threshold = threshold, max_iterations=max_iter)
            beta = optimizer.fit(X, Y)
            self.const = beta[0].tolist()
            self.coeff = beta[1:]
        else:
            self.x = X
            self.y = Y
            X = check_df(X)
            Y = check_df(Y)
            try:
                n_data, n_feature = X.shape
            except:
                n_data = X.shape[0]

            if len(Y.shape) != 1:
                raise ValueError
            X_ = np.c_[np.ones((n_data, 1)), X]
            beta = np.dot(np.dot(np.linalg.inv(np.dot(X_.transpose(), X_)), X_.transpose()), Y)
            self.const = beta[0].tolist()
            self.coeff = beta[1:]
        return self


    def predict(self, test_X):
        test_X = check_df(test_X)
        if len(test_X.shape) == 1:
            test_X = np.reshape(test_X,(test_X.shape[0], 1))
        self.predicted = np.dot(test_X, self.coeff) + self.const

        return self.predicted

    def summary(self):
        # R-square
        y_predicted = self.predicted
        target_mean = self.y.mean()
        target = self.y
        self.r2 = np.sum(np.power(y_predicted - target_mean, 2)) / np.sum(np.power(target - target_mean, 2))
        # R-square adjusted
        n_data, n_feature = self.x.shape
        r2 = 1 - self.r2
        self.adj_r2 = 1- (n_data * r2 - r2)/(n_data - n_feature - 1)
        # 설명변수 유의도

        return

