'''
The collection of example simulated functions used in the paper.
'''
import numpy as np
from scipy.stats import t as t_dist, norm, cauchy, laplace
from scipy.stats import multivariate_t

class Benchmark:
    def __init__(self):
        pass

    def noiseless(self, X):
        raise NotImplementedError

    def quantile(self, X, q):
        raise NotImplementedError

    def sample(self, X):
        raise NotImplementedError


class Scenario1(Benchmark):
    def __init__(self):
        super().__init__()
        self.n_in = 5
        self.beta = np.array([0.5, 1.2, -0.8])

    def noiseless(self, X):
        # 包含随机森林擅长的阈值效应
        threshold_effect = np.where(X[:, 0] > 0.6, 2 * np.sqrt(X[:, 1]), -X[:, 1] ** 2)

        # 包含神经网络擅长的非线性变换
        nn_effect = 0.5 * np.sin(3 * np.pi * X[:, 2]) * np.exp(X[:, 0])

        return threshold_effect + nn_effect

    def quantile(self, X, q):
        # 异方差噪声（与特征相关）
        scale =1
        #scale = 0.3 + 0.5 * np.abs(X[:, 0] - X[:, 1])
        return self.noiseless(X) + scale * t_dist.ppf(q, 3)

    def sample(self, X):
        scale = 1
        # scale = 0.3 + 0.5 * np.abs(X[:, 0] - X[:, 1])
        return self.noiseless(X) + scale * np.random.standard_t(3, X.shape[0])

class Scenario2(Benchmark):
    def __init__(self):
        super().__init__()
        self.n_in = 3

    def noiseless(self, X):
        return self.g2(self.g1(X))

    def quantile(self, X, q):
        return self.noiseless(X) + self.g3(X) * t_dist.ppf(q, 2)

    def sample(self, X):
        return self.noiseless(X) + self.g3(X) * np.random.standard_t(2, size=X.shape[0])

    def g1(self, X):
        return np.array([np.sqrt(X[:,0]) + X[:,0]*X[:,1], np.cos(2*np.pi*X[:,1])]).T

    def g2(self, X):
        return np.sqrt(X[:,0] + X[:,1]**2) + X[:,0]**2 * X[:,1]

    def g3(self, X):
        return np.linalg.norm(X - 0.5, axis=1)

class Scenario3(Benchmark):
    def __init__(self):
        super().__init__()
        self.n_in = 3
        self.beta = np.array([1.5, -0.7, 0.3])

    def noiseless(self, X):
        y1 = np.where(X[:, 0] > X[:, 1],
                      np.log(1 + X[:, 0] - X[:, 1]),
                      np.exp(X[:, 1] - X[:, 0]) - 1)

        y2 = 0.5 * np.sin(2 * np.pi * (X[:, 2] + X[:, 0])) * (X[:, 0] + X[:, 1])

        y3=y1+y2
        y4=y1-y2

        return np.column_stack([y3, y4])

    def quantile(self, X, q):
        return self.noiseless(X) + t_dist.ppf(q, 3)

    def sample(self, X):
        return self.noiseless(X) + np.random.standard_t(3, size=(X.shape[0], 2))

def generate_correlated_t_noise(n_samples, dim, df=3, rho=0.7):
    cov = np.full((dim, dim), rho)
    np.fill_diagonal(cov, 1.0)

    noise = multivariate_t.rvs(loc=np.zeros(dim), shape=cov, df=df, size=n_samples)
    return noise


class Scenario3(Benchmark):
    def __init__(self):
        # super().__init__()
        self.n_in = 3
        self.rho = 0.7

    def noiseless(self, X):

        y1 = np.where(X[:, 0] > X[:, 1],
                      np.log(1 + np.abs(X[:, 0] - X[:, 1])),  # 加 abs 防止 log 负数
                      np.exp(X[:, 1] - X[:, 0]) - 1)
        y2 = 0.5 * np.sin(2 * np.pi * (X[:, 2] + X[:, 0])) * (X[:, 0] + X[:, 1])

        # 混合输出
        y3 = y1 + y2
        y4 = y1 - y2
        return np.column_stack([y3, y4])

    def quantile(self, X, q):
        return self.noiseless(X) + t_dist.ppf(q, df=3)

    def sample(self, X):
        noise = generate_correlated_t_noise(X.shape[0], dim=2, df=3, rho=self.rho)
        return self.noiseless(X) + noise


class Scenario4(Benchmark):
    def __init__(self):
        # super().__init__()
        self.n_in = 2
        self.rho = 0.7

    def g1(self, X):
        return np.array([np.abs(X[:, 0]), np.prod(X, axis=1)]).T

    def g2(self, X):
        return np.array([np.sqrt(np.abs(X[:, 0]) + X[:, 1] ** 2), X.sum(axis=1) ** 3]).T

    def noiseless(self, X):
        return self.g2(self.g1(X))

    def quantile(self, X, q):
        return self.noiseless(X) + t_dist.ppf(q, df=3)

    def sample(self, X):
        noise = generate_correlated_t_noise(X.shape[0], dim=2, df=3, rho=self.rho)
        return self.noiseless(X) + noise