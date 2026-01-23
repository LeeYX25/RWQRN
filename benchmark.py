'''
Runs the quantile regression benchmarks .
'''
import numpy as np


from funcs import Scenario1, Scenario2, Scenario3, Scenario4
from RWQRN_model import QuantileNetwork



def run_benchmarks(demo=True):
    N_test = 1000
    sample_sizes = 100
    quantiles = np.array([0.1,0.2])
    func = Scenario1()
    tau=0.01
    model = QuantileNetwork(quantiles=quantiles)
    X_test = np.random.random(size=(N_test,func.n_in))
    y_test = func.sample(X_test)
    y_sample = np.array([func.sample(X_test)])

    # Get the ground truth quantiles
    y_quantiles = np.array([func.quantile(X_test, q) for q in quantiles]).T
    X_train = np.random.random(size=(sample_sizes,func.n_in))
    y_train = func.sample(X_train)
    model.fit(X_train, y_train,tau)
    preds = model.predict(X_test)
    mse = ((y_quantiles - preds)**2).mean(axis=0)
    print('\t',mse)

def run_multivariate_benchmarks(demo=True):
    N_test = 1000
    sample_sizes = 100
    quantiles = np.array([0.1,0.2])
    func = Scenario3()
    tau=0.01
    model = QuantileNetwork(quantiles=quantiles,loss="geometric")#or loss="geometric" to use geometric loss
    X_test = np.random.random(size=(N_test,func.n_in))
    y_test = func.sample(X_test)
    y_sample = np.array([func.sample(X_test)])

    # Get the ground truth quantiles
    y_quantiles = np.transpose(np.array([func.quantile(X_test, q) for q in quantiles]), [1, 2, 0])
    X_train = np.random.random(size=(sample_sizes,func.n_in))
    y_train = func.sample(X_train)
    model.fit(X_train, y_train,tau)
    preds = model.predict(X_test)
    mse = ((y_quantiles - preds)**2).mean(axis=0)
    print('\t',mse)



if __name__ == '__main__':
    np.random.seed(42)
    import torch
    torch.manual_seed(42)

    np.set_printoptions(precision=2, suppress=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #run_benchmarks(False)

        run_multivariate_benchmarks(False)













































