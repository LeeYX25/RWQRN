import numpy as np
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from utils import create_folds, batches
from torch_utils import clip_gradient, logsumexp
from skgarden import MondrianForestRegressor
from skgarden import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from collections import Counter
import math


'''RWQRN'''
class QuantileNetworkModule(nn.Module):
    def __init__(self, X_means, X_stds, y_mean, y_std, n_out):
        super(QuantileNetworkModule, self).__init__()
        self.X_means = X_means
        self.X_stds = X_stds
        self.y_mean = y_mean
        self.y_std = y_std
        self.n_in = X_means.shape[1]
        self.n_out = n_out
        self.fc_in = nn.Sequential(
                nn.Linear(self.n_in, 200),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.BatchNorm1d(200),
                nn.Linear(200, 200),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.BatchNorm1d(200),
                nn.Linear(200, self.n_out if len(self.y_mean.shape) == 1 else self.n_out * self.y_mean.shape[1]))
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        fout = self.fc_in(x)

        if len(self.y_mean.shape) != 1:
            fout = fout.reshape((-1, self.y_mean.shape[1], self.n_out))

        if self.n_out == 1:
            return fout

        #Ensure non - crossing
        return torch.cat((fout[...,0:1], fout[...,0:1] + torch.cumsum(self.softplus(fout[...,1:]), dim=-1)), dim=-1)
        
    def predict(self, X):
        self.eval()
        self.zero_grad()
        tX = autograd.Variable(torch.FloatTensor((X - self.X_means) / self.X_stds), requires_grad=False)
        fout = self.forward(tX)
        return fout.data.numpy() * self.y_std[...,None] + self.y_mean[...,None]

class QuantileNetwork:
    def __init__(self, quantiles, loss='marginal'):
        self.quantiles = quantiles
        self.label = 'Quantile Network'
        self.filename = 'nn'
        self.lossfn = loss
        if self.lossfn != 'marginal':
            self.label += f' ({self.lossfn})'

    def fit(self, X, y,tau):
        norm=False
        self.model = fit_quantiles(X, y, tau,quantiles=self.quantiles, lossfn=self.lossfn,verbose=True)

    def predict(self, X):
        return self.model.predict(X)

def fit_quantiles(X, y, tau, quantiles=0.5, lossfn = 'marginal',
                    nepochs=100, val_pct=0.1,
                    batch_size=50, target_batch_pct=0.01,
                    min_batch_size=20, max_batch_size=100,
                    verbose=True, lr=1e-1, weight_decay=0.0, patience=5,
                    init_model=None, splits=None, file_checkpoints=True,
                    clip_gradients=False, **kwargs):
    verbose=True
    if file_checkpoints:
        import uuid
        tmp_file = '/tmp/tmp_file_' + str(uuid.uuid4())

    if batch_size is None:
        batch_size = min(X.shape[0], max(min_batch_size, min(max_batch_size, int(np.round(X.shape[0]*target_batch_pct)))))
        if verbose:
            print('Auto batch size chosen to be {}'.format(batch_size))

    device='cuda'
    n = X.shape[0]
    losstype=0
    X_data = X
    y_data = y
    weight = get_rfmatrix(X_data, y_data)
    weight = torch.FloatTensor(weight).to(device)
    Xmean = X.mean(axis=0, keepdims=True)
    Xstd = X.std(axis=0, keepdims=True)
    Xstd[Xstd == 0] = 1 # Handle constant features
    ymean, ystd = y.mean(axis=0, keepdims=True), y.std(axis=0, keepdims=True)
    tX = autograd.Variable(torch.FloatTensor((X - Xmean) / Xstd), requires_grad=False)
    tY = autograd.Variable(torch.FloatTensor((y - ymean) / ystd), requires_grad=False)
    # Create train/validate splits
    if splits is None:
        indices = np.arange(X.shape[0], dtype=int)
        np.random.shuffle(indices)
        train_cutoff = int(np.round(len(indices)*(1-val_pct)))
        train_indices = indices[:train_cutoff]
        validate_indices = indices[train_cutoff:]
    else:
        train_indices, validate_indices = splits

    if np.isscalar(quantiles):
        quantiles = np.array([quantiles])

    tquantiles = autograd.Variable(torch.FloatTensor(quantiles), requires_grad=False)
    tquantiles = tquantiles.to(device)

    # Initialize the model
    model = QuantileNetworkModule(Xmean, Xstd, ymean, ystd, quantiles.shape[0]) if init_model is None else init_model

    # Save the model to file
    if file_checkpoints:
        torch.save(model, tmp_file)
    else:
        import pickle
        model_str = pickle.dumps(model)

    # Setup the SGD method
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, nesterov=True, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # Track progress
    train_losses, val_losses, best_loss = np.zeros(nepochs), np.zeros(nepochs), None
    num_bad_epochs = 0

    if verbose:
        print('ymax and min:', tY.max(), tY.min())


    # Univariate quantile loss
    def quantile_loss(yhat, tidx,tau,weight,u):
        z = tY[tidx,None] - yhat
        tidx = tidx.detach().cpu().numpy()
        z=z.to(device)
        tmp_x = tX[tidx,None].to(device)
        tmp_y = tY[tidx,None].to(device)
        wbatch_size = len(tidx)
        tmp_w = weight[np.tile(tidx, (wbatch_size, 1)).T.ravel(),np.tile(tidx, (1, wbatch_size))].reshape(wbatch_size, -1).T.to(device)
        tmp_w =  tmp_w - torch.diag(torch.diag(tmp_w))
        tmp_fx = yhat
        if(len(yhat.shape)==1):  #Predicting sigle quantile
            tmp_y = tmp_y.squeeze(1)
            loss1= torch.max(tquantiles[None]*z, (tquantiles[None] - 1)*z).mean() #computing  loss_emp
            tmp_my = torch.tile(tmp_y, (wbatch_size, 1)).to(device)
            tmp_mfx = torch.tile(tmp_fx, (1, wbatch_size)).to(device)
            lossmatrix= (tmp_my - tmp_mfx)* tmp_w
            loss2matrix=torch.max(tquantiles[None]*lossmatrix, (tquantiles[None] - 1)*lossmatrix)
            loss2=torch.mean(loss2matrix)*n/(n-1)   #computing loss_pair
            loss = tau * loss1 + (1 - tau) * loss2  #trade off
        else: #Predicting multiple quantiles
            dy=yhat.shape[1]
            tmp_y=tmp_y.expand(-1, dy)
            loss1 = torch.max(tquantiles[None] * z, (tquantiles[None] - 1) * z)
            loss1=torch.mean(loss1,dim=0)
            tmp_my = torch.tile(tmp_y.unsqueeze(0), (wbatch_size, 1,1)).to(device)
            tmp_mfx = torch.tile(tmp_fx.unsqueeze(1), (1, wbatch_size,1)).to(device)
            lossmatrix = (tmp_my - tmp_mfx) * tmp_w.unsqueeze(-1)
            loss2matrix = torch.max(tquantiles[None] * lossmatrix, (tquantiles[None] - 1) * lossmatrix)
            loss2 = torch.mean(loss2matrix,dim=[0,1]) * n / (n - 1)
            loss = tau * loss1 + (1 - tau) * loss2
        return loss

    # Marginal quantile loss for multivariate response
    def marginal_loss(yhat, tidx, tau, weight,u):
        n = yhat.shape[0]
        z =tY[tidx,:,None] - yhat
        z=z.to(device)
        loss1 = torch.max(tquantiles[None, None] * z, (tquantiles[None, None] - 1) * z).mean()
        wbatch_size=len(tidx)
        tmp_x = tX[tidx,:,None] .to(device)
        tmp_y = tY[tidx,:,None] .to(device)
        tmp_w = weight[np.tile(tidx, (wbatch_size, 1)).T.ravel(),
        np.tile(tidx, (1, wbatch_size))].reshape(wbatch_size, -1).T.to(device)
        tmp_w =  tmp_w - torch.diag(torch.diag(tmp_w))
        tmp_fx = yhat
        if(yhat.ndim==2):
            tmp_y=tmp_y.squeeze(2)
            tmp_fx=tmp_fx.squeeze(2)
            tmp_my = torch.tile(tmp_y.unsqueeze(0), (wbatch_size, 1,1)).to(device)
            tmp_mfx = torch.tile(tmp_fx.unsqueeze(1), (1, wbatch_size, 1)).to(device)
            lossmatrix = (tmp_my - tmp_mfx) * tmp_w[:, :, None].unsqueeze(-1)  # (batch_size, batch_size, m)
            loss2matrix = torch.max(tquantiles[None, None] * lossmatrix, (tquantiles[None, None] - 1) * lossmatrix)
            loss2 = torch.mean(loss2matrix) * n / (n - 1)
            loss = tau * loss1 + (1 - tau) * loss2
        else:
            dy=yhat.shape[2]
            tmp_y=tmp_y.expand(-1, -1,dy)
            tmp_my = torch.tile(tmp_y.unsqueeze(0), (wbatch_size, 1,1,1)).to(device)
            tmp_mfx = torch.tile(tmp_fx.unsqueeze(1), (1, wbatch_size,1,1)).to(device)
            lossmatrix = (tmp_my - tmp_mfx) * tmp_w[:, :, None].unsqueeze(-1)  # (batch_size, batch_size, m)
            loss2matrix = torch.max(tquantiles[None, None] * lossmatrix, (tquantiles[None, None] - 1) * lossmatrix)
            loss2 = torch.mean(loss2matrix,dim=[0,1]) * n / (n - 1)
            loss = tau * loss1 + (1 - tau) * loss2
        return loss

    def generate_valid_u(n_quantiles, d_dim, device='cpu'):
        """
        Generates valid geometric quantile vectors u, ensuring ||u|| < 1.

        Args:
            n_quantiles (int): Number of quantiles (M)
            d_dim (int): Dimension of response variable (D)
            device: 'cpu' or 'cuda'

        Returns:
            u_vectors (Tensor): Shape (M, D)
        """
        # 1. Generate random directions on the unit sphere
        u_dirs = torch.randn(n_quantiles, d_dim, device=device)
        u_dirs = u_dirs / (torch.norm(u_dirs, dim=1, keepdim=True) + 1e-8)

        # 2. Randomly scale magnitudes to [0, 0.99]
        # Critical: Norm must be strictly < 1 to ensure the loss is convex and non-negative
        magnitudes = torch.rand(n_quantiles, 1, device=device) * 0.99

        u_vectors = u_dirs * magnitudes
        return u_vectors

    def geometric_loss(yhat, tidx, tau, weight_matrix, u_vectors):
        """
        Computes the Geometric Quantile Loss within the RWQRN framework.

        Args:
            yhat (Tensor): Predictions, shape (Batch, D, M)
            tidx (list/array): Indices for the current batch
            tau (float): Trade-off parameter
            weight_matrix (Tensor): Global Random Forest weight matrix (N, N)
            u_vectors (Tensor): Geometric quantile vectors, shape (M, D)

        Returns:
            total_loss (Tensor): Scalar loss
        """
        # Ensure the Feature Dimension (D) is the last dimension for norm/dot calculations
        # Transform (Batch, Dim, Quantiles) -> (Batch, Quantiles, Dim)
        y_true = tY[tidx, :]
        if yhat.ndim == 3 and yhat.shape[1] == y_true.shape[1]:
            yhat = yhat.permute(0, 2, 1)

        B, M, D = yhat.shape
        device = yhat.device

        # Ensure all tensors are on the correct device
        u_vectors = u_vectors.to(device)  # (M, D)

        # Broadcast y_true from (B, D) to (B, 1, D)
        r_self = y_true.unsqueeze(1) - yhat

        # Norm ||r|| along feature dimension D
        norm_self = torch.norm(r_self, dim=2)

        # Inner product <u, r>
        dot_self = (r_self * u_vectors.unsqueeze(0)).sum(dim=2)

        loss_self = torch.mean(norm_self + dot_self)

        # --- 2. Compute Weighted Risk (Local RF Loss) ---
        # Target Y_j (neighbor): (B, 1, 1, D)
        Y_j = y_true.view(B, 1, 1, D)

        # Prediction f(X_i) (anchor): (1, B, M, D)
        f_X_i = yhat.view(1, B, M, D)

        # r_cross shape: (Batch_j, Batch_i, Quantiles, Dim)
        r_cross = Y_j - f_X_i

        # Pairwise Norm & Dot product
        norm_cross = torch.norm(r_cross, dim=3)
        dot_cross = (r_cross * u_vectors.view(1, 1, M, D)).sum(dim=3)

        L_cross = norm_cross + dot_cross  # (B, B, M)

        if isinstance(weight_matrix, torch.Tensor):
            batch_weights = weight_matrix[np.ix_(tidx, tidx)].to(device)
        else:
            batch_weights = torch.tensor(weight_matrix[np.ix_(tidx, tidx)], device=device, dtype=yhat.dtype)

        batch_weights.fill_diagonal_(0)

        W = batch_weights.unsqueeze(2)

        if B > 1:
            loss_rf = (L_cross * W).sum() / (B * (B - 1))
        else:
            loss_rf = torch.tensor(0.0, device=device)

        total_loss = tau* loss_self + (1 - tau) * loss_rf

        return total_loss


    # Create the quantile loss function
    if len(tY.shape) == 1 or tY.shape[1] == 1:
        lossfn = quantile_loss
    elif lossfn == 'marginal':
        print('Using marginal loss')
        lossfn = marginal_loss
    elif lossfn == 'geometric':
        print('Using geometric loss')
        lossfn = geometric_loss
        losstype=3

            

    # Train the model
    for epoch in range(nepochs):
        if verbose:
            print('\t\tEpoch {}'.format(epoch+1))
            sys.stdout.flush()

        # Track the loss curves
        train_loss = torch.Tensor([0]).to(device)
        if losstype==3:
            uvec = generate_valid_u(n_quantiles=tquantiles.shape[0], d_dim=y.shape[1], device=device)
        else:
            uvec=None
        for batch_idx, batch in enumerate(batches(train_indices, batch_size, shuffle=True)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tBatch {}'.format(batch_idx))
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            model.train()

            model.zero_grad()

            yhat = model(tX[tidx])

            loss = lossfn(yhat, tidx,tau,weight,uvec).mean().to(device)

            loss.backward()

            if clip_gradients:
                clip_gradient(model)

            optimizer.step()

            train_loss += loss.data

            if np.isnan(loss.data.cpu().numpy()):
                import warnings
                warnings.warn('NaNs encountered in training model.')
                break

        validate_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(validate_indices, batch_size, shuffle=False)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tValidation Batch {}'.format(batch_idx))
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            model.eval()

            model.zero_grad()

            yhat = model(tX[tidx])

            validate_loss=validate_loss.to(device)
            validate_loss += lossfn(yhat, tidx,tau,weight,uvec).sum().to(device)

        train_losses[epoch] = train_loss.data.cpu().numpy() / float(len(train_indices))
        val_losses[epoch] = validate_loss.data.cpu().numpy() / float(len(validate_indices))

        if num_bad_epochs > patience:
            if verbose:
                print('Decreasing learning rate to {}'.format(lr*0.5))
            scheduler.step(val_losses[epoch])
            lr *= 0.5
            num_bad_epochs = 0

        # If the model blew up and gave us NaNs, adjust the learning rate down and restart
        if np.isnan(val_losses[epoch]):
            if verbose:
                print('Network went to NaN. Readjusting learning rate down by 50%')
            if file_checkpoints:
                os.remove(tmp_file)
            return fit_quantiles(X, y, tau, quantiles=quantiles, lossfn=lossfn,
                    nepochs=nepochs, val_pct=val_pct,
                    batch_size=batch_size, target_batch_pct=target_batch_pct,
                    min_batch_size=min_batch_size, max_batch_size=max_batch_size,
                    verbose=verbose, lr=lr*0.5, weight_decay=weight_decay, patience=patience,
                    init_model=init_model, splits=splits, file_checkpoints=file_checkpoints,  **kwargs)

        # Check if we are currently have the best held-out log-likelihood
        if epoch == 0 or val_losses[epoch] <= best_loss:
            if verbose:
                print('\t\t\tSaving test set results.      <----- New high water mark on epoch {}'.format(epoch+1))
            # If so, use the current model on the test set
            best_loss = val_losses[epoch]
            if file_checkpoints:
                torch.save(model, tmp_file)
            else:
                import pickle
                model_str = pickle.dumps(model)
        else:
            num_bad_epochs += 1
        
        if verbose:
            print('Validation loss: {} Best: {}'.format(val_losses[epoch], best_loss))

    # Load the best model and clean up the checkpoints
    if file_checkpoints:
        model = torch.load(tmp_file)
        os.remove(tmp_file)
    else:
        import pickle
        model = pickle.loads(model_str)

    # Return the conditional density model that marginalizes out the grid
    return model


def rf_graph(x):
    G = np.zeros((len(x), len(x)))

    for i in range(len(x)):
        G[i, np.where(x == x[i])[0]] = 1

    nodes = Counter(x)
    nodes_num = np.array([nodes[i] for i in x])

    return G, G / nodes_num.reshape(-1, 1)


def get_rfweight(rf, x):
    n = x.shape[0]

    leaf = rf.apply(x)
    ntrees = leaf.shape[1]
    G_unnorm = np.zeros((n, n))
    G_norm = np.zeros((n, n))

    for i in range(ntrees):
        tmp1, tmp2 = rf_graph(leaf[:, i])
        G_unnorm += tmp1
        G_norm += tmp2

    return G_unnorm / ntrees, G_norm / ntrees

def get_rfmatrix(x,y):

    params_mrf = {'min_samples_split': [2, 3, 4, 5, 6, 7,8,9,10]}
    if(len(y.shape)==1):
        model = MondrianForestRegressor(n_estimators=100)
        print("Use MondrianForestRegressor")
    else:
        model = RandomForestRegressor(n_estimators=100)
        print("Use RandomForestRegressor")
    reg_mrf = GridSearchCV(model, params_mrf)
    n=x.shape[0]
    reg_mrf.fit(x, y)
    mrf = reg_mrf.best_estimator_
    mrf.fit(x, y)
    mrfw= get_rfweight(mrf, x)[0]
    return mrfw
