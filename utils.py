#%% Utility Functions for QuantCon2016 Presentation, Covariance Matrix Regularization with AutoEncoders
# @ Erk Subasi, 2016, Limmat Capital Alternative Investments AG


import numpy as np
from numpy.linalg import norm
from numpy import diag, inf
from numpy import copy, dot
from pandas_datareader import data, wb

import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
import cPickle as pickle

import tensorflow as tf
import time
import sys
import tqdm

import zipline
from zipline import TradingAlgorithm
from zipline.utils.factory import load_bars_from_yahoo
from zipline.api import (add_history, history, set_slippage, slippage,set_commission, commission, 
                         order_target_percent,get_datetime)

np.random.seed(123)
# Turn off progress printing 
solvers.options['show_progress'] = False


def ShowCovarianceMatrix(covmat,names,fileName=None, title=None):
    plt.figure(figsize=(7,7))
    plt.imshow(covmat,interpolation='none',aspect=1, vmin= 1.1*np.min(-0.0005,np.min(covmat)), 
               vmax=1.1*np.max(0.0005,np.max(covmat)), cmap='PRGn')
    if title is not None:
        plt.title(title)
    plt.xticks(range(len(names)),names,rotation='vertical')
    plt.yticks(range(len(names)),names,rotation='horizontal')
    if fileName is not None:
        plt.savefig(fileName, bbox_inches='tight')
	plt.close()
    
    
def ShowWeights(weights,names,fileName=None):
    plt.figure(figsize=(7,7))
    plt.bar(range(len(names)),weights,label='Weights')   
    plt.xticks(range(len(names)),names,rotation='vertical')
    if fileName is not None:
        plt.savefig(fileName, bbox_inches='tight')
	plt.close()

def ShowCovarsWeights(covmats,weights,names,fileName=None):
    N = len(names)
    f, axarr = plt.subplots(2, 2, figsize=(12,12)) # sharex='col', sharey='row')
    axarr[0, 0].set_title('Sample Covariance')
    axarr[0,0].imshow(covmats[0],interpolation='none',aspect=1, vmin= 1.0*np.min(-0.0005,np.min(covmats[0])), vmax=1.0*np.max(0.0005,np.max(covmats[0])), cmap='seismic')
    axarr[1, 0].set_title('AutoEncoder Covariance')
    axarr[1,0].imshow(covmats[1],interpolation='none',aspect=1, vmin= 1.0*np.min(-0.0005,np.min(covmats[1])), vmax=1.0*np.max(0.0005,np.max(covmats[1])), cmap='seismic')    
    axarr[0, 1].set_title('Sample Min-Variance Weights')
    axarr[0,1].bar(range(len(names)),weights[0],label='Weights')   
    axarr[1, 1].set_title('AutoEncoder Min-Variance Weights')
    axarr[1,1].bar(range(len(names)),weights[1],label='Weights')   

    plt.sca(axarr[0, 0])
    plt.yticks(range(N), names, rotation='horizontal')
    plt.sca(axarr[1, 0])
    plt.yticks(range(N), names, rotation='horizontal')
    plt.xticks(range(N), names, rotation='vertical')
    
    plt.sca(axarr[1, 1])
    plt.xticks(range(len(names))+0.5*np.ones(len(names)), names, rotation='vertical')

    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    
    if fileName is not None:
        plt.savefig(fileName, bbox_inches='tight')
	plt.close(f)

       
def cov2corr(covmat):
    N = covmat.shape[0]
    vola = np.sqrt(np.diag(covmat))
    corrmat = covmat * 1
    for j in range(0, N):
        for k in range(j, N):
            corrmat[j,k] = covmat[j,k] / (vola[j]*vola[k])
            corrmat[k,j] = corrmat[j,k]
    return corrmat


def cov2corrvola(covmat):
    N = covmat.shape[0]
    vola = np.sqrt(np.diag(covmat))
    corrmat = covmat * 1
    for j in range(0, N):
        for k in range(j, N):
            corrmat[j,k] = covmat[j,k] / (vola[j]*vola[k])
            corrmat[k,j] = corrmat[j,k]
    return corrmat, vola


def corr2cov(corrmat, vola):
    N = corrmat.shape[0]
    covmat = corrmat * 1 
    for j in range(0, N):
        for k in range(j, N):
            covmat[j,k] = corrmat[j,k] * (vola[j]*vola[k])
            covmat[k,j] = covmat[j,k]
    return covmat


def occlude_input(x, rate):
    x_cp = np.copy(x)
    inp_to_drop = np.random.rand(x_cp.shape[0], x_cp.shape[1]) < rate
    x_cp[inp_to_drop] = 1e-9
    return x_cp


# based on the Quantopian sample
def optimal_portfolio_cvxopt(C, R=None, EF=False):
    n = C.shape[0]    # num of assets in the port
    if R is None:
        R = np.ones(n)   # min. variance setting
        
    # Convert to cvxopt matrices
    # minimize: w  * mu*S * w + mean_rets *x
    S = opt.matrix(C) 
    pbar = opt.matrix(R)    
    
    # Create constraint matrices
    # Gx < h: Every weights is positive
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    # Ax = b: sum of all weights equal to 1
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    if EF:
        effr_mean, effr_var, effr_weights = [], [], []
        N = 50
        mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

        # Calculate efficient frontier weights using quadratic programming
        portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                      for mu in mus]
        ## CALCULATE RISKS AND RETURNS FOR FRONTIER
        effr_mean = [blas.dot(pbar, x) for x in portfolios]
        effr_var = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
        effr_weights = [np.asarray(x).ravel() for x in portfolios]
        
        ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
        m1 = np.polyfit(returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        # CALCULATE THE OPTIMAL PORTFOLIO
        wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
        return effr_mean, effr_var, effr_weights
    else:
        wt = solvers.qp(opt.matrix(S), -pbar, G, h, A, b)['x']
        return np.asarray(wt).ravel()



class ExceededMaxIterationsError(Exception):
    def __init__(self, msg, matrix=[], iteration=[], ds=[]):
        self.msg = msg
        self.matrix = matrix
        self.iteration = iteration
        self.ds = ds

    def __str__(self):
        return repr(self.msg)


def nearcorr(A, tol=[], flag=0, max_iterations=100, n_pos_eig=0,
             weights=None, verbose=False,
             except_on_too_many_iterations=True):
    """
    X = nearcorr(A, tol=[], flag=0, max_iterations=100, n_pos_eig=0,
        weights=None, print=0)

    Finds the nearest correlation matrix to the symmetric matrix A.

    ARGUMENTS
    ~~~~~~~~~
    A is a symmetric numpy array or a ExceededMaxIterationsError object

    tol is a convergence tolerance, which defaults to 16*EPS.
    If using flag == 1, tol must be a size 2 tuple, with first component
    the convergence tolerance and second component a tolerance
    for defining "sufficiently positive" eigenvalues.

    flag = 0: solve using full eigendecomposition (EIG).
    flag = 1: treat as "highly non-positive definite A" and solve
    using partial eigendecomposition (EIGS). CURRENTLY NOT IMPLEMENTED

    max_iterations is the maximum number of iterations (default 100,
    but may need to be increased).

    n_pos_eig (optional) is the known number of positive eigenvalues
    of A. CURRENTLY NOT IMPLEMENTED

    weights is an optional vector defining a diagonal weight matrix diag(W).

    verbose = True for display of intermediate output.
    CURRENTLY NOT IMPLEMENTED

    except_on_too_many_iterations = True to raise an exeption when
    number of iterations exceeds max_iterations
    except_on_too_many_iterations = False to silently return the best result
    found after max_iterations number of iterations

    ABOUT
    ~~~~~~
    This is a Python port by Michael Croucher, November 2014
    Thanks to Vedran Sego for many useful comments and suggestions.

    Original MATLAB code by N. J. Higham, 13/6/01, updated 30/1/13.
    Reference:  N. J. Higham, Computing the nearest correlation
    matrix---A problem from finance. IMA J. Numer. Anal.,
    22(3):329-343, 2002.
    """

    # If input is an ExceededMaxIterationsError object this
    # is a restart computation
    if (isinstance(A, ExceededMaxIterationsError)):
        ds = copy(A.ds)
        A = copy(A.matrix)
    else:
        ds = np.zeros(np.shape(A))

    eps = np.spacing(1)
    if not np.all((np.transpose(A) == A)):
        raise ValueError('Input Matrix is not symmetric')
    if not tol:
        tol = eps * np.shape(A)[0] * np.array([1, 1])
    if weights is None:
        weights = np.ones(np.shape(A)[0])
    X = copy(A)
    Y = copy(A)
    rel_diffY = inf
    rel_diffX = inf
    rel_diffXY = inf

    Whalf = np.sqrt(np.outer(weights, weights))

    iteration = 0
    while max(rel_diffX, rel_diffY, rel_diffXY) > tol[0]:
        iteration += 1
        if iteration > max_iterations:
            if except_on_too_many_iterations:
                if max_iterations == 1:
                    message = "No solution found in "\
                              + str(max_iterations) + " iteration"
                else:
                    message = "No solution found in "\
                              + str(max_iterations) + " iterations"
                raise ExceededMaxIterationsError(message, X, iteration, ds)
            else:
                # exceptOnTooManyIterations is false so just silently
                # return the result even though it has not converged
                return X

        Xold = copy(X)
        R = X - ds
        R_wtd = Whalf*R
        if flag == 0:
            X = proj_spd(R_wtd)
        elif flag == 1:
            raise NotImplementedError("Setting 'flag' to 1 is currently\
                                 not implemented.")
        X = X / Whalf
        ds = X - R
        Yold = copy(Y)
        Y = copy(X)
        np.fill_diagonal(Y, 1)
        normY = norm(Y, 'fro')
        rel_diffX = norm(X - Xold, 'fro') / norm(X, 'fro')
        rel_diffY = norm(Y - Yold, 'fro') / normY
        rel_diffXY = norm(Y - X, 'fro') / normY

        X = copy(Y)

    return X


def proj_spd(A):
    # NOTE: the input matrix is assumed to be symmetric
    d, v = np.linalg.eigh(A)
    A = (v * np.maximum(d, 0)).dot(v.T)
    A = (A + A.T) / 2
    return(A)
