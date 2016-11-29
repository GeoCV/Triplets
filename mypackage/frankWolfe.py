from ste import *
from new_utils import *

import numpy as np 
from scipy.sparse.linalg import eigs 
import Utils
from time import time 
import matplotlib.pyplot as plt

def frankWolfe(X0, S, maxits=100, epsilon=1e-3, debug=False):
    """
    Inputs:
    (ndarray) X0: initial embedding
    (list) S: the triplets in [win, lose, center] order
    (int) maxits: maximum number of iterations
    (float) epsilon: stopping condition
    (bool) debug: controls verbosity

    Returns: 
    (ndarray) X: the learned embedding
    (dict) stats: performance statistics for this algorithm
    """
    stats = {}    
    stats['emp'] = []                           # list of empirical loss per iteration
    stats['log'] = []                           # list of log loss per iteration
    stats['time_per_iter'] = []                 # list of times taken per iteration
    stats['avg_time_per_iter'] = 0              # mean time for single iteration

    M = np.dot(X0, X0.T)
    it = 0       # iteration counter
    dif = float('inf')
    Gnorm = float('inf')
    n, d = X0.shape

    while it < maxits and (dif > epsilon and Gnorm > epsilon):
        start = time()                           # start time
        M_old = M
        alpha = 10/(it + 2)                      # heuristic step size
        G = ste_loss_convex(M, S, 2, descent_alg='full_grad')
        _, v = eigs(-1.*G, k=1)                     # get largest eigenvalue
        M = M + alpha*(np.outer(v,v) - M)       # perform rank-1 update
        end = time()                            # end time

        # stopping variables:
        dif = np.linalg.norm(M - M_old, ord='fro')
        Gnorm = np.linalg.norm(G, ord='fro')            # norm of gradient
        it += 1

        # check if there is any progress:
        if it > 5:
            smallest = min(stats['log'][::-1][:5]) # last ten guys
            biggest = max(stats['log'][::-1][:5])  # last ten guys             
            if abs(smallest - biggest) < epsilon:
                print('No progress')              
                break

        # save stats
        current_losses = ste_loss_convex(M, S, 1)
        stats['log'].append(current_losses['log_loss'])
        stats['emp'].append(current_losses['empirical_loss'])
        stats['time_per_iter'].append(end - start)
        stats['avg_time_per_iter'] = np.mean(stats['time_per_iter'])

        if debug:
            print('iter=%d, emp_loss=%f, log_loss=%f, Gnorm=%f, dif=%f' %(it, stats['emp'][-1], stats['log'][-1], Gnorm, dif))
    
    _, X = Utils.transform_MtoX(M, d)
    return X, stats 

if __name__ == '__main__':
    n = 20
    d = 2
    Xtrue = Utils.center_data(np.random.rand(n,d))
    pulls = int(10*n*d*np.log(n))
    S, bayes_err = Utils.getTriplets(Xtrue, pulls, noise=True)
    print("estiamted best error is: %f" %bayes_err)
    X0 = Utils.center_data(np.random.rand(n,d))
    Xhat, stats = frankWolfe(X0, S, debug=True)

    _, Xpro, _ = Utils.procrustes(Xtrue, Xhat)
    Utils.twodplot(Xtrue, Xpro)
    plt.show()

