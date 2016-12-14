from __future__ import division
from ste import *
from new_utils import *

import numpy as np 
from scipy.sparse.linalg import eigsh 
import Utils
from time import time 
import matplotlib.pyplot as plt

def eigen_embed(X0, S, alpha=600, method='rankD',maxits=100, epsilon=1e-3, debug=False):
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
    scale = n**2

    alpha = alpha/0.98
    while it < maxits:
        start = time()                           # start time
        M_old = M
        G = ste_loss_convex(M, S, 2, descent_alg='full_grad')

        if dif < epsilon or Gnorm/scale < epsilon:
            print("Stopping condition achieved")
            break

        # Frank-Wolfe method
        if method=='FW':        
            # alpha = 10/(it + 2)                      	# step size to guarantee a sublinear rate
            # alpha = 0.5
            alpha=0.8*alpha
            _, v = eigsh(G, k=1)          # get largest eigenvalue
            M = M + alpha*(np.outer(v,v) - M)           # perform rank-1 update

        # Rank D projection
        elif method=='rankD':
            alpha = 0.98*alpha
            w,V = eigsh(M + alpha*G, k=d)       # take a gradient step and immediately compute only top d eigenvectors and eigenvalues
            M = V.dot(np.diag(w)).dot(V.T)      # finish projection by re-forming rank d M

        else:
            raise AssertionError("method must be either 'FW for Frank-Wolfe algorithm or 'rankD' for rank d projection algorithm")

        end = time()                                    # ending time

        # stopping variables:
        dif = np.linalg.norm(M - M_old, ord='fro')
        Gnorm = np.linalg.norm(G, ord='fro')            # norm of gradient
        it += 1

        # check if there is any progress:
        if it > 30:
            smallest = min(stats['log'][::-1][:10])     # last ten iterates
            biggest = max(stats['log'][::-1][:10])      # last ten iterates             
            if abs(smallest - biggest) < 10*epsilon:
                print('No progress')              
                break

        # save stats
        current_losses = ste_loss_convex(M, S, 1)
        stats['log'].append(current_losses['log_loss'])
        stats['emp'].append(current_losses['empirical_loss'])
        stats['time_per_iter'].append(end - start)
        stats['avg_time_per_iter'] = np.mean(stats['time_per_iter'])

        # if not (dif > epsilon and Gnorm > epsilon):            
        #     break
        
        if debug:
            Mnorm = np.linalg.norm(M, ord='fro')            # norm of the Gram matrix to ensure that we do not blow up embedding
                                                            # this is especially important for FW since the set we solve over 
                                                            # must be compact. This ensures we can assume boundedness
            print('iter=%d, emp_loss=%f, log_loss=%f, avg Gnorm=%.10e, Mnorm=%f, dif=%f' %(it, stats['emp'][-1], stats['log'][-1], Gnorm/scale, Mnorm, dif))
    
    print(np.linalg.matrix_rank(M))
    _, X = Utils.transform_MtoX(M, d)
    stats['avg_time_per_iter'] = sum(stats['time_per_iter'])/(it+1)
    stats['embedding'] = X.tolist()    
    return X, stats 


if __name__ == '__main__':
    n = 100
    d = 5
    noise = True

    Xtrue = Utils.center_data(np.random.rand(n,d))
    pulls = int(10*n*d*np.log(n))
    S, bayes_err = Utils.getTriplets(Xtrue, pulls, noise=noise)
    print("estiamted best error is: %f" %bayes_err)
    X0 = Utils.center_data(np.random.rand(n,d))
    Xhat, stats = eigen_embed(X0, S, alpha=1, method='FW', epsilon=1.1e-6, debug=True)

    triplets_test, testSet_error = Utils.getTriplets(Xtrue, pulls, noise=noise)
    print('Test set error is: ', testSet_error)
    print(ste_loss(Xhat, triplets_test, 1))

    if d == 2:
        _, Xpro, _ = Utils.procrustes(Xtrue, Xhat)
        Utils.twodplot(Xtrue, Xpro)
        plt.show()
