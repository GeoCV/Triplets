from ste import *
from new_utils import *

import numpy as np 
import utils

def frankWolfe(X0, S, maxits=100, epsilon=1e-3, verbose=False):
    """
    Inputs:
    (ndarray) X0: initial embedding
    (list) S: the triplets in [win, lose, center] order
    (int) maxits: maximum number of iterations
    (float) epsilon: stopping condition
    (bool) verbose: controls verbosity

    Returns: 
    (ndarray) X: the learned embedding
    (dict) stats: performance statistics for this algorithm
    """
    stats = {}    
    stats['emp'] = [] # list of empirical loss per iteration
    stats['log'] = [] # list of log loss per iteration
    stats['time_per_iter'] = [] # list of times taken per iteration
    stats['avg_time_per_iter'] = 0  # mean time for single iteration

    M = np.dot(X0, X0)
    t = 0       # iteration counter
    dif = float('inf')

    while t < maxits and dif > epsilon:
        M_old = M
        alpha = 2/(t + 2)       # heuristic step size
        G = ste_loss_convex(M, S, 2, descent_alg='full_grad')
        _, v = eigs(G, k=1)     # get largest eigenvalue
        M = M + alpha*(np.outer(v,v) - M)       # perform rank-1 update

        dif = np.linalg.norm(M - M_old)

