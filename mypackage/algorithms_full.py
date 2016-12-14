from ste import *
from new_utils import *
import time

import numpy as np
from numpy import argmin, argmax
from numpy.random import random, randn
from Utils import center_data, getTriplets

import sys

def triplet_algorithms(f, 
                       S,
                       X0,
                       d,
                       descent_alg, 
                       step_size_func,
                       iters=100,
                       epsilon = 0.001,
                       toler= 10**-5,
                       proj=None,
                       debug=False):

    '''
    File contains the main triplets algorithm [still needs work]


    :param f: objective function 
    :param S: set of triplets
    :param X0: starting iterate
    :param d: ambient dimesnion
    :param descent_alg: choice for descent method (e.g. full_gradient/sgd etc.)
    :param iters: number of iterations
    :param epsilon: Accuracy parameter
    :param tolerance: Tolerance parameter
    :param proj: Variable to decide if we should project or not.

    :type f: python function with specific structure
    :type S: list of lists
    :type X0: a d dimensional vector
    :type d: int
    :type descent_alg: function
    :type iters: int
    :type epsilon: int
    :type toler: float
    :type proj: Boolean
    '''    

    # Stats I wish to collect about my experiment
    stats = {}    
    stats['emp'] = [] # list of empirical loss per iteration
    stats['log'] = [] # list of log loss per iteration
    stats['time_per_iter'] = [] # list of times taken per iteration
    stats['avg_time_per_iter'] = 0  # mean time for single iteration
    stats['status'] = 0 # convergence status
    stats['epoch_count'] = 0
    
    X_curr = X0
    n = len(X0)
    
    # FOR SVRG
    X_tilde = X0
    p_full = f(X_tilde, S, 2, descent_alg='full_grad')

    if descent_alg == 'sgd':
        iters = n*iters
        
    emp_X_curr = f(X0, S, 1)['empirical_loss']
    log_X_curr = f(X0, S, 1)['log_loss']

    stats['emp'].append(emp_X_curr)
    stats['log'].append(log_X_curr)

    # stopping condition variables
    dif = float('inf')
    Gnorm = float('inf')
    p = None

    # what to scale the gradient by for fair comparison
    scale = np.prod(X0.shape)

    # accuracy achieved
    if stats['emp'][-1] < epsilon:
        return stats

    alpha = step_size_func #  for now constant stepsize only


    # EPOCHS
    for iteration in range(1,iters):
        if dif < epsilon or Gnorm/scale < epsilon:
            print('Stopping condition achieved')
            print('EPOCH', iteration//n,'LOG ERROR', log_X_new, 'Emp error', emp_X_new, 'dif=', dif, 'Gnorm=',Gnorm)
            break    
        start = time.time()
                    
        # Every iteration is an epoch anyway
        stats['epoch_count'] += 1
        alpha = 0.98*alpha

        p = f(X_curr, S, 2, descent_alg=descent_alg)

            
        # Make sure we get the step size correct
        flag = False
        while flag == False:
            try:
                
                X_new = X_curr + alpha*p

                # PROJECTION
                if proj != None:

                    if proj == projected_psd:
                        X_new = proj(X_new)
                    else:
                        X_new = proj(X_new,d)

                emp_X_new = f(X_new, S, 1)['empirical_loss']
                log_X_new = f(X_new, S, 1)['log_loss']
                
                if iteration > 4:        
                    biggest = max(stats['log'][::-1][:3])  # last 3 guys

                    # function is increaseing compared to last 3 iterates
                    # update last guy to be present guy and re-do the whole thing                    
                    if log_X_new - biggest > 0.05:
                        stats['log'][-1] = log_X_new 
                        stats['status'] = -1
                        raise OverflowError

                # update last guy, so we can fairly compare
                flag = True
                
            except OverflowError:
                print('Step size was too big, halving it. Iteration #: ',iteration)
                stats['status'] = -1
                alpha /=2
                flag == False

        # Step size found
        stats['log'].append(log_X_new)
        stats['emp'].append(emp_X_new)
                
        if debug:
            print('EPOCH:', iteration, 'LOG ERROR', log_X_new, 'Emp error', emp_X_new, 'dif=', dif, 'avg Gnorm=',Gnorm/scale)

        end = time.time()
        stats['time_per_iter'].append((end - start))
        dif = np.linalg.norm(X_curr - X_new, ord='fro')
        Gnorm = np.linalg.norm(p, ord='fro')
        X_curr = X_new
    
    # print(X_curr.shape, np.linalg.matrix_rank(X_curr))
    stats['avg_time_per_iter'] = sum(stats['time_per_iter'])/(iteration+1)
    stats['embedding'] = X_curr.tolist()

    print('Exiting')
    
    return stats        


if __name__ == '__main__':

    #Create data
    dimensions= 5
    number_of_points= 100
    noise=False

    X = random((number_of_points, dimensions))
    X = center_data(X)
    n,d = X.shape
    pulls = 10*int(number_of_points*dimensions*np.log(number_of_points))

    triplets, error = getTriplets(X, pulls, noise=noise)
    print('Estimated error is:', error)

    X0 = np.random.random((n,d))
    M0 = np.array(X0 @ X0.T)
    
    stats= triplet_algorithms(ste_loss_convex, 
                           triplets,
                           M0,                       
                           d,
                           'full_grad', 
                           600,
                           iters=5000,
                           epsilon = 3e-7,
                           proj=projected_psd,
                           debug=True
    )

    Xhat = stats['embedding']

    triplets_test, testSet_error = getTriplets(X, pulls, noise=noise)
    print('Test set error is: ', testSet_error)
    print(ste_loss_convex(Xhat, triplets_test, 1))