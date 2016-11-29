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
    emp_X_curr = f(X0, S, 1)['empirical_loss']
    log_X_curr = f(X0, S, 1)['log_loss']

    stats['emp'].append(emp_X_curr)
    stats['log'].append(log_X_curr)

    
    # accuracy achieved
    if stats['emp'][-1] < epsilon:
        return stats

    alpha = step_size_func #  for now constant stepsize only

    n = len(X0)

    # EPOCHS
    for iteration in range(iters):    

        start = time.time()

        if descent_alg == 'sgd':
            # Shrink every epoch
            if iteration %n == 0:
                stats['epoch_count'] += 1
                alpha = 0.9*alpha
                
                if debug and descent_alg=='sgd':
                    print('Shrinking alpha', alpha)
                    print(iteration, 'LOG ERROR', log_X_new, 'Emp error', emp_X_new)

            # Get descent direction: Currently all the work for FG and sgd
            # need to fit SVRG in this frameworks
            p = f(X_curr, S, 2, descent_alg=descent_alg)
                    
        elif descent_alg == 'full_grad':
            stats['epoch_count'] += 1
            alpha = 0.98*alpha

            # Get descent direction: Currently all the work for FG and sgd
            # need to fit SVRG in this frameworks
            p = f(X_curr, S, 2, descent_alg=descent_alg)

        elif descent_alg == 'svrg':

            # Need to find a new descent direction now for SVRFG

        # Make sure we get the step size correct
        flag = False
        while flag == False:
            try:
                X_new = X_curr + alpha*p

                if proj != None:
                    X_new = proj(X_new, d)

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
                
        if debug and descent_alg=='full_grad':
            print(iteration, 'LOG ERROR', log_X_new, 'Emp error', emp_X_new)
        
        # accuracy achieved
        if stats['emp'][-1] < epsilon:
            print('Accuracy reached in {} iterations'.format(iteration))
            break
            
        # gradient is very very small
        if descent_alg == 'full_grad' or descent_alg == 'sgd':
            if norm(p) < 10**-6:
                stats['status'] = 0                
                print('Gradient too small')
                break
                
        # No substantial increase in last ten guys
        # has to run for atleast 20 iterations
        if iteration > 20:
            smallest = min(stats['log'][::-1][:10]) # last ten guys
            biggest = max(stats['log'][::-1][:10])  # last ten guys             
            if abs(smallest - biggest) < toler:
                print('No progress')
                stats['status'] = 0                
                break
        
            # divergence : currently very ad hoc
            smallest = min(stats['log'][::-1][:10]) # last 10 guys
            biggest = max(stats['log'][::-1][:10])  # last 10 guys

            smallest_ind = argmin(stats['log'][::-1][:10]) # last 10 guys
            biggest_ind = argmax(stats['log'][::-1][:10])  # last 10 guys                

            # function value has changed greatly and the function is increaseing
            if abs(smallest - biggest) > 1 and smallest_ind > biggest_ind:
                print('Divergence')
                stats['status'] = -1
                break

            
        end = time.time()
        stats['time_per_iter'].append((end - start))
        X_curr = X_new
    
    stats['avg_time_per_iter'] = sum(stats['time_per_iter'])/(iteration+1)
    stats['embedding'] = X_curr

    print('Exiting')
    
    return stats        


if __name__ == '__main__':

    #Create data
    dimensions= 15
    number_of_points= 100

    X = random((number_of_points, dimensions))
    X = center_data(X)
    n,d = X.shape
    pulls = 1000
    triplets, error = getTriplets(X, pulls)

    X0 = np.random.random((n,d))
    stats4= triplet_algorithms(ste_loss, 
                           triplets,
                           X0,                       
                           d,
                           'full_grad', 
                           10,
                           iters=5000,
                           epsilon = 0.01,
                           proj=None,
                           debug=True
                          )


