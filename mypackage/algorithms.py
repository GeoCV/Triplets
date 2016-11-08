from ste import *
from new_utils import *
import time


def triplet_algorithms(f, 
                       S,
                       X0,
                       d,
                       descent_alg, 
                       step_size_func,
                       iters=100,
                       epsilon = 0.001,
                       proj=None):

    '''
    File contains the main triplets algorithm [still needs work]


    :param f: objective function 
    :param S: set of triplets
    :param X0: starting iterate
    :param d: ambient dimesnion
    :param descent_alg: choice for descent method (e.g. full_gradient/sgd etc.)
    :param iters: number of iterations
    :param epsilon: Accuracy parameter
    :type f: python function with specific structure
    :type S: list of lists
    :type X0: a d dimensional vector
    :type d: int
    :type descent_alg: function
    :type iters: int
    :type epsilon: int
    '''    
    
    stats = {}    
    stats['emp'] = []
    stats['log'] = []
    stats['time_per_iter'] = []
    stats['avg_time_per_iter'] = 0    
    stats['status'] = 0
    
    X_curr = X0
    emp_X_curr = f(X0, S, 1)['empirical_loss']
    log_X_curr = f(X0, S, 1)['log_loss']

    print(emp_X_curr)
    stats['emp'].append(emp_X_curr)
    stats['log'].append(log_X_curr)

    
    # accuracy achieved
    if stats['emp'][-1] < epsilon:
        return stats
            
    for i in range(iters):    
        
        start = time.time()
        
        p = f(X_curr, S, 2)[descent_alg] 
        alpha = step_size_func #  for now constant stepsize only 
        
        try:
            X_new = X_curr + alpha*p
            
            if proj != None:
                X_new = proj(X_new, d)
            
            emp_X_new = f(X_new, S, 1)['empirical_loss']
            log_X_new = f(X_new, S, 1)['log_loss']
            
        except OverflowError:
            print('Step size too big: math range overflow',i)
            stats['status'] = -1
            break
            
        stats['emp'].append(emp_X_new)
        stats['log'].append(log_X_new)

        # accuracy achieved
        if stats['emp'][-1] < epsilon:
            print('Accuracy reached in {} iterations'.format(i))
            break
            
        # gradient is very very small
        if descent_alg == 'full_grad' or descent_alg == 'sgd':
            if norm(p) < 10**-6:
                print('Gradient too small')
                break
                
        # No substantial increase in last ten guys
        # has to run for atleast 10 iterations
        if i > 10:
            smallest = min(stats['log'][::-1][:10])
            biggest = max(stats['log'][::-1][:10])                        
            if abs(smallest - biggest) < 10**-5:
                print('No progress')
                break
        
        # divergence 
            smallest = min(stats['log'][::-1][:3])
            biggest = max(stats['log'][::-1][:3])                        
            if abs(smallest - biggest) > 1:
                print('Divergence')
                stats['status'] = -1
                break
        
        end = time.time()
        stats['time_per_iter'].append((end - start))
        X_curr = X_new
    
    stats['avg_time_per_iter'] = sum(stats['time_per_iter'])/(i+1)
    stats['embedding'] = X_curr
    
    return stats        
