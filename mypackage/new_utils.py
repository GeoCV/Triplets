from __future__ import division
from math import exp, log
from numpy.linalg import norm, eigh
from numpy import mat, dot, zeros, diag
from numpy.random import randint
#-------------------------------------------------------------------
# Basic loss and scoring functions
#-------------------------------------------------------------------

def logistic_loss(x, opt):
    
    if opt == 1:

        loss = log(1 + exp(x))            
            
        return loss
    
    
    # Gradient
    elif opt == 2:
        return 1/(1+ exp(-1*x))    

def scoreX(X, q, opt):
    
    i,j,k = q[0], q[1], q[2]
    
    if opt == 1:
        
        return norm(X[i] - X[k])**2 - norm(X[j] - X[k])**2
    
    elif opt == 2:        
        n, d = X.shape
        G = zeros((n,d))
        
        G[i] = 2*(X[i] - X[k])
        G[j] = 2*(X[k] - X[j])
        G[k] = 2*(X[j] - X[i])

        return G

def scoreM(M, q, opt):

    i,j,k = q[0], q[1], q[2]

    if opt == 1:
        return M[i,i] -2*M[i,k] + 2*M[j,k] - M[j,j]
    
    elif opt == 2:        
        n = M.shape[0]
        # pattern for computing gradient
        H = mat([[1.,0.,-1.],
                 [ 0.,  -1.,  1.],
                 [ -1.,  1.,  0.]])
        # compute gradient 
        G = zeros((n,n))
        G[[[x] for x in q],q] = H
        return G

def projected(M, d):
    '''
    Project onto rank d psd matrices
    '''
    n, n = M.shape
    D, V = eigh(M)
    perm = D.argsort()
    bound = max(D[perm][-d], 0)
    for i in range(n):
        if D[i] < bound:
            D[i] = 0
    M = dot(dot(V,diag(D)),V.transpose());
    return M

#--------------------------------------------------
# Descent methods
#--------------------------------------------------

def fullGD_X(f, X, S):   
    
    n,d = X.shape
    G = zeros((n,d))    
    for q in S:
        G += f(X, q, 2)
    return -1*G/len(S)  


def SGD_X(f, X, S):

    ind = randint(len(S))

    q = S[ind]

    return -1*f(X, q, 2)


def SVRG_X(f, X, S, full_grad=None, y=None):

    '''
    y: The time I did full gradient descent, that point
    '''
    ind = randint(len(S))

    q = S[ind]

    return -1*(f(X, q, 2) - f(y, q, 2) + full_grad)
    

