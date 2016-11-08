import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from numpy import dot
from numpy.random import randint
from numpy.linalg import norm

import random
import time

'''
A file with helper functions common accross
all algorithms
'''

def center_data(X):
    '''
    Given a matrix of coordinates X, center the matrix around 0

    :param X: Matrix to be centred. 
    :type X: nxd numpy array
    
    Usage:
    X = center_data(X)

    '''
    n,d = X.shape
    
    # this is just mean subtraction in each coordinate
    X = X - 1./n*np.dot(np.ones((n,n)), X)
    
    return X

def getTripletScore(X,q):
    """
    Given X,q=[i,j,k] returns $$score = ||x_i - x_k||^2 - ||x_j - x_k||^2$$

    If score < 0 then the triplet agrees with the embedding, otherwise it does not 
    i.e i is closer to k than k

    Usage:
        score = getTripletScore(X,[3,4,5])
    """
    i,j,k = q

    return dot(X[i],X[i]) -2*dot(X[i],X[k]) + 2*dot(X[j],X[k]) - dot(X[j],X[j])


def getTriplets(X,pulls,shift=1,steepness=1,noise=False):
    """
    Generate a random set of #pulls triplets
    Params:
    X    : The true embedding that preserves all the triplet comparisons
    pull : number of triplets comparison
    shift : first measure noise (click bias)
    steepness : second measure of noise 
    
    Returns: 
    S : list of lists of indices which represent triplets as
        [i,j,k] - i winner(closer to k), j is loser(further from k) 
        and center.
        
    error: In noise model are the percentage of bad triplets. In the noiseless
           case it should be 0.
    """    
    S = []
    n,d = X.shape
    error = 0.
    for i in range(0,pulls):
        # get random triplet
        q, score = getRandomQuery(X)

        # align it so it agrees with Xtrue: "q[2] is more similar to q[0] than q[1]"
        if score > 0:
            q = [q[i] for i in [1,0,2]]
        # add some noise
        if noise:
            if rand() > 1/(shift+exp(steepness*getTripletScore(X,q))):
                q = [ q[i] for i in [1,0,2]]
                error+=1
        S.append(q)   
    error /= float(pulls)
    
    return S,error


def getRandomQuery(X):
    """
    Outputs a triplet [i,j,k] chosen uniformly at random from all possible triplets \
    and score = abs( ||x_i - x_k||^2 - ||x_j - x_k||^2 )
    
    Inputs:
        n (integer) : total number of points in emebedding
    Outputs:
        [(int) i, (int) j, (int) k] q : where k in [n], i in [n]-k, j in [n]-k-j        
    Usage:
        q = getRandomQuery(X)       
    """

    n,d = X.shape    
    i = randint(n)
    j = randint(n)
    
    while (j==i):
        j = randint(n)
    k = randint(n)
    while (k==i) | (k==j):
        k = randint(n)
    q = [i, j, k]

    score = getTripletScore(X,q)
    
    return q, score




def procrustes(X, Y, scaling=True, reflection='best'):
    """
    http://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()
    # centred Frobenius norm
    normX = sqrt(ssX)
    normY = sqrt(ssY)
    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY
    print(n,m)
    print(ny,my)
    if my < m:
        Y0 = concatenate((Y0, zeros(n, m-my)),0)
    # optimum rotation matrix of Y
    A = dot(X0.T, Y0)
    U,s,Vt = linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = dot(V, U.T)

    if reflection is not 'best':
        # does the current solution use a reflection?
        have_reflection = linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*dot(Y0, T) + muX
    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*dot(muY, T)
    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}
    return d, Z, tform


def twodplot(X, Y):
    n,d = X.shape
    plt.figure(1)
    # Plot Xtrue
    plt.subplot(131)
    plt.axvline(x=pow(n,-1./3))
    plt.axhline(y=pow(n,-1./3))
    plt.plot(*zip(*X), marker='o', color='r', ls='')
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    
    # Plot Xhat
    plt.subplot(132)
    plt.axvline(x=pow(n,-1./3))
    plt.axhline(y=pow(n,-1./3))
    plt.plot(*zip(*Y), marker='o', color='b', ls='')
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    
    # Overlap plot
    plt.subplot(133)
    plt.axvline(x=pow(n,-1./3))
    plt.axhline(y=pow(n,-1./3))
    plt.plot(*zip(*X), marker='o', color='r', ls='')
    #plt.xlim([0,1])
    #plt.ylim([0,1])

    for i in range(n):
        point = X[i,:].tolist()
        if d==1:
            point = [point[0],0]
        plt.annotate(str(i),
                     textcoords='offset points',
                     xy=(point[0], point[1]),
                     xytext = (-5, 5),
                     ha = 'right',
                     va = 'bottom',
                     color='red',
                     arrowprops = dict(arrowstyle = '-',
                                       connectionstyle = 'arc3,rad=0'))
    plt.plot(*zip(*Y), marker='o', color='b', ls='')
    for i in range(n):
        point = Y[i,:].tolist()
        if d==1:
            point = [point[0],0]
        plt.annotate(str(i),
                     textcoords='offset points',
                     xy=(point[0], point[1]),
                     xytext = (-5, -5),
                     ha = 'right',
                     va = 'bottom',
                     color='blue',
                     arrowprops = dict(arrowstyle = '-',
                                       connectionstyle = 'arc3,rad=0'))
    

def onedplot(Xtrue, Xhat):
    n = len(Xtrue)
    Xtrue = Xtrue[:, 0]
    Xhat = Xhat[:, 0]
    print(Xtrue)
    # plt.stem(ans, [1]*n)
    # plt.stem(pts, [1]*n, markerfmt='ro', linefmt='r')
    #plt.show()

    plt.figure(1)
    
    # Plot Xtrue
    plt.subplot(131)
    plt.stem(Xtrue, [1]*n, markerfmt='ro', linefmt='r')
    plt.axis([0,1,0,2])
    # Plot Xhat
    plt.subplot(132)
    plt.stem(Xhat, [1]*n, markerfmt='bo', linefmt='b')
    plt.axis([0,1,0,2])
    # Overlap plot
    plt.subplot(133)
    plt.stem(Xtrue, [1]*n, markerfmt='ro', linefmt='r')
    plt.axis([0,1,0,2])
    for i in range(n):
        plt.annotate(str(i),
                     textcoords='offset points',
                     xy=(Xtrue[i], 0),
                     xytext = (-5, 5),
                     ha = 'right',
                     va = 'bottom',
                     color='red',
                     arrowprops = dict(arrowstyle = '-',
                                       connectionstyle = 'arc3,rad=0'))
    plt.stem(Xhat, [1]*n, markerfmt='bo', linefmt='b')
    plt.axis([0,1,0,2])
    for i in range(n):
        plt.annotate(str(i),
                     textcoords='offset points',
                     xy=(Xhat[i], 1),
                     xytext = (-5, -5),
                     ha = 'right',
                     va = 'bottom',
                     color='blue',
                     arrowprops = dict(arrowstyle = '-',
                                       connectionstyle = 'arc3,rad=0'))
    plt.show()

