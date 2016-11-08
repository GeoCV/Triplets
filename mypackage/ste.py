from math import exp, log
from numpy.linalg import norm
from numpy import zeros

from new_utils import *

def ste_loss_triplet(X, q, opt):

    # Logistic loss of triplet score
    if opt == 1:
        triplet_score = scoreX(X, q, opt) 
        return logistic_loss(triplet_score, opt)
    
    # Gradient
    elif opt == 2:        
        triplet_score = scoreX(X, q, 1)   
        return logistic_loss(triplet_score, opt)*scoreX(X,q, opt)

def ste_loss(X, S, opt):
    
    if opt == 1:
        emp_loss = 0
        log_loss = 0
        for q in S:        
            triplet_score = scoreX(X,q,1)        
            if triplet_score > 0:
                emp_loss +=1

            log_loss += ste_loss_triplet(X,q,1)

        avg_emp_loss = emp_loss/len(S)
        avg_log_loss = log_loss/len(S)

        return {'empirical_loss': avg_emp_loss,
                'log_loss': avg_log_loss
           }

    elif opt == 2:

        full_grad = fullGD_X(ste_loss_triplet, X, S)
        return {'full_grad': full_grad
           }

    
def ste_loss_triplet_gram(M, q, opt):

    # Logistic loss of triplet score
    if opt == 1:
        triplet_score = scoreM(M, q, opt) 
        return logistic_loss(triplet_score, opt)
    
    # Gradient
    elif opt == 2:        
        triplet_score = scoreM(M, q, 1)   
        return logistic_loss(triplet_score, opt)*scoreM(M,q, opt)
    

def ste_loss_convex(M, S, opt):

    if opt == 1:
        emp_loss = 0
        log_loss = 0
        for q in S:        

            triplet_score = scoreM(M,q,1)        

            if triplet_score > 0:
                emp_loss +=1

            log_loss += ste_loss_triplet_gram(M,q,1)

        avg_emp_loss = emp_loss/len(S)
        avg_log_loss = log_loss/len(S)

        return {'empirical_loss': avg_emp_loss,
                'log_loss': avg_log_loss
           }
    

    elif opt == 2:
        full_grad = fullGD_X(ste_loss_triplet_gram, M, S)
        return {'full_grad': full_grad
           }
