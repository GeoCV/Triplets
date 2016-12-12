import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import pandas as pd
from Utils import *

from algorithms import *
from new_utils import *

from eigenEmbed import*

def predictX(X_true, X_predict, test_triplets):
    
    true = []
    pred = []
    for q in test_triplets:
        
        true_label = np.sign(scoreX(X_true, q, 1))
        pred_label = np.sign(scoreX(X_predict, q, 1))
        
        true.append(true_label)
        pred.append(pred_label)
    
    acc = accuracy_score(true, pred)
    print("Correctly predicted {0:.2f} % of the unseen triplets".format(acc*100))    
    
    return acc

def predictM(M_true, M_predict, test_triplets):
    
    true = []
    pred = []
    for q in test_triplets:
        
        true_label = np.sign(scoreM(M_true, q, 1))
        pred_label = np.sign(scoreM(M_predict, q, 1))
        
        true.append(true_label)
        pred.append(pred_label)
    
    acc = accuracy_score(true, pred)
    print("Correctly predicted {0:.2f} % of the unseen triplets".format(acc*100))    
    
    return acc

#Create data
dimensions= 3
number_of_points= 40

X = np.random.random((number_of_points, dimensions))
X = center_data(X)

n,d = X.shape
pulls = 10*int(number_of_points*dimensions*np.log(number_of_points))
# split percentage fraction of triplets are train:
# the rest are train
split_percentage = 0.8

# NOISE RUINS EVERYTHING, talk to blake about how to measure acc in this case
# winner, loser, head
tot_triplets, error = getTriplets(X, int(pulls), noise=True) 
print('Dataset error: ', error)

triplets = tot_triplets[:int(len(tot_triplets)*split_percentage)]
test_triplets = tot_triplets[int(len(tot_triplets)*split_percentage):]

print('Dimensions:{} \nNumber of points:{} \nPulls:{}'.format(dimensions,number_of_points, len(tot_triplets)))
print()
print('TRAIN: ', len(triplets))
print('TEST: ', len(test_triplets))

epsilons = np.linspace(0.005, 0.1,5)[::-1]

# different algorithms to compute:

# Non Convex FULL GD
print('Non convex full GD')
exp = -1
X0 = np.random.random((n,d))
stats_non_convex_single_exp_full_gd = triplet_algorithms(ste_loss, 
                           triplets,
                           X0,                       
                           d,
                           'full_grad', 
                           50,
                           iters=200,
                           epsilon = epsilons[exp],
                           proj=None,
                           debug=True
                          )

# Non Convex SGD
print('Non convex sgd')
stats_non_convex_single_exp_sgd = triplet_algorithms(ste_loss, 
                           triplets,
                           X0,                       
                           d,
                           'sgd', 
                           0.2,
                           iters=5000,
                           epsilon = epsilons[exp],
                           proj=None,
                           debug=True
                          )

# Non Convex SVRG
print('Non cnvex svrg')
stats_non_convex_single_exp_svrg = triplet_algorithms(ste_loss, 
                           triplets,
                           X0,                       
                           d,
                           'svrg', 
                           0.2,
                           iters=5000,
                           epsilon = epsilons[exp],
                           proj=None,
                           debug=True
                          )

# CONVEX STE FULL GRAD Single experiment
print('Convex STE')
M0 = X0 @ X0.T
stats_convex_single_exp_full_gd = triplet_algorithms(ste_loss_convex, 
                       triplets,
                       M0,                       
                       d,                            
                       'full_grad', 
                        800,
                       iters=200,
                       epsilon =epsilons[exp],
                       proj=projected_psd,
                       debug= True)

# CONVEX STE SGD Single experiment
print('Convex SGD')
stats_convex_single_exp_sgd = triplet_algorithms(ste_loss_convex, 
                       triplets,
                       M0,                       
                       d,                            
                       'sgd', 
                        1,
                       iters=5000,
                       epsilon =epsilons[exp],
                       proj=projected_psd,
                       debug= True)
    
# CONVEX STE SVRG Single experiment
print('Convex SVRG')
stats_convex_single_exp_svrg = triplet_algorithms(ste_loss_convex, 
                       triplets,
                       M0,                       
                       d,                            
                       'svrg', 
                        100,
                       iters=5000,
                       epsilon =epsilons[exp],
                       proj=projected_psd,
                       debug= True)

# rank-D projection
print('Rank D projection')
Xhat, stats_rankD = eigen_embed(X0, triplets, method='rankD', epsilon=epsilons[exp], debug=True)

# Frank-Wolfe method
print('Frank Wolfe')
Xhat, stats_FW = eigen_embed(X0, triplets, method='FW', epsilon=epsilons[exp], debug=True)

# PLOTTING CODE

# df = pd.DataFrame([
#                 stats_non_convex_single_exp_full_gd['emp'],
#                 stats_non_convex_single_exp_sgd['emp'],
#                 stats_non_convex_single_exp_svrg['emp'],
#                 stats_rankD['emp'],
#                    stats_convex_single_exp_full_gd['emp'],                    
#                    stats_convex_single_exp_svrg['emp'],
#                    stats_convex_single_exp_sgd['emp'],  
#                    stats_FW['emp']                 
#                   ]).T

# df.columns = [
#               'Non convex: Factored full gradient descent',
#               'Non convex: Factored SGD',
#               'Non convex: Factored SVRG',
#               'Non convex: Rank D projection',
#               'Convex: Projection onto PSD cone full gradient descent', 
#               'Convex: Projection onto PSD cone SVRG',
#               'Convex: Projection onto PSD cone SGD',
#               'Convex: Frank Wolfe'
              
#              ]

# ax = df.iloc[:1000].plot(figsize=(18,8), fontsize=18)
# ax.set_ylabel('0-1 loss', fontsize=22)
# ax.set_xlabel('Epochs', fontsize=22)
# ax.set_title('Epsilon= {}'.format(epsilons[exp]), fontsize=22)
# ax.legend(fontsize=18, loc='center left', bbox_to_anchor=(1, 0.5));

