import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
from Utils import *
from algorithms_full import *
from new_utils import *
from eigenEmbed import*
import json

def save(filename, stats, train_set, test_set):
  total_dict = {'train_set': train_set,
                'test_set':  test_set,
                'stats': stats}
  with open(filename, 'w') as f:
    json.dump(total_dict, f)
    return

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
np.random.seed(42)
dimensions= 5
number_of_points= 100
noise = True

X = np.random.random((number_of_points, dimensions))
X = center_data(X)

n,d = X.shape
pulls = int(12.5*number_of_points*dimensions*np.log(number_of_points))
# split percentage fraction of triplets are train:
# the rest are train
split_percentage = 0.8

# NOISE RUINS EVERYTHING, talk to blake about how to measure acc in this case
# winner, loser, head
tot_triplets, error = getTriplets(X, int(pulls), noise=noise) 
print('Dataset error: ', error)

triplets = tot_triplets[:int(len(tot_triplets)*split_percentage)]
test_triplets = tot_triplets[int(len(tot_triplets)*split_percentage):]

print('Dimensions:{} \nNumber of points:{} \nPulls:{}'.format(dimensions,number_of_points, len(tot_triplets)))
print()
print('TRAIN: ', len(triplets))
print('TEST: ', len(test_triplets))

X0 = np.random.random((n,d))

# different algorithms to compute:

# Non Convex FULL GD
# print('Non convex full GD')
# stats_non_convex_single_exp_full_gd = triplet_algorithms(ste_loss, 
#                            triplets,
#                            X0,                       
#                            d,
#                            'full_grad', 
#                            150,
#                            iters=1000,
#                            epsilon = 2e-5,
#                            proj=None,
#                            debug=True
#                           )
# save('./outputs/BurerMonteiro_no_noise.json', stats_non_convex_single_exp_full_gd, triplets, test_triplets)

# print('Convex STE')
# M0 = X0 @ X0.T
# stats_convex_single_exp_full_gd = triplet_algorithms(ste_loss_convex, 
#                        triplets,
#                        M0,                       
#                        d,                            
#                        'full_grad', 
#                         600,
#                        iters=200,
#                        epsilon =3e-7,
#                        proj=projected_psd,
#                        debug= True)
# save('./outputs/STE_no_noise.json', stats_convex_single_exp_full_gd, triplets, test_triplets)

# rank-D projection
# print('Rank D projection')
# Xhat, stats_rankD = eigen_embed(X0, triplets, alpha=800, method='rankD', epsilon=8.3e-7, debug=True)
# save('./outputs/rankD_no_noise.json', stats_rankD, triplets, test_triplets)

# Frank-Wolfe method
# print('Frank Wolfe')
# Xhat, stats_FW = eigen_embed(X0, triplets, alpha=1, method='FW', epsilon=1.1e-6, debug=True)
# save('./outputs/FW_no_noise.json', stats_FW, triplets, test_triplets)

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

