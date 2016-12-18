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

import matplotlib.pyplot as plt 
import seaborn as sns

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

def getTime(d):
    n = len(d['stats']['emp'])
    return [sum(d['stats']['time_per_iter'][:i]) for i in range(n)]

BM_noise = json.loads(open('./outputs/BurerMonteiro_noise.json', 'r').read())
BM_no_noise = json.loads(open('./outputs/BurerMonteiro_no_noise.json', 'r').read())

STE_noise = json.loads(open('./outputs/STE_noise.json', 'r').read())
STE_no_noise = json.loads(open('./outputs/STE_no_noise.json', 'r').read())

rankD_noise = json.loads(open('./outputs/rankD_noise.json', 'r').read())
rankD_no_noise = json.loads(open('./outputs/rankD_no_noise.json', 'r').read())

FW_noise = json.loads(open('./outputs/FW_noise.json', 'r').read())
FW_no_noise = json.loads(open('./outputs/FW_no_noise.json', 'r').read())

########################################################### Error plots with noise ###########################################################
# empirical loss plots with noise:
df = pd.DataFrame([
                    BM_noise['stats']['emp'][:40],
                    rankD_noise['stats']['emp'],
                   STE_noise['stats']['emp'],                    
                   FW_noise['stats']['emp']                 
                  ]).T

df.columns = [
              'Non convex: Factored full gradient descent',
              'Non convex: Rank D projection',
              'Convex: STE', 
              'Convex: Frank-Wolfe'              
             ]

ax = df.iloc[:1000].plot(figsize=(18,8), fontsize=18)
ax.set_ylabel('Empirical loss', fontsize=22)
ax.set_xlabel('Iterations', fontsize=22)
ax.set_title('Empirical loss per iteration across embedding methods with noisy data', fontsize=22) 
plt.axhline(y=0.365, label='True training error')
ax.legend(fontsize=18, loc='best')  
plt.show()


# Logistic loss plots with noise:
df = pd.DataFrame([
                    BM_noise['stats']['log'][:40],
                    rankD_noise['stats']['log'],
                   STE_noise['stats']['log'],                    
                   FW_noise['stats']['log']                 
                  ]).T

df.columns = [
              'Non convex: Factored full gradient descent',
              'Non convex: Rank D projection',
              'Convex: STE', 
              'Convex: Frank-Wolfe'              
             ]

ax = df.iloc[:1000].plot(figsize=(18,8), fontsize=18)
ax.set_ylabel('Logistic loss', fontsize=22)
ax.set_xlabel('Iterations', fontsize=22)
ax.set_title('Logistic loss per iteration across embedding methods with noisy data', fontsize=22) 
plt.axhline(y=0.638, label='True training error')
ax.legend(fontsize=18, loc='best')  
plt.show()

########################################################### Error plots without noise ###########################################################
# empirical loss without noise
df = pd.DataFrame([
                    BM_no_noise['stats']['emp'][:40],
                    rankD_no_noise['stats']['emp'][:40],
                   STE_no_noise['stats']['emp'],                    
                   FW_no_noise['stats']['emp']                 
                  ]).T

df.columns = [
              'Non convex: Factored full gradient descent',
              'Non convex: Rank D projection',
              'Convex: STE', 
              'Convex: Frank-Wolfe'              
             ]

ax = df.iloc[:1000].plot(figsize=(18,8), fontsize=18)
ax.set_ylabel('Empirical loss', fontsize=22)
ax.set_xlabel('Iterations', fontsize=22)
ax.set_title('Empirical loss per iteration across embedding methods with clean data', fontsize=22) 
ax.legend(fontsize=18, loc='best')  
# ax.set_ylim([0, 1.5])
plt.show()


# Logistic loss plots without noise:
df = pd.DataFrame([
                    BM_no_noise['stats']['log'][:40],
                    rankD_no_noise['stats']['log'][:40],
                   STE_no_noise['stats']['log'],                    
                   FW_no_noise['stats']['log']                 
                  ]).T

df.columns = [
              'Non convex: Factored full gradient descent',
              'Non convex: Rank D projection',
              'Convex: STE', 
              'Convex: Frank-Wolfe'              
             ]

ax = df.iloc[:1000].plot(figsize=(18,8), fontsize=18)
ax.set_ylabel('Logistic loss', fontsize=22)
ax.set_xlabel('Iterations', fontsize=22)
ax.set_title('Logistic loss per iteration across embedding methods with clean data', fontsize=22) 
ax.legend(fontsize=18, loc='best')  
ax.set_ylim([0, 1.])
plt.show()

########################################################### Time Error plots with noise ###########################################################
# empirical loss
plt.figure(1)
plt.plot(getTime(BM_noise)[:40], BM_noise['stats']['emp'][:40], label='Non convex: Factored full gradient descent')
plt.plot(getTime(rankD_noise), rankD_noise['stats']['emp'], 'r', label='Non convex: Rank D projection')
plt.plot(getTime(STE_noise), STE_noise['stats']['emp'], 'g', label='Convex: STE')
plt.plot(getTime(FW_noise), FW_noise['stats']['emp'], 'k', label='Convex: Frank Wolfe')
plt.axhline(y=0.365, label='True training error')
plt.xlabel('Time (sec)', fontsize=22)
plt.ylabel('Empirical Loss', fontsize=22)
plt.title('Empirical loss versus time across embedding methods with noisy data', fontsize=22)
plt.legend(fontsize=18, loc='best')
plt.show()

# logistic loss
plt.figure(2)
plt.plot(getTime(BM_noise)[:40], BM_noise['stats']['log'][:40], label='Non convex: Factored full gradient descent')
plt.plot(getTime(rankD_noise), rankD_noise['stats']['log'], 'r', label='Non convex: Rank D projection')
plt.plot(getTime(STE_noise), STE_noise['stats']['log'], 'g', label='Convex: STE')
plt.plot(getTime(FW_noise), FW_noise['stats']['log'], 'k', label='Convex: Frank Wolfe')
plt.axhline(y=0.638, label='True training error')
plt.xlabel('Time (sec)', fontsize=22)
plt.ylabel('Logistic Loss', fontsize=22)
plt.title('Logistic loss versus time across embedding methods with noisy data', fontsize=22)
plt.legend(fontsize=18, loc='best')
plt.show()

########################################################### Time Error plots without noise ###########################################################
# empirical loss
plt.figure(3)
plt.plot(getTime(BM_no_noise)[:40], BM_no_noise['stats']['emp'][:40], label='Non convex: Factored full gradient descent')
plt.plot(getTime(rankD_no_noise), rankD_no_noise['stats']['emp'], 'r', label='Non convex: Rank D projection')
plt.plot(getTime(STE_no_noise), STE_no_noise['stats']['emp'], 'g', label='Convex: STE')
plt.plot(getTime(FW_no_noise), FW_no_noise['stats']['emp'], 'k', label='Convex: Frank Wolfe')
plt.xlabel('Time (sec)', fontsize=22)
plt.ylabel('Empirical Loss', fontsize=22)
plt.title('Empirical loss versus time across embedding methods with clean data', fontsize=22)
plt.legend(fontsize=18, loc='best')
plt.show()

# logistic loss
plt.figure(4)
plt.plot(getTime(BM_no_noise)[:40], BM_no_noise['stats']['log'][:40], label='Non convex: Factored full gradient descent')
plt.plot(getTime(rankD_no_noise)[:40], rankD_no_noise['stats']['log'][:40], 'r', label='Non convex: Rank D projection')
plt.plot(getTime(STE_no_noise), STE_no_noise['stats']['log'], 'g', label='Convex: STE')
plt.plot(getTime(FW_no_noise), FW_no_noise['stats']['log'], 'k', label='Convex: Frank Wolfe')
plt.xlabel('Time (sec)', fontsize=22)
plt.ylabel('Logistic Loss', fontsize=22)
plt.title('Logistic loss versus time across embedding methods with clean data', fontsize=22)
plt.legend(fontsize=18, loc='best')
plt.ylim(ymax=1)
plt.show()