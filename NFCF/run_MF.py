# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:25:15 2019

@author: islam
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score

from utilities import get_instances_with_random_neg_samples, get_test_instances_with_random_samples
from performance_and_fairness_measures import getHitRatio, getNDCG, differentialFairnessMultiClass, computeEDF, computeAbsoluteUnfairness

import math
import heapq # for retrieval topK

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from collaborative_models import matrixFactorization

#%%The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% pre-training NCF model with user-page pairs
def train_epochs(model,df_train, epochs, lr, batch_size,num_negatives, unsqueeze=False):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model.train()
    for i in range(epochs):
        j = 0
        for batch_i in range(0,np.int64(np.floor(len(df_train)/batch_size))*batch_size,batch_size):
            data_batch = (df_train[batch_i:(batch_i+batch_size)]).reset_index(drop=True)
            #train_user_input, train_item_input, train_ratings = get_instances_with_neg_samples(data_batch, probabilities, num_negatives,device)
            train_user_input, train_item_input, train_ratings = get_instances_with_random_neg_samples(data_batch, num_uniqueLikes, num_negatives,device)
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)
            y_hat = model(train_user_input, train_item_input)
            y_hat_prime = y_hat.reshape(len(y_hat), 1)
            loss = criterion(y_hat_prime, train_ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch: ', i, 'batch: ', j, 'out of: ',np.int64(np.floor(len(df_train)/batch_size)), 'average loss: ',loss.item())
            j = j+1


#%% model evaluation: hit rate and NDCG
def evaluate_model(model,df_val,top_K,random_samples, num_items):
    model.eval()
    avg_HR = np.zeros((len(df_val),top_K))
    avg_NDCG = np.zeros((len(df_val),top_K))
    
    for i in range(len(df_val)):
        test_user_input, test_item_input = get_test_instances_with_random_samples(df_val[i], random_samples,num_items,device)
        y_hat = model(test_user_input, test_item_input)
        y_hat = y_hat.cpu().detach().numpy().reshape((-1,))
        test_item_input = test_item_input.cpu().detach().numpy().reshape((-1,))
        map_item_score = {}
        for j in range(len(y_hat)):
            map_item_score[test_item_input[j]] = y_hat[j]
        for k in range(top_K):
            # Evaluate top rank list
            ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
            gtItem = test_item_input[0]
            avg_HR[i,k] = getHitRatio(ranklist, gtItem)
            avg_NDCG[i,k] = getNDCG(ranklist, gtItem)
    avg_HR = np.mean(avg_HR, axis = 0)
    avg_NDCG = np.mean(avg_NDCG, axis = 0)
    return avg_HR, avg_NDCG

def fairness_measures(model, df_val, num_items, protectedAttributes,subgroup):
    model.eval()
    user_input = torch.LongTensor(df_val['user_id'].values).to(device)
    item_input = torch.LongTensor(df_val['like_id'].values).to(device)
    y_hat = model(user_input, item_input)

    avg_epsilon = computeEDF(protectedAttributes, y_hat, num_items, item_input, device)
    U_abs = computeAbsoluteUnfairness(protectedAttributes, y_hat, num_items, item_input, device)

    avg_epsilon = avg_epsilon.cpu().detach().numpy().reshape((-1,)).item()
    U_abs = U_abs.cpu().detach().numpy().reshape((-1,)).item()
    if subgroup == 'age':
        print(f"average differential fairness for age: {avg_epsilon: .3f}")
        print(f"absolute unfairness for age: {U_abs: .3f}")
    else:
        print(f"average differential fairness for gender: {avg_epsilon: .3f}")
        print(f"absolute unfairness for gender: {U_abs: .3f}")

    return avg_epsilon, U_abs

#probabilities = np.loadtxt('train-test/distributionLikes.txt') # frequency distribution of likes
#%% load data

#%% set hyperparameters
emb_size = 128
hidden_layers = np.array([emb_size, 64, 32, 16])
output_size = 1
num_epochs = 25
learning_rate = 0.001
batch_size = 2048 
num_negatives = 5

random_samples = 100
top_K = 10

train_data = pd.read_csv("train-test/train_userPages.csv", encoding='utf-8')
test_data = pd.read_csv("train-test/test_userPages.csv")
num_uniqueUsers = len(train_data.user_id.unique())
num_uniqueLikes = len(train_data.like_id.unique())


#%% start training the NCF model

MF = matrixFactorization(num_uniqueUsers, num_uniqueLikes, emb_size).to(device)
#MF.load_state_dict(torch.load('trained-models/MF'))

train_epochs(MF,train_data,num_epochs,learning_rate,batch_size,num_negatives,unsqueeze=True)
torch.save(MF.state_dict(), "trained-models/MF")

#%% evaluate the model
train_users= pd.read_csv("train-test/train_usersID.csv",names=['user_id'])
test_users = pd.read_csv("train-test/test_usersID.csv",names=['user_id'])

train_careers= pd.read_csv("train-test/train_concentrationsID.csv",names=['like_id'])
test_careers = pd.read_csv("train-test/test_concentrationsID.csv",names=['like_id'])

train_protected_attributes= pd.read_csv("train-test/train_protectedAttributes.csv")
test_protected_attributes = pd.read_csv("train-test/test_protectedAttributes.csv")

# =============================================================================
# train_labels= pd.read_csv("train-test/train_labels.csv",names=['labels'])
# test_labels = pd.read_csv("train-test/test_labels.csv",names=['labels'])
#
# unique_concentrations = (pd.concat([train_careers['like_id'],train_labels['labels']],axis=1)).reset_index(drop=True)
# unique_concentrations = unique_concentrations.drop_duplicates(subset='like_id', keep='first')
#
# unique_careers = unique_concentrations.sort_values(by=['like_id']).reset_index(drop=True)
# unique_careers.to_csv('train-test/unique_careers.csv',index=False)
# =============================================================================

train_data = (pd.concat([train_users['user_id'],train_careers['like_id']],axis=1)).reset_index(drop=True)
test_data = (pd.concat([test_users['user_id'],test_careers['like_id']],axis=1)).reset_index(drop=True)
num_uniqueCareers = len(train_data.like_id.unique())

test_gender = test_protected_attributes['gender'].values
test_age = test_protected_attributes['age_group'].values

HR_array, NDCG_array = [], []
for topk in [1,5,7,10]:
    HR, NDCG = [], []
    for time in range(10):
        avg_HR_fineTune, avg_NDCG_fineTune = evaluate_model(MF,test_data.values,topk,random_samples, num_uniqueCareers)
        HR.append(avg_HR_fineTune)
        NDCG.append(avg_NDCG_fineTune)
    final_HR, final_NDCG = np.mean(HR), np.mean(NDCG)
    HR_array.append(topk)
    HR_array.append(final_HR)
    NDCG_array.append(topk)
    NDCG_array.append(final_NDCG)

epsilon_age, abs_age, epsilon_gender, abs_gender = [], [], [], []
for i in range(10):
    avg_epsilon_gender, U_abs_gender = fairness_measures(MF,test_data,num_uniqueCareers,test_gender,'gender')
    avg_epsilon_age, U_abs_age = fairness_measures(MF,test_data,num_uniqueCareers,test_age,'age')
    epsilon_age.append(avg_epsilon_age)
    epsilon_gender.append(avg_epsilon_gender)
    abs_age.append(U_abs_age)
    abs_gender.append(U_abs_gender)
avg_epsilon_gender = np.mean(epsilon_gender)
avg_epsilon_age = np.mean(epsilon_age)
U_abs_gender = np.mean(abs_gender)
U_abs_age = np.mean(abs_age)

np.savetxt('results/avg_HR_MF.txt',[HR_array])
np.savetxt('results/avg_NDCG_MF.txt',[NDCG_array])
np.savetxt('results/gender/avg_epsilon_MF.txt',[avg_epsilon_gender])
np.savetxt('results/gender/U_abs_MF.txt',[U_abs_gender])
np.savetxt('results/age/avg_epsilon_MF.txt',[avg_epsilon_age])
np.savetxt('results/age/U_abs_MF.txt',[U_abs_age])
#sys.stdout.close()