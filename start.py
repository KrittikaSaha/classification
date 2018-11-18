#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:00:32 2018

@author: alaridatascience
"""


#cd /home/alaridatascience/pCloudDrive/WORK/fsecure
!conda list 
#print(pd.read_csv.__doc__) 
#print(pd.concat.__doc__)

# HIGH DIMENSIONAL BINARY CLASSIFICATION
# if you need 10 points in one dimension and you have p features, then you need power(10,p) points 

def readInNumpy(traindir="train_data.csv",train_labels="train_labels.csv",testdir="test_data.csv"):
    import numpy as np
    train_numpy = np.genfromtxt(traindir, delimiter=',')
    train_labels_numpy = np.genfromtxt(train_labels, delimiter=',')
    test_numpy = np.genfromtxt(testdir, delimiter=',')
    return([train_numpy,train_labels_numpy,test_numpy])

#[train_numpy,train_labels_numpy,test_numpy] = readInNumpy()

######################## READIG IN DATA ###########33


def readInPandas(traindir="train_data.csv",train_labels="train_labels.csv",testdir="test_data.csv"):
    import pandas as pd
    train = pd.read_csv(traindir,header=None)
    train_labels = pd.read_csv(train_labels)
    test =  pd.read_csv(testdir,header=None)'
    return([train,train_labels,test])

[train,train_labels,test] = readInPandas()

complete_data = pd.concat([train,test])
subsample = complete_data.sample(frac=0.2)

#################### DROPPING HIGHLY CORRELATED FEATURES ###############################################

#print(pd.DataFrame.drop.__doc__)
def dropHighCorrelation(data):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)) # # Select upper triangle of correlation matrix
    # # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    low_cor_data = data.drop(data.columns[to_drop], axis=1)
    return(low_cor_data)

data_low_cor = dropHighCorrelation(subsample)

##########################333 FIRST WE WILL TRY TO REDUCE THE DIMENSIONALITY #########33
# https://datascience.stackexchange.com/questions/4942/high-dimensional-data-what-are-useful-techniques-to-know
# https://scikit-learn.org/stable/modules/manifold.html






################ TRY p-PCA and removing HIGHLY CORRELATED FEATURES ##############


















############################ BAYESIAN APPROACH ###########################33

# =============================================================================
# import edward as ed
# import inferpy as inf
# from inferpy.models import Normal, Bernoulli, Categorical
# 
# d, N =  10, 500
# 
# # model definition
# with inf.ProbModel() as m:
# 
#     #define the weights
#     w0 = Normal(0,1)
#     w = Normal(0, 1, dim=d)
# 
#     # define the generative model
#     with inf.replicate(size=N):
#         x = Normal(0, 1, observed=True, dim=d)
#         y = Bernoulli(logits=w0+inf.dot(x, w), observed=True)
# 
# 
# # toy data generation
# x_train = Normal(loc=0, scale=1, dim=d).sample(N)
# y_train = Bernoulli(probs=0.4).sample(N)
# data = {x.name: x_train, y.name: y_train}
# 
# # compile and fit the model with training data
# m.compile()
# m.fit(data)
# 
# print(m.posterior([w, w0]))
# 
# 
# =============================================================================
