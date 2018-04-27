#!/usr/bin/env python3
# -*- coding: utf-8 -*-



######################################################
######################################################
#		Grid Search CV with KNN
######################################################
######################################################



# Description

# this script is about using GridSearchCV with KNN. It is an easy 
# tutorial
# it is based on the "winequality-white.csv" dataset



# import 

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier



# dataframe creation

data = pd.read_csv('../datasets/winequality-white.csv', sep=";")
print(data.head())



# creating our matrix of features X and our vector target y

X = data.as_matrix(data.columns[:-1])
y = data.as_matrix([data.columns[-1]])
y = y.flatten()



# let's plot our feature's distribution

fig = plt.figure(figsize=(16, 12))

for feat_idx in range(X.shape[1]):
    ax = fig.add_subplot(3,4, (feat_idx+1))
    h = ax.hist(X[:, feat_idx], bins=50, color='steelblue',
                normed=True, edgecolor='none')
    ax.set_title(data.columns[feat_idx], fontsize=14)



# consider y not as a numercial but as a boolean value

y_class = np.where(y<6, 0, 1)



# split train and test data

X_train, X_test, y_train, y_test = \
    train_test_split(X, y_class, test_size=0.3)



# let's standardize our features for X

std_scaler = StandardScaler().fit(X_train)

X_train_std = std_scaler.transform(X_train)
X_test_std = std_scaler.transform(X_test)

		# StandardScaler().fit_transform() also very good :) 



# let's plot our standardized data for X


fig = plt.figure(figsize=(16, 12))

for feat_idx in range(X_train_std.shape[1]):
	ax = fig.add_subplot(3,4, (feat_idx+1))
	h = ax.hist(X_train_std[:, feat_idx],
			 bins=50, color='steelblue',
			normed=True, edgecolor='none')
	ax.set_title(data.columns[feat_idx], fontsize=14)
plt.show()



# we now are looking for good hyper parametres 

# if you have a "normal" computer try this : 

n_neighbors_range = range(1,20)
param_grid = {"n_neighbors":n_neighbors_range, 
				"weights": ["uniform", "distance"],}


		# if you have a super calculator with 10 GPU and 6 core i7 next gen, try this :) 
		# n_neighbors_range = range(1,30)

		# param_grid = {	'n_neighbors':n_neighbors_range, 
		# 				"weights": ["uniform", "distance"],
		# 				"algorithm": ["auto", "ball_tree", "kd_tree", "brute"]}

score = 'accuracy'
knn = KNeighborsClassifier()
gknn = GridSearchCV(knn, param_grid, cv=10, scoring=score)



# train our model

gknn.fit(X_train_std, y_train)



# print best params selected by GridSearchCV

print("best params : {}".format(gknn.best_params_))



# print related performances

results = list(zip(gknn.cv_results_['mean_test_score'], 
		gknn.cv_results_['std_test_score'], gknn.cv_results_['params']))
results.sort(reverse=True, key=lambda x : x[0])
results = results[:10]

print("CV results :")
for mean, std, params in results:
	print( "\t{} = {:0.3f} (+/-{:0.03f}) for {}".format(score,mean, std * 2, params ))


# let's study the predictive preformance

y_pred = gknn.predict(X_test_std)

print("\nTest score : {:0.3f}"\
		.format(accuracy_score(y_test, y_pred)))


