#!/usr/bin/env python3
# -*- coding: utf-8 -*-



######################################################
######################################################
#		Grid Search CV with various methods
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier



# dataframe creation

data = pd.read_csv('../datasets/winequality-white.csv', sep=";")
print(data.head())



# creating our matrix of features X and our vector target y

X = data.as_matrix(data.columns[:-1])
y = data.as_matrix([data.columns[-1]])
y = y.flatten()



		# # let's plot our feature's distribution

		# fig = plt.figure(figsize=(16, 12))

		# for feat_idx in range(X.shape[1]):
		#     ax = fig.add_subplot(3,4, (feat_idx+1))
		#     h = ax.hist(X[:, feat_idx], bins=50, color='steelblue',
		#                 normed=True, edgecolor='none')
		#     ax.set_title(data.columns[feat_idx], fontsize=14)



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



		# # let's plot our standardized data for X


		# fig = plt.figure(figsize=(16, 12))

		# for feat_idx in range(X_train_std.shape[1]):
		# 	ax = fig.add_subplot(3,4, (feat_idx+1))
		# 	h = ax.hist(X_train_std[:, feat_idx],
		# 			 bins=50, color='steelblue',
		# 			normed=True, edgecolor='none')
		# 	ax.set_title(data.columns[feat_idx], fontsize=14)
		# plt.show()



#########################################################################
#	Try various Classifier but without any params
#########################################################################


model_list = [	KNeighborsClassifier, 
				LogisticRegression, 
				LinearSVC,
				RandomForestClassifier, 
				DecisionTreeClassifier]

# "random_state":[1]

params_list = [	{}, {"random_state":[1]}, {"random_state":[1]}, 
				{"random_state":[1]},  {"random_state":[1]}]


for Model, param_grid in zip(model_list, params_list) : 

	print("""
###############################################################
MODELE :  {} WITHOUT pre-parametrage
###############################################################""".format(Model.__name__))

	score = 'accuracy'
	model = Model()
	gmodel = GridSearchCV(	model, param_grid, cv=10, 
							scoring=score)


	# train our model

	gmodel.fit(X_train_std, y_train)


	# print best params selected by GridSearchCV

	print("best params : {}".format(gmodel.best_params_))


	# print related performances

	results = list(zip(gmodel.cv_results_['mean_test_score'], 
			gmodel.cv_results_['std_test_score'], gmodel.cv_results_['params']))
	results.sort(reverse=True, key=lambda x : x[0])
	results = results[:10]

	print("CV results :")
	for mean, std, params in results:
		print( "\t{} = {:0.3f} (+/-{:0.03f}) for {}".format(score,mean, std * 2, params ))


	# let's study the predictive preformance

	y_pred = gmodel.predict(X_test_std)

	print("\nTest score : {:0.3f}"\
			.format(accuracy_score(y_test, y_pred)))




#########################################################################
#	Try various Classifier but with specifics params
#########################################################################



		# model_list = [	KNeighborsClassifier, 
		# 				LogisticRegression, 
		# 				LinearSVC,
		# 				RandomForestClassifier,
		# 				DecisionTreeClassifier ]


		# params_list = [	{"n_neighbors":range(1,20),"weights": ["uniform", "distance"],"random_state":[1]},
		# 				{'C': np.logspace(-5, 5, 100) , 'penalty':['l1','l2'], "random_state":[1]},
		# 				{ 'C': np.logspace(-5, 5, 100), "probability": [True, False],"random_state":[1] },
		# 				{"n_estimators": np.logspace(1, 45, 100), "oob_score":[True, False], 'warm_start': [True,False] , "random_state":[1]},
		# 				{"max_depth":[3,5, 7, 10, 20, 50], "max_features": ['auto', 'sqrt', "sqrt2", 'log2'],"random_state":[1] }]





		# for Model, param_grid in zip(model_list, params_list) : 

		# 	print("""
		# ###############################################################
		# MODELE :  {} avec pre-parametrage
		# ###############################################################""".format(Model.__name__))

		# 	score = 'accuracy'
		# 	model = Model()
		# 	gmodel = GridSearchCV(	model, param_grid, cv=10, 
		# 							scoring=score, random_state=0)


		# 	# train our model

		# 	gmodel.fit(X_train_std, y_train)


		# 	# print best params selected by GridSearchCV

		# 	print("best params : {}".format(gmodel.best_params_))


		# 	# print related performances

		# 	results = list(zip(gmodel.cv_results_['mean_test_score'], 
		# 			gmodel.cv_results_['std_test_score'], gmodel.cv_results_['params']))
		# 	results.sort(reverse=True, key=lambda x : x[0])
		# 	results = results[:10]

		# 	print("CV results :")
		# 	for mean, std, params in results:
		# 		print( "\t{} = {:0.3f} (+/-{:0.03f}) for {}".format(score,mean, std * 2, params ))


		# 	# let's study the predictive preformance

		# 	y_pred = gmodel.predict(X_test_std)

		# 	print("\nTest score : {:0.3f}"\
		# 			.format(accuracy_score(y_test, y_pred)))





#########################################################################
#	Focus on Radom Forest
#########################################################################


Model = RandomForestClassifier
param_grid	= {"n_estimators":  [100, 200, 300, 400, 500, 500, 600, 700, 800, 900, 1000] , 'oob_score':[True, False], 'warm_start': [True, False] , "random_state":[1]}
# 300, 500, 600, 700, 800, 900, 1000, 1000, 1200

print("""
###############################################################
MODELE :  Only {} avec pre-parametrage
###############################################################""".format(Model.__name__))

score = 'accuracy'
model = Model()
gmodel = GridSearchCV(	model, param_grid, cv=10, 
						scoring=score)


# train our model

gmodel.fit(X_train_std, y_train)


# print best params selected by GridSearchCV

print("best params : {}".format(gmodel.best_params_))


# print related performances

results = list(zip(gmodel.cv_results_['mean_test_score'], 
		gmodel.cv_results_['std_test_score'], gmodel.cv_results_['params']))
results.sort(reverse=True, key=lambda x : x[0])
results = results[:10]

print("CV results :")
for mean, std, params in results:
	print( "\t{} = {:0.3f} (+/-{:0.03f}) for {}".format(score,mean, std * 2, params ))


# let's study the predictive preformance

y_pred = gmodel.predict(X_test_std)

print("\nTest score : {:0.3f}"\
		.format(accuracy_score(y_test, y_pred)))




######################################################################
# My_GridSearchCV
######################################################################




def my_grid_search_cv(Model=None, params=params_dict, cv=None, scoring=None) : 


	results_list = list()
	for param1 in param1_poss : 
		for param_2 in param_2 : 

			# create kfold
				kfold_score = list()

				for kfold_train, kfold_test in kfold : 

					X_kfold_train = None
					y_Kfold_test = None

						model = Model(param=param, scoring=scoring)

						model.fit(X_kfold_train, y_Kfold_test)


						# print best params selected by GridSearchCV

						print("best params : {}".format(gmodel.best_params_))

	# decompress param_dict

	# for 
	# first define our k folds with a fix random state	