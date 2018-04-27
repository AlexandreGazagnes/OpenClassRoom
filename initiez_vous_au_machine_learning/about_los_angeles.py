#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#######################################
#######################################
#   About Los Angeles
#######################################
#######################################



# Description

# this script is a complete study about los Angeles house's prices
# based on sklearn dataset california houses



# import

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt

from sklearn.linear_model import *
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import fetch_california_housing





# dataset creation 

california = fetch_california_housing()


df = pd.DataFrame(california.data, columns=california.feature_names)
df["Prices"] = california.target * 100000

print(df.describe())
print(df.head())
print(df.tail())
print(df.dtypes)
print(df.shape)
print(df.ndim)



# any Nan?

for k in df.columns : 
	number = df[k].isna().sum() + df[k].isnull().sum()
	print("NaN/Null number in column {} : {}".format(k, number))
print()

# OK Good ! 



# just keep los angeles (not all california)

max_latitude = 34.329608 # north_boudary
max_longitude = -118.664373 # west_boudary
min_longitude =  -118.163978 #  east_boundary


longitude_corr = lambda x : True if max_longitude <= x <= min_longitude else False

df = df[df["Longitude"].map(longitude_corr)]

mask = df["Latitude"] <= max_latitude

df = df[mask]

print(df.shape)



# first plot 

fig, ax = plt.subplots(1,9, figsize=(16, 12))

for i, feature in enumerate(df.columns) : 
	ax[i].boxplot(df[feature])
	ax[i].set_title(feature)

plt.suptitle("Feature's distribution")
plt.show()



		# # rescale ?

		# std_scale = StandardScaler().fit(df)
		# df_scale = pd.DataFrame(std_scale.transform(df), columns=df.columns)


		# # show rescale plot 

		# fig, ax = plt.subplots(1,9, figsize=(16, 12))

		# for i, feature in enumerate(df_scale.columns) : 
		# 	ax[i].boxplot(df_scale[feature])
		# 	ax[i].set_title(feature)

		# plt.suptitle("Feature's distribution (rescaled")
		# plt.show()


		# beginin with sample : ) 

		# pc = 0.9
		# df = df.sample(frac=pc)
		# print(df.shape)
		# print(df.describe())  

# try linear_reression
####################################


# define X, y
X, y = df.drop(["Prices"], axis=1), df["Prices"]

# define test/train data
X_train, X_test, y_train, y_test = train_test_split(X,y)


# launch all LInear reression 
for Model in [LinearRegression, Lasso, Ridge, BayesianRidge, LassoCV, LassoLarsCV,
					 HuberRegressor, Lars, LassoLars, RidgeCV] : 


	model = Model()
	model.fit(X_train, y_train)
	error = round(100 * (1-model.score(X_test, y_test)), 2)

	print("{} : erreur de {} %".format(Model.__name__, error))



# try knn 
######################


# define quantiles 10

y = pd.cut(df["Prices"], 10, labels=range(10), include_lowest=True)
y = y.astype("int16")

# define X, y
X, y = df.drop(["Prices"], axis=1), y


# define test/train data
X_train, X_test, y_train, y_test = train_test_split(X,y)

# launch KNN
errors= list()
k_range = range(1, 50)

for k in k_range : 

	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train, y_train)

	error = round(knn.score(X_test, y_test), 2)
	errors.append(error)

plt.plot(k_range, errors)
plt.title("error rate of knn with various k")
plt.xlabel("k"); plt.ylabel("error rate")
plt.show()	 



# try knn 
######################


# define quantiles 10

y = pd.cut(df["Prices"], 10, labels=range(10), include_lowest=True)
y = y.astype("int16")


# define X, y
X = pd.DataFrame(dict(long=df.Longitude, lat=df.Latitude))
print(len(X))

# define test/train data
X_train, X_test, y_train, y_test = train_test_split(X,y)

# launch KNN
errors= list()
k_range = range(1, 50)

for k in k_range : 

	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train, y_train)

	error = round(knn.score(X_test, y_test), 2)
	errors.append(error)

plt.plot(k_range, errors)
plt.title("error rate of knn with various k")
plt.xlabel("k"); plt.ylabel("error rate")
plt.show()	 



# try knn
####################################""


# let's try an better 'sklearnic' version :) 

# define quantiles 10

y = pd.cut(df["Prices"], 10, labels=range(10), include_lowest=True)


# define X, y
X, y = df.drop(["Prices"], axis=1), y

# define test/train data
X_train, X_test, y_train, y_test = train_test_split(X,y)




# lets standardize our features 

std_scale = StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)


param_grid = {'n_neighbors':range(1, 50)}
score = 'accuracy'


# Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée

clf = GridSearchCV(KNeighborsClassifier(),param_grid, cv=10, scoring=score)
clf.fit(X_train_std, y_train)


# Afficher le(s) hyperparamètre(s) optimaux

print( "Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:",)
print( clf.best_params_)


print( "Résultats de la validation croisée :")
for mean, std, params in zip(clf.cv_results_['mean_test_score'], # score moyen
	clf.cv_results_['std_test_score'], # écart-type du accuracy_score
	clf.cv_results_['params']):

	print("{} = {} (+/-{}) for {}"\
		.format(score, round(mean, 3), round(std * 2,3), params ))