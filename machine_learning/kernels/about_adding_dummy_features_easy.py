#!/usr/bin/env python3
# -*- coding: utf-8 -*-



######################################################
######################################################
#		Adding Noise features
######################################################
######################################################



# Description

# this script is short study regarding testing a model with additional 
# random vairables, just for fun, just to learn : ) 
# it is based on the "boston" dataset



# import 
import math
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


from sklearn.datasets import load_boston

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_log_error

# from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC
# from sklearn.tree import DecisionTreeClassifier



# constant

SHOW = False



# dataset and dataframe creation

boston = load_boston()

if SHOW : 
	print(boston.DESCR)
	[print(i) for i in boston]

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["PRICE"] = boston.target



###################################################################
#		first exploration
###################################################################


if SHOW : 
	print(df.head())
	print(df.tail())

	print(df.describe())

	print(df.dtypes)


if SHOW : 
	nb_feat_unique =  [(feat, len(df[feat].unique())) for feat in df.columns]
	nb_feat_unique.sort(key=lambda x : x[1], reverse=True)
	[print("{} : {}".format(feat, nb)) for feat, nb in nb_feat_unique]


if SHOW : 
	df.hist(grid=True,bins=50, figsize=(10,11))
	plt.show() 

	df.boxplot(figsize=(13,7))
	plt.show()


# explore PRICE main properties

if SHOW : 

	print(df["PRICE"].describe())

	print("Skewness: {}".format(df["PRICE"].skew()))
	print("Kurtosis: {}".format(df["PRICE"].kurt()))
 
	sns.distplot(df["PRICE"])
	plt.show()


# explore corr matrix

if SHOW :
	corrmat = df.corr()
	input("continuer?\t")
	f, ax = plt.subplots(figsize=(12, 9))
	sns.heatmap(corrmat, vmax=.8, square=True, fmt='.2f', annot=True)
	plt.show()


# explore null/Nan values by col

total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

if SHOW : 
	print(missing_data)


# explore null/Nan values by row

total = df.T.isnull().sum().sort_values(ascending=False)
percent = (df.T.isnull().sum()/df.T.isnull().count()).sort_values(ascending=False)
missing_observation = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_observation = missing_observation[missing_observation["Total"]>0]

if SHOW : 
	print(missing_observation)



###################################################################
#		delete ouliers
###################################################################


# first study corelation
size = int(math.ceil(len(df.columns)**0.5))

if SHOW : 
	for i, col in enumerate(df.columns) : 
		plt.subplot(size, size, i+1)
		plt.scatter(df[col], df["PRICE"], marker=".")
		plt.title(col)
	plt.subplots_adjust(hspace=0.2)
	plt.show()




# unvariate analysis with standardizing data

price_scaled = StandardScaler().fit_transform(df['PRICE'][:,np.newaxis])
low_range = price_scaled[price_scaled[:,0].argsort()][:10]
high_range= price_scaled[price_scaled[:,0].argsort()][-10:]

if  SHOW : 
	print('outer range (low) of the distribution:')
	print(low_range)
	print('\nouter range (high) of the distribution:')
	print(high_range)



# bivariate analysis with standardizing data


if  SHOW : 
	for i, col in enumerate(df.columns) : 
		var = col
		plt.subplot(size, size, i+1)
		plt.scatter(df[col],price_scaled, marker=".")
		plt.title(col)
	plt.show()



###################################################################
#		prepare data for regression
###################################################################


# separate X,y

X = df.drop(["PRICE"], axis=1) 
y = df["PRICE"] * 1000


# rescale

scaler = StandardScaler().fit(X)
X = scaler.transform(X)


# test train split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)



###################################################################
#		baseline basic linear regression
###################################################################




# intiate and train the model

lr = LinearRegression()
params = {},
glr = GridSearchCV(lr, params, cv=10, scoring="neg_mean_squared_error")
glr.fit(X_train, y_train)


# pred
y_pred = glr.predict(X_test)
lr_score = mean_squared_log_error(y_test, y_pred, )

# show results

if not SHOW :
	print("score with linear regression, normal dataset") 
	print(lr_score)

if SHOW :
	plt.scatter(y_pred, y_test,marker = ".")
	plt.show()



###################################################################
#		2nd baseline Ridge and/or Lasso
###################################################################


# just for fun to try to use Ridge and/or Lasso : 

# initiate and train the model
ridge = Ridge()
alphas = np.logspace(-5,5, 800)
params = { "alpha": alphas,	 "normalize":[ False], }
gridge = GridSearchCV(ridge, params, cv=10, scoring="neg_mean_squared_error")
gridge.fit(X_train, y_train)


# pred 
y_pred = gridge.predict(X_test)
ridge_score = mean_squared_log_error(y_test, y_pred, )


# show results

if not SHOW : 
	print("score with ridge regression, normal dataset")
	print(ridge_score)

if SHOW : 
	plt.scatter(y_pred, y_test,)
	plt.show()


if SHOW : 
	results = list(zip(gridge.cv_results_['mean_test_score'], 
	gridge.cv_results_['std_test_score'], gridge.cv_results_['params']))
	results.sort(reverse=True, key=lambda x : x[0])
	results = results[:10]
	print(results)

if SHOW : 
	print(gridge.best_score_)
	print(gridge.best_params_)

if not SHOW : 
	plt.plot(alphas, gridge.cv_results_['mean_test_score'])
	plt.plot([alphas[0], alphas[-1]], [lr_score, lr_score])
	plt.xscale("log")
	plt.axis('tight')
	plt.show()








###################################################################
#		2nd Adding noizy features
###################################################################


# adding 10 new features : 

df["random_0"] = np.random.randint(0,100, len(df))
df["random_1"] = np.random.randint(0,100, len(df))
df["random_2"] = np.random.randint(0,1000, len(df))
df["random_3"] = np.random.randint(0,10000, len(df))
df["random_4"] = np.random.randint(0,100000, len(df))
df["random_5"] = np.random.randint(-100,100, len(df))
df["random_6"] = np.random.randint(-100,100, len(df))
df["random_7"] = np.random.randint(-1000,1000, len(df))
df["random_8"] = np.random.randint(-10000,10000, len(df))
df["random_9"] = np.random.randint(-100000,100000, len(df))

if SHOW : 
	print(df.head())


# separate X,y

X = df.drop(["PRICE"], axis=1) 
y = df["PRICE"] * 1000


# rescale

scaler = StandardScaler().fit(X)
X = scaler.transform(X)


# test train split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)



# define Meta params : 

model_list = [LinearRegression, Lasso, Ridge]
params_list = [{},{"alpha": np.logspace(-5,5, 800)}, {"alpha": np.logspace(-5,5, 800)}]


# main loop : 

for Model, params in zip(model_list,params_list) : 


	# lauch Model

	model = Model()
	params = {}
	gmodel = GridSearchCV(model, params, cv=10, scoring="neg_mean_squared_error")
	gmodel.fit(X_train, y_train)


	# pred
	y_pred = gmodel.predict(X_test)
	score = mean_squared_log_error(y_test, y_pred, )


	# show results

	if  not SHOW : 
		print("score with {}, degraded dataset".format(Model.__name__))
		print(score)

	if  SHOW :
		plt.scatter(y_pred, y_test,marker = ".")
		plt.show()
