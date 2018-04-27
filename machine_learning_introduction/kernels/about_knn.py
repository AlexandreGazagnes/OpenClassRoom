#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################
#######################################
#   About K-Nearest-Neighbors
#######################################
#######################################



# Description

# this script is about K-Nearest-Neighbors
# classification using sklearn
# nothing outsanding, just few calulation for fun
# with famous Iris dataset



######################################
# import
######################################


import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler



######################################
# cleaning and exploration
######################################


# creating a fake dataframe 

df = pd.read_csv("iris_data.csv")



# first print exploration

print("Row DataFrame : ")
print(df.head())
print(df.tail())
print(df.describe())
print("\ndataframe info: \nshape : {}\ntypes :\n{}\ndim : {}\n"\
		.format(df.shape, df.dtypes, df.ndim))



# any NaN?

for k in df.columns : 
	print("NaN number in column {} : {}".format(k, df[k].isna().sum()))
print()



# number or class : 

print("How many flower varieties (class)? : {}\n".format(df["class"].unique()))



# replace "Iris-flower" by flower in df.class

df["class"] = df["class"].map(lambda flower : flower.replace("Iris-", ""))



# just for ploting
def set_color(flower) : 
	if "setosa" in flower : 
		return "red"
	elif "versicolor" in flower : 
		return "blue"
	elif "virginica" in flower : 
		return "green"
	else : 
		raise ValueError("Flower unknowned")

df["color"] = df["class"].map(set_color)



# print ehanced dataframe

print("New DataFrame : ")
print(df.head())



# first raw plot : 


fig, ax = plt.subplots(2,2,figsize=(16, 12))

loc = [[0,0], [0,1], [1,0], [1,1]]

for val, loc in zip(df.drop(["class", "color"], axis=1).columns, loc) : 
	for flower in df["class"].unique() : 
		mask = df["class"] == flower
		ax[loc[0], loc[1]].scatter(	df.loc[mask, val], df.loc[mask, "class"], 
						c=df.loc[mask, "color"], label = flower, marker='.')
	ax[loc[0], loc[1]].set_title("flower type by {} ".format(val)); 
	ax[loc[0], loc[1]].legend(loc="upper left")

plt.suptitle('Flower type, feature by feature', fontsize=16)
plt.subplots_adjust(wspace=0.5)
plt.show()



# second raw plot 

fig, ax = plt.subplots(1,2,figsize=(16, 12))

for i, val in enumerate(["sepal", "petal"])  :

	x_len, y_wid = "{}_length".format(val), "{}_width".format(val)

	for flower in df["class"].unique() : 
		mask = df["class"] == flower
		ax[i].scatter(	df.loc[mask, x_len], df.loc[mask, y_wid], 
						c=df.loc[mask, "color"], label = flower, marker='.')
	ax[i].set_title("{} length/width".format(val)); ax[i].legend(loc="upper left")
	ax[i].set_xlabel("length") ; ax[i].set_ylabel("width")

plt.suptitle("Flower type, but grouped by feature's family", fontsize=16)
plt.subplots_adjust(wspace=0.5)
plt.show()



# third raw plot


fig, ax = plt.subplots(1,4,figsize=(16, 12))

for i, val in enumerate(df.drop(["color", "class"], axis=1).columns):
	ax[i].hist(df.loc[:,val], bins=50)
	ax[i].set_title("distribution of {}".format(val))
	ax[i].set_xlabel("val") ; ax[i].set_ylabel("count")
	ax[i].set_title("distribution of {}".format(val))

plt.suptitle("Features's global distribution", fontsize=16)
plt.subplots_adjust(wspace=0.5)
plt.show()



# last raw plot

fig, ax = plt.subplots(1,4,figsize=(16, 12))

for i, val in enumerate(df.drop(["class", "color"], axis=1).columns) : 
		
	box_plot_list = [ df.loc[df["class"] == flower, val] \
						for  flower in df["class"].unique()]
	ax[i].boxplot(box_plot_list)
	ax[i].set_xticklabels(df["class"].unique()) ; ax[i].set_ylabel(val)
	ax[i].set_title("distribution of {}".format(val))

plt.suptitle("Feature's distribution but grouped by flower type", fontsize=16)
plt.subplots_adjust(wspace=0.5)
plt.show()



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#		DELETE OUTLINERS
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



##############################################
#	KNN in use
##############################################


# first let's seperate features from target

X = df.drop(["class", "color"], axis=1)
y = df['class']



# let's have a brute force dummy KNN 

errors= list()
k_range = range(1, 50)

for k in k_range : 

	# split into train and test
	X_train, X_test, y_train, y_test = train_test_split(X, y, 
		test_size=0.25, random_state=42)

	# instantiate learning model (k = 3)
	knn = KNeighborsClassifier(n_neighbors=k)

	# fitting the model
	knn.fit(X_train, y_train)

	# predict the response
	pred = knn.predict(X_test)
	# 	print(pd.DataFrame(dict(predict=pred, tested=y_test)))

	# evaluate accuracy
	errors.append(100 * round(1 - knn.score(X_test, y_test),4))


plt.plot(k_range, errors)
plt.title("error rate of knn with various k")
plt.xlabel("k"); plt.ylabel("error rate")
plt.show()	 



# is any feature more important or performant ? 

# create design matrix X and target vector y
X_petal = df.drop(["class", "color", "sepal_width", "sepal_length"], axis=1)
X_sepal = df.drop(["class", "color", "petal_width", "petal_length"], axis=1)

# prepare for ploting
fig, ax = plt.subplots(2,2)

# main features loop
for i, var in enumerate([(X_sepal, "sepal"), (X_petal, "petal")]) : 
	
	X_feat, name = var
	errors = list()
	k_range = range(1, 50)

	# for each k
	for k in k_range : 

		# split into train and test
		X_train, X_test, y_train, y_test = train_test_split(X_feat, y, 
			test_size=0.33, random_state=42)

		# instantiate, fitting and predict
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train, y_train)
		pred = knn.predict(X_test)

		# evaluate accuracy
		errors.append(100 * round(1 - knn.score(X_test, y_test),4))
	
	# plot error vs k
	ax[0,i].plot(k_range, errors)
	ax[0,i].set_title("KNN result for {}".format(name))
	ax[0,i].set_xlabel("k") ; ax[0,i].set_ylabel("error in %")
	ax[0,i].set_xticks(range(0, 50, 5)), ax[0,i].set_yticks(range(0, 35, 2))

# also plot (or replot) sepal/petal classification 
for i, val in enumerate(["sepal", "petal"])  :
	x_len, y_wid = "{}_length".format(val), "{}_width".format(val)
	for flower in df["class"].unique() : 
		mask = df["class"] == flower
		ax[1, i].scatter(	df.loc[mask, x_len], df.loc[mask, y_wid], 
						c=df.loc[mask, "color"], label = flower, marker='.')
	ax[1, i].set_title("{} length/width".format(val)); ax[1, i].legend(loc="upper left")
	ax[1, i].set_xlabel("length") ; ax[1, i].set_ylabel("width")

plt.show()



# for sure on feature is more performant than the other :) 
# just for fun let's redo the same operation but without sepal_width

errors= list()
k_range = range(1, 50)

for k in k_range : 

	# split into train and test
	X_train, X_test, y_train, y_test = \
		train_test_split(X.drop(["sepal_width"], axis=1), y, 
		test_size=0.25, random_state=42)

	# instantiate learning model (k = 3)
	knn = KNeighborsClassifier(n_neighbors=k)

	# fitting the model
	knn.fit(X_train, y_train)

	# predict the response
	pred = knn.predict(X_test)
	# 	print(pd.DataFrame(dict(predict=pred, tested=y_test)))

	# evaluate accuracy
	errors.append(100 * round(1 - knn.score(X_test, y_test),4))


plt.plot(k_range, errors)
plt.title("error rate of knn with various k")
plt.xlabel("k"); plt.ylabel("error rate")
plt.show()	 



# lets implement a Cross validation method, but from scratch (without sklearn 
# dedicated module) 
# CV stands for "Cross Validation :)"

# separate the dataframe in  : 
CV_SIZE = 5

# creating a list of index pairs strat/stop for each slice of the CV 
CV_range, CV_mod = len(df) // CV_SIZE, len(df) % CV_SIZE
CV_list = np.array((CV_SIZE-1) * [CV_range] + [CV_range + CV_mod]) 
CV_list = [[i - CV_range, i] for i in CV_list.cumsum()]

# depreciated but possible :  shuffle dataframe 
# df = shuffle(df)
# X = df.drop(["class", "color"], axis=1)
# y = df['class']

errors  = list()
k_range = range(1, 50)

for k in k_range : 

	CV_errors = list()

	for i,j in CV_list : 

		X_train, X_test = X.drop(df.index[i:j], axis=0), X[i:j]
		y_train, y_test = y.drop(df.index[i:j], axis=0), y[i:j]

		# instantiate, fitting and predict
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train, y_train)
		pred = knn.predict(X_test)

		# evaluate accuracy
		CV_errors.append(100 * round(1 - knn.score(X_test, y_test),4))
	
	errors.append(np.array(CV_errors).mean()) 

# plot error vs k
plt.plot(k_range, errors)
plt.title("error rate of knn with various k")
plt.xlabel("k"); plt.ylabel("error rate")
plt.xticks(range(0, 50, 2))
plt.show()	 

# good but not  optimal, for a Cross Validation method




# let's try an better 'sklearnic' version :) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



# lets standardize our features 

std_scale = StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)


input("hallo?\n")

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

y_pred = clf.predict(X_test_std)
