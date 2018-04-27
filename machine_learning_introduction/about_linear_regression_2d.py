#!/usr/bin/env pythonX
# -*- coding: utf-8 -*-



#######################################
#   linear regression
#######################################



# this script is about linear regression using sklearn
# 
#



# import 

from math import sqrt

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



# create dataframe 

df = pd.read_csv("house.csv")

print("\ndf summary")
print("############\n")
print(type(df))
print(df.columns)
print(df.index)
print(df.shape)
print(df.head())
print(df.tail())
print("\n\n")



############################################
#	1st part, just work of 10 ligns of df
############################################


# reduce df

df = df[df.index <=10]

txt = "df reduced"
print("{}\n######################\n\n{}\n\n"\
	  .format(txt, df))



# sort df

df.sort_values("surface", inplace=True)
df = df.reindex(range(len(df)))

txt="df sorted by surface"
print("{}\n######################\n\n{}\n\n"\
	  .format(txt, df))



# show various caluclations

print(df.surface)
print(df.surface.shape)
print(df.surface.T)
print(df.surface.T.shape)
# print(pd.Series(df.surface[:, np.newaxis]))



###################################################
#	2nd part, work on normal df with sklearn
###################################################


# first visual exploration

df = pd.read_csv("house.csv")
plt.scatter(df.surface, df.loyer, label="values", color="darkblue", marker=".")
plt.show()


# data cleaning, outliners deleted 

df = df[df.surface<150]


# second graph more acurate

plt.scatter(df.surface, df.loyer, label="values", color="darkblue", marker=".")
plt.title("house price and surface datas")
plt.xlabel("surface")
plt.ylabel("loyer")
plt.legend(loc="upper left")
plt.show()



# cretion of our regressoin model using sk

model = LinearRegression()

# don't forget to reshape x values :) 
surface_reshaped = df.surface[:, np.newaxis]
model.fit(surface_reshaped, df.loyer)



# just for fun

a = round(model.coef_[0], 4)
b = round(model.intercept_,4)
print("y = a.x + b, with a={} and b={}\n\n".format(a, b))



# let's now prepare our graph with x and y values for plot

x_min, x_max = min(df.surface), max(df.surface)
x_scope = [x_min, x_max]
y1 = [a * x + b for x in x_scope]
y2 = model.predict(np.array(x_scope)[:, np.newaxis])

# of course y1 and y2 are same graph ! yellow + red ? :)
plt.scatter(df.surface, df.loyer, label="values", color="darkblue", marker=".")
plt.plot(x_scope, y1, label="comprenhsion list", color="yellow", alpha=0.5)
plt.plot(x_scope, y2, label="sklearn predict", color="red", alpha=0.5)
plt.title("house price and surface datas")
plt.xlabel("surface")
plt.ylabel("loyer")
plt.legend(loc="upper left")
plt.show()



######################################################
#	3rd part : Linear regression "behind the woods"
######################################################


# linear regression is behind the woods a matrix caculation.

# first we build a matrix from our x values, filling with 1 first lign
# and taking the Transpose (inverse?)
X = np.matrix([np.ones(df.shape[0]), df.surface]).T
print(X)

# indem for y, not with a maxtrix but a vector
y = np.matrix(df.loyer).T
print(y)

# lets solve this equation : (X.T.dot(X))exp(-1).dot(X.T).dot(y)
# np usefull method is 'linalg'
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

b2, a2 = round(theta.item(0),2), round(theta.item(1),2)
print("y = a.x + b, with a={} and b={}\n\n".format(a2, b2))

# we have of course, the same result from sklearn calculation
assert b2 == b
assert a2 == a



################################################################
#	4th part : sample and training/testing dataset 
################################################################


# # first we will not work on the entire dataset, imagine we have 1 000 000 datas
# # so we will have a sample of 10% of the dataset (very very bad idea :) 
# P = 0.1
# sample = np.random.randint(df.size, size=df.size*P)


# # lets create our sampled df
# sampled_df = df[sample]


# # second we will split our new df in traing and testing
# xtrain, xtest, ytrain, ytest = train_test_split(sampled_df.surface, 
# 								sampled_df.loyer, 
# 								train_size=0.8)

# # all good?
# for k in [xtrain, xtest, ytrain, ytest] : 
# 	print(k.shape)




################################################################
#	5th part : Trying Linear Regression as a noob 
################################################################


# let's try to do to linear regression without any external libreary
# we will use brute force in first time, then dictotomic 


df = pd.read_csv("house.csv")
df = df[df.surface<150]
# df.sort_values("surface", inplace=True)
# df = df.reindex(range(len(df)))

def error(data, pred) : 
	return (data - pred) **2

def error_score(errors) : 
	return (1/len(errors)) * sqrt(sum(errors))

def estimate_pred(x_list, a, b) : 
	return [((a * x) + b) for x in x_list]

def update_results(results, error_score, a, b ) : 
	results.append( (error_score, (a,b)) )

results = list()

for a in range(0, 100, 5) : 
	for b in range(-0, 500, 10) : 
		y = estimate_pred(df.surface, a, b)
		error_list = [error(data, pred) for data, pred in zip(df.loyer, y)]
		score = error_score(error_list)
		update_results(results, score, a , b)


print((30, 400) in [i[1] for i in results])
y = estimate_pred(df.surface, 30, 400)
error_list = [error(data, pred) for data, pred in zip(df.loyer, y)]
score = error_score(error_list)
print("score", score)


results = sorted(results, reverse=False)
print(results[0:20])

a3, b3 = results[0][1]
y3 = [a3 * x + b3 for x in x_scope]



plt.scatter(df.surface, df.loyer, label="values", color="darkblue", marker=".")
plt.plot(x_scope, y1, label="comprenhsion list", color="yellow", alpha=0.5)
plt.plot(x_scope, y2, label="sklearn predict", color="red", alpha=0.5)
plt.plot(x_scope, y3, label="brute force predict", color="green", alpha=0.5)

plt.title("house price and surface datas")
plt.xlabel("surface")
plt.ylabel("loyer")
plt.legend(loc="upper left")
plt.show()



results = list()

for a in range(150, 500, 1) :
	a = a/10 
	for b in range(200, 500, 1) : 
		y = estimate_pred(df.surface, a, b)
		error_list = [error(data, pred) for data, pred in zip(df.loyer, y)]
		score = error_score(error_list)
		update_results(results, score, a , b)

print((30, 400) in [i[1] for i in results])
y = estimate_pred(df.surface, 30, 400)
error_list = [error(data, pred) for data, pred in zip(df.loyer, y)]
score = error_score(error_list)
print("score", score)


results = sorted(results, reverse=False)
print(results[0:20])

a4, b4 = results[0][1]
y4 = [a4 * x + b4 for x in x_scope]



plt.scatter(df.surface, df.loyer, label="values", color="darkblue", marker=".")
plt.plot(x_scope, y1, label="comprenhsion list", color="yellow", alpha=0.5)
plt.plot(x_scope, y2, label="sklearn predict", color="red", alpha=0.5)
plt.plot(x_scope, y4, label="brute force predict", color="green", alpha=0.5)

plt.title("house price and surface datas")
plt.xlabel("surface")
plt.ylabel("loyer")
plt.legend(loc="upper left")
plt.show()


print("len de results = ", str(len(results)))


results_ch = 	results[0:100:2] \
			+ 	results[100 : 300 : 3]\
			+ 	results[300 : 1000 : 5]\
			+ 	results[1000 : 10000 : 10]\
			+ 	results[10000 :  : 100]


print("len de results = ", str(len(results_ch)))

results_tuple  = [(round(i[0],2), i[1][0], i[1][1])   for i in results_ch]
z, x, y =zip(*results_tuple)
print(x[:3], y[:3], z[:3] )



# this is much more better, lets plot this

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

plt.show()



# lets implement dychotomic search 


#######################" TO DO ################################"
a_min = 0 
a_max = 10000
b_min = 0
b_max = 1000
