#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#######################################
#   linear regression
#######################################



# this script is about linear regression using sklearn
# 
# desc



# import 

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


# # reduce df

# df = df[df.index <=10]

# txt = "df reduced"
# print("{}\n######################\n\n{}\n\n"\
# 	  .format(txt, df))



# # sort df

# df.sort_values("surface", inplace=True)
# df = df.reindex(range(len(df)))

# txt="df sorted by surface"
# print("{}\n######################\n\n{}\n\n"\
# 	  .format(txt, df))



# # show various caluclations

# print(df.surface)
# print(df.surface.shape)
# print(df.surface.T)
# print(df.surface.T.shape)
# # print(pd.Series(df.surface[:, np.newaxis]))



###################################################
#	2nd part, work on normal df with sklearn
###################################################


# first visual exploration

# plt.scatter(df.surface, df.loyer, label="values", color="darkblue", marker=".")
# plt.show()


# data cleaning, outliners deleted 

df = df[df.surface<150]


# second graph more acurate

plt.scatter(df.surface, df.loyer, label="values", color="darkblue", marker=".")
plt.title("house price and surface datas")
plt.xlabel("surface")
plt.ylabel("loyer")
plt.legend(loc="upper left")
# plt.show()



# cretion of our regressoin model using sk

model = LinearRegression()

# don't forget to reshape x values :) 
surface_reshaped = df.surface[:, np.newaxis]
model.fit(surface_reshaped, df.loyer)



# just for fun

a = round(model.coef_[0], 2)
b = round(model.intercept_,2)
print("y = a.x + b, with a={} and b={}\n\n".format(a, b))



# let's now prepare our graph with x and y values for plot

x_min, x_max = min(df.surface), max(df.surface)
x_scope = [x_min, x_max]
y1 = [a * x + b for x in x_scope]
y2 = model.predict(np.array(x_scope)[:, np.newaxis])

# of course y1 and y2 are same graph ! BUT...

plt.plot(x_scope, y1, label="comprenhsion list", color="yellow", alpha=0.5)
plt.plot(x_scope, y2, label="sklearn predict", color="red", alpha=0.5)
plt.legend()
plt.show()



######################################################
#	3rd part : Linear regression "behind the woods"
######################################################


# linear regression is a matrix caculation.

# first we build a matrix from our x, fillirng with 1 first lign
# and taking the Transpose (inverse?)
X = np.matrix([np.ones(df.shape[0]), df.surface]).T
print(X)

# indem for y, not with a ùaxtrix but a vector
y = np.matrix(df.loyer).T
print(y)

# lets solve this equation : (X.T.dot(X))exp(-1).dot(X.T).dot(y)
# np method is 'linalg'
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta)
b2, a2 = round(theta.item(0),2), round(theta.item(1),2)

# we have of course, the same result from sklearn calculation
assert b2 == b
assert a2 == a



################################################################
#	4th part : sample and training/testing dataset 
################################################################


# first we will not work on the entire dataset, imagine we have 1 000 000 datas
# so we will have a sample of 10% of the dataset (very very bad idea :) 
P = 0.1
sample = np.random.randint(df.size, size=df.size*P)

# lets create our sampled df
sampled_df = df[sample]


# second we will split our new df in traing and testing
xtrain, xtest, ytrain, ytest = train_test_split(sampled_df.surface, 
								sampled_df.loyer, 
								train_size=0.8)


for k in [xtrain, xtest, ytrain, ytest] : 
	print(k.shape)