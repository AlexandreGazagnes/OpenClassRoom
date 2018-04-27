#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#######################################
#######################################
#   About Polynomial Regression
#######################################
#######################################



# Description

# this kernel is about polynomial regression using sklearn
# nothing outsanding, just few calulation for fun :)
# no related dataset or other studies



# import

from collections import Counter

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt

from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score



# global params (constants) 

train_size = 0.8

k = 2 # coef of x ** k
a = 1 # # a for y= a.x**k +b
b = 0 # b for y= a.x**k +b

al = 0.5 # coef of "experimental error" exepted float [min 0, max 1]

degre_max = 7 # max x***coef to try in regression model
nb_of_exp = 100# nb of experimetal test (x_exp)



# creating a fake dataframe 

x_the = np.linspace(-100, 100, 1000)
x_exp = np.random.randint(-100, 100, nb_of_exp)

y_the =  pd.Series([((a*xi**k) + b) for xi in x_the])
y_exp = pd.Series([((a*xi**k) + b) for xi in x_exp])
y_exp += (1000 * al) * np.random.randn(len(y_exp))



# ploting data

plt.plot(x_the, y_the, c="g", label="theory")
plt.scatter(x_exp, y_exp, marker=".", label="experience" )
plt.title("test polynomial regression")
plt.xlabel("x")
plt.ylabel("y")

print("\n\nPolynomial Regression, for {} experimental values and {} of error\n"\
		.format(nb_of_exp, al))



# polynomial regression

for deg in range(degre_max) : 

	# split train/test dataset
	x_train, x_test, y_train, y_test = train_test_split(x_exp, y_exp, 
		train_size=train_size , test_size=round(1-train_size, 2))

	# creating and fiting the model, by piping both polynomial and linear reg.
	polynomial_features = PolynomialFeatures(degree=deg, include_bias=True)
	linear_regression = LinearRegression()
	model = Pipeline([("polynomial_features", polynomial_features),
						 ("linear_regression", linear_regression)])
	model.fit(x_train[:, np.newaxis], y_train)

	# estimating error (1-r2 in %)
	error = round(100 * (1 - model.score(x_test[:, np.newaxis], y_test)), 2)

	# ploting model prediction
	y_pred = model.predict(x_the[:, np.newaxis]) 
	plt.plot(x_the, y_pred, label="regression deg : {}, err : {}%"\
		.format(deg, error))
	
	# creating some text, with degree, error, and estimated polynomial equation
	txt = str("Degré {}, error {}% for y = ".format(deg, error)) 
	for i, j in enumerate(linear_regression.coef_) : 
		if not i : continue
		txt +="{} x**{} + ".format(round(j, 2), i)
	
	txt += str(round(linear_regression.intercept_, 2))
	print(txt)


# ploting result

plt.legend(loc="upper left")
plt.show()
