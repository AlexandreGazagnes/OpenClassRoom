#!/usr/bin/env pythonX
# -*- coding: utf-8 -*-



#############################
# my first linear regression
#############################



# import 

import pandas as pd 
import numpy as np 
import sklearn.linear_model as sk
import matplotlib.pyplot as plt
import random as rd



# build a fake noisy dataset 

x = np.arange(30)
a = 3 
b = 129	
y = [(a * xi) + b + (rd.choice([-2,-1, 1, 2]) * rd.random() * 15) for xi in x  ]



# show graph

plt.scatter(x, y)
plt.show()



# build our first linear regression 

model = sk.LinearRegression(fit_intercept=True)

# warning : you have to reshape x
model.fit(x[:, np.newaxis],y)



# print y = a x +b

a = model.coef_
b = model.intercept_
print("la droite de regression est de forme f(x) = {} * x + {}"\
	.format(a, b))



# define x values and reshape : 

x_min, x_max = min(x), max(x)
x_val = np.linspace(x_min, x_max, 10)
x_val = x_val[:, np.newaxis]



# define y from linear regression

y_val = model.predict(x_val)



# plot final result 

plt.scatter(	x, y, \
 				color="black", label="raws datas")

plt.plot(		x_val, y_val,\
				color="red", label="linear regression")

plt.title("first linear regression")
plt.xlabel("x")
plt.ylabel("y")

plt.legend()

plt.show()
