#!/usr/bin/env pythonX
# -*- coding: utf-8 -*-


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
import random as rd


fig = plt.figure()
for i in range(1, 10) : 
	ax = fig.add_subplot(3,3,i)
	name = "test "+str(i)
	ax.plot(np.random.randn(1000).cumsum(), "k", label=name)
	ax.set_xticks([0,500,1000])
ax.legend(loc="best")
plt.show()	


 ################   OR 	#################