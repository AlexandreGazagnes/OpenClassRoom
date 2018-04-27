#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#######################################
#   about ridge and lasso
#######################################



# this is a sandbox script in order to test and manipulate ridge, ridgeCV, lasso
# lassoCV etc etc
# this work will be based on a radomised dataset 
# no direct link with external datasets our other studies



# import 


import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 
sns.set()

from sklearn.linear_model import * 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline



# dataframe creation


# we will work on a linear regression, degree 1 : 
# y = b0  + (b1 * x1) + (b2 * x2) + (b3 * x3) + (b4 * x4)....
# with n features

n_features = 3


# so firt lets creatre n coef

true_coefs = np.random.randint(-10, 10, size=n_features)
print(true_coefs)
print(len (true_coefs))

# lets now create our x1 and x2 features

n_obs = 10


