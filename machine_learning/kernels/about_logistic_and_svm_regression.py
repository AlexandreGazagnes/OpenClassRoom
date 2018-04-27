#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#######################################
#   about logistic and svm
#######################################



# this is a sandbox script in order to test and manipulate logistic and svm
# classificator
# this work will be based on a mushroom.csv dataset
# no direct link with external datasets our other studies



# import 

import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score, \
									GridSearchCV

from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC



# dataframe creation

raw_data = pd.read_csv('../datasets/mushrooms.csv')



############################################################
# preprocessing : encodin in categorical features
############################################################


labelencoder=LabelEncoder()

for col in raw_data.columns:

    raw_data[col] = labelencoder.fit_transform(raw_data[col])

print(data.head())



############################################################
# split test and train data
############################################################


X = raw_data.iloc[:,1:23]
y = raw_data.iloc[:,0] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



############################################################
# standard logistic regression 
############################################################



# model creation

lr = LogisticRegression()
lr.fit(X_train,y_train)


# results management

y_prob = lr.predict_proba(X_test)[:,1] 
y_pred = np.where(y_prob > 0.5, 1, 0) 

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

print(roc_auc)


# show results

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate') ; plt.xlabel('False Positive Rate')

plt.show()



############################################################
# optimied logistic regression
############################################################


# model creation

lr = LogisticRegression()
params = {'C': np.logspace(-3, 3, 7) , 'penalty':['l1','l2'] }
lr_gs = GridSearchCV(lr, params, cv=10)
lr_gs.fit(X_train, y_train)

print(lr_gs.best_params_)


# results manaement

y_prob = lr.predict_proba(X_test)[:,1] 
y_pred = np.where(y_prob > 0.5, 1, 0) 

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)



############################################################
# sandard Support Vector Machine
############################################################

# model creation 

svm = LinearSVC()
params = { 'C': np.logspace(-3, 3, 7) }
gs_svm = GridSearchCV(lr, params, cv=10)
gs_svm.fit(X_train, y_train)

print(gs_svm.best_params_)


# results manaement

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)