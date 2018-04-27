#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#######################################
#######################################
#   Multi Dimensionnal Regression
#######################################
#######################################



# Description

# this script is about linear and polynomial regression using sklearn
# in a n-dim (3 here but serialisable)
# it comes after the fist exercice "about_linear_regression.py"



# importons les librairies

from collections import Counter

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import *
from sklearn.svm import LinearSVR, NuSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score



# définissions quelques constantes 

NB_TEST = 500
TRAIN_SIZE = 0.8
TEST_SIZE = 0.2



# Créons le dataframe

df = pd.read_csv("house_data.csv")



############################################
#	PARTIE 1 : Nettoyage des données
############################################


# dans cette partie nous allons effectuer un certain nombre
# d'opérations sur notre dataframe initial afin de le "nettoyer", 
# et de permettre un travail plus aisé par la suite



# affichons les informations essentielles du dataframe

print(	"affichons les 5 premieres et 5 derniers lignes\n{}\n{}\n"\
		.format(df.head(), df.tail()))
print(	"affichons quelques informations utiles\nshape : {}\ntypes :\n{}\ndim : {}\n"\
		.format(df.shape, df.dtypes, df.ndim))



# regradons s'il y a des valeurs manquantes

for k in df.columns : 
	print("nombre de NaN dans la colonne {} : {}".format(k, df[k].isna().sum()))

mask = df["surface"].isna()

print("\nvoici les lignes en question : \n{}\n".format(df[mask]))



# supprimons ces valeurs manquantes

df = df.dropna()



# assurons nous que le travail a été fait

print('après avoir nettoyez nos données :')
for k in df.columns : 
	print("nombre de NaN dans la colonne {} : {}".format(k, df[k].isna().sum()))

print("notre dataframe est maintenant de dimension {}".format(df.shape))



# Supprimons les outliners 

df =  df[df["surface"] < 180]



# apportons quelques améliorations "personnelles" à notre dataframe

# reindexer notre dataframe
df.index = range(len(df))

# unifier la langue des colonnes (on garde le francais)
df = df.rename(index=str, columns={"price": "loyer"})

# retyper les valeurs d'arrondissement (en int)
df["arrondissement"] = df["arrondissement"].astype("int16")

# verifions le résultat
print("affichons quelques informations utiles\nshape : {}\ntypes :\n{}\ndim : {}\n"\
		.format(df.shape, df.dtypes, df.ndim))



					# ############################################
					# #	PARTIE 2 : Exploration des données
					# ############################################


					# # dans cette partie, nous allons afficher un grand nombre de
					# # graphiques afin de visualiser et d'explorer "à la main"
					# # notre jeu de données.
					# # Nombre de ces graphes ne sont pas demandés dans l'exercice, mais 
					# # ce travail a une importance pour la suite du travail effectué


					# # Affichons les graphiques que nous pouvons faire de facon "brute"

					# fig, ax = plt.subplots(1, 3, figsize=(15,6))

					# ax[0].scatter(df.surface, df.loyer, marker=".")	
					# ax[0].set_xlabel("surface") ; ax[0].set_ylabel("loyer")
					# ax[0].set_title("loyer en fonction de la surface")

					# ax[1].scatter(df.arrondissement, df.loyer, marker=".")
					# ax[1].set_xlabel("arrondissement") ; ax[1].set_ylabel("loyer")
					# ax[1].set_title("loyer en fonction de l'arrondissement")

					# ax[2].scatter(df.surface, df.arrondissement, marker=".")
					# ax[2].set_xlabel("surface") ; ax[2].set_ylabel("arrondissement")
					# ax[2].set_title("arrondissement en fonction de la surface")

					# plt.show()



					# # on voit bien que cela n'est pas très utile...
					# # tentons donc une visualisation 3d

					# fig = plt.figure(figsize=(8,6))
					# ax = fig.add_subplot(111, projection='3d')
					# ax.scatter(df.surface, df.arrondissement, df.loyer)
					# ax.set_xlabel("surface") ; ax.set_ylabel("arrondissement")
					# ax.set_zlabel("loyer")
					# ax.set_title("loyer en fonction de la surface et de l'arrondissement - vue 3d")

					# plt.show()



					# # en jouant un peu avec les axes, on peut voir que les prix dans le 10e arr 
					# # sont moins elevés que ailleurs, à surface identique
					# # essayons d'une autre facon...

					# sampled_df = df.sort_values(by="loyer")	

					# min_, max_ = min(sampled_df.loyer), max(sampled_df.loyer)
					# ecart = (max_ - min_)/10

					# echelle = [[min_ + (ecart * i), min_ + (ecart * (i+1)) - 0.001] for i  in range(0,10)]
					# echelle[-1][1]+=0.01

					# def set_cat(price) : 
					# 	for i, minimax in enumerate(echelle) : 
					# 		if minimax[0]<=price<=minimax[1] : 
					# 			return i+1

					# sampled_df["cat"] = sampled_df["loyer"].map(set_cat)

					# def set_color(cat) : 
					# 	if cat < 2 : return "yellow"
					# 	elif 2 <=cat < 4: return "orange"
					# 	elif 4<= cat < 6 : return "red"
					# 	elif 6 <= cat <8 : return "purple"
					# 	else : return "black" 

					# sampled_df["color"] = sampled_df["cat"].map(set_color)

					# fig, ax = plt.subplots(1, 1, figsize=(15,8))
					# area = sampled_df["cat"] * 4 * 10**2
					# ax.scatter(	sampled_df.surface, sampled_df.arrondissement, s=area, 
					# 				c=sampled_df["color"],alpha=0.3)
					# ax.set_xlabel("surface") ; ax.set_ylabel("arrondissement")
					# ax.set_title("loyer en fonction de la surface et de l'arrondissement - vue 2d")
					# plt.show()



					# # il semble que le 10e arrondissement soit effectivement un peu moins cher...
					# # regardons alors du coté du prix/surface

					# df["loyer_m2"] = df["loyer"] / df["surface"]

					# fig = plt.figure(figsize=(15,8))

					# ax = fig.add_subplot(1,1,1)
					# ax.boxplot(	[df.loc[df.arrondissement == j, "loyer_m2"] for j in\
					# 				df.arrondissement.unique()])
					# ax.set_xticklabels(df.arrondissement.unique())
					# ax.set_xlabel("arrondissement") ; ax.set_ylabel("loyer_m2")
					# ax.set_title("loyer_m2 par arrondissement")

					# plt.show()



					# # cette fois les box plot ne trahissent pas, le 10e arrondissement, est bien le 
					# # moins cher, de peu certes, mais le moins cher.



##################################################
#	PARTIE 3 Rappel : Regression Lineaire simple
##################################################


# dans cette partie nous revenons rapidement sur la regression linéaire 
# simple, c'est à dire, juste sur la matrice prix/surface
# ce qui nous interesse notamment, c'est le score de cette regression


# creons les données train et test
surface_train, surface_test, loyer_train, loyer_test \
	= train_test_split(df.surface, df.loyer, 
		train_size=TRAIN_SIZE, test_size=TEST_SIZE)


# creons et entrainons le modele sur les données train
model = LinearRegression()
model.fit(surface_train[:, np.newaxis], loyer_train)


# calculons les données prédides sur les données test 
loyer_pred = model.predict(surface_test[:, np.newaxis])


# caluclons les erreurs : 
model_error = 100 * round((1 - model.score(surface_test[:, np.newaxis], loyer_test)) ,4)
print("avec un seul feature (surface), apres avoir refait l'operation une seule fois: ")
print("l'erreur modele est de {}%,\n".format(round(model_error,2)))



#  pour ajuster notre calcul, faisons le plusieurs fois et prennons 
#  la moynne des erreurs

model_errors = list()
nb_test=NB_TEST

for _ in range(nb_test) : 
	surface_train, surface_test, loyer_train, loyer_test \
		= train_test_split(df.surface, df.loyer, train_size=TRAIN_SIZE, test_size=TEST_SIZE)

	model = LinearRegression()
	model.fit(surface_train[:, np.newaxis], loyer_train)

	loyer_pred = model.predict(surface_test[:, np.newaxis])

	model_error = 100 * round((1 - model.score(surface_test[:, np.newaxis], loyer_test)) ,4)
	model_errors.append(model_error)

# enfin faisons les moyennes :

model_error = round(sum(model_errors)/len(model_errors),2)
print("avec un seul feature (surface), apres avoir refait l'operation {} fois, en moyenne : "\
		.format(nb_test) )
print("l'erreur modele est de {}%, \n".format(model_error,))



##################################################
#	PARTIE 4 : Regression Lineaire Multiple
##################################################


# Hypothese 1 : En ajoutant le feature "arrondissement", on peut améliorer le 
# modele

# creons les données train et test
X_train, X_test, loyer_train, loyer_test \
	= train_test_split(df.drop(["loyer"], axis=1), df.loyer, 
		train_size=TRAIN_SIZE, test_size=TEST_SIZE)


# creons et entrainons le modele sur les données train
model = LinearRegression()
# attention les "features" ne sont plus un vecteur, mais bel et bien une matrice
model.fit(X_train,loyer_train)

# calculons les données prédides sur les données test
loyer_pred = model.predict(X_test)

# caluclons les erreurs : 
model_error = 100 * round((1 - model.score(X_test, loyer_test)) ,4)
print("avec deux features (surface, arrondissement), apres avoir refait l'operation une seule fois: ")
print("l'erreur modele est de {}%,\n".format(round(model_error,2)))


# pour ajuster notre calcul, faisons le plusieurs fois et prennons 
# la moynne des erreurs (sera fait ainsi jusqu'à la fin du script)

model_errors = list()
nb_test=NB_TEST

for _ in range(nb_test) : 
	X_train, X_test, loyer_train, loyer_test \
		= train_test_split(df.drop(["loyer"], axis=1), df.loyer, 
			train_size=TRAIN_SIZE, test_size=TEST_SIZE)

	model = LinearRegression()
	model.fit(X_train,loyer_train)

	loyer_pred = model.predict(X_test)

	model_error = 100 * round((1 - model.score(X_test, loyer_test)) ,4)
	model_errors.append(model_error)


# enfin faisons les moyennes : 
model_error = round(sum(model_errors)/len(model_errors),2)
print("avec deux features (surface, arrondissement), apres avoir refait l'operation {} fois, en moyenne : "\
		.format(nb_test) )
print("l'erreur modele est de {}%, \n".format(round(model_error,2)))



# globalement on améliore la perofmance, on passe de +/- 19.5 à +/- 17.5 %
# d'erreur soit 10% d'amélioration.
# Essayons de faire mieux...


# Hypothese 2 : Il est mieux de faire une regression lineaire par arrondissement

for arr in df.arrondissement.unique() : 

	# on ne travaille plus sur df mais df par arrondissement
	df_arrond = df[df.arrondissement == arr]

	model_errors = list()
	nb_test=nb_test

	for _ in range(nb_test) : 
		surface_train, surface_test, loyer_train, loyer_test \
			= train_test_split(df_arrond.surface, df_arrond.loyer, 
				train_size=TRAIN_SIZE, test_size=TEST_SIZE)

		model = LinearRegression()
		model.fit(surface_train[:, np.newaxis], loyer_train)

		loyer_pred = model.predict(surface_test[:, np.newaxis])

	model_error = 100 * round((1 - model.score(surface_test[:, np.newaxis], loyer_test)) ,4)
	model_errors.append(model_error)

	# enfin faisons les moyennes : 
	model_error = round(sum(model_errors)/len(model_errors),2)
	print("apres avoir refait l'operation {} fois, pour l'arr {}, en moyenne : "\
			.format(nb_test, arr) )
	print("l'erreur modele est de {}%".format(round(model_error,2)))

print()

# cela marche tres bien pour le 10e arrondissement mais pas pour le 1er ...
# l'idée n'est pas retenue! 
# Essayons de faire mieux...


# Hypothese 3 : D'autres modeles linéarires sont meilleurs que LinearRegression 


for methode in 	[LinearRegression, Lasso, Ridge, BayesianRidge, LassoCV, LassoLarsCV,
					 HuberRegressor, Lars, LassoLars, RidgeCV, LinearSVR] : 

	model_errors = list()

	for _ in range(nb_test) : 
		X_train, X_test, loyer_train, loyer_test \
			= train_test_split(df.drop(["loyer"], axis=1), df.loyer, 
				train_size=TRAIN_SIZE, test_size=TEST_SIZE)

		model = methode()
		model.fit(X_train,loyer_train)

		loyer_pred = model.predict(X_test)

		model_error = 100 * round((1 - model.score(X_test, loyer_test)) ,4)
		model_errors.append(model_error)

		# enfin faisons les moyennes : 
		modele_error = round(sum(model_errors)/len(model_errors),2)

	print("avec deux features (surface, arrondissement), apres avoir refait l'operation {} fois, en moyenne : ".format(nb_test) )
	print("l'erreur modele de la methode {} est de {}%, \n".format(methode.__name__, round(model_error,2)))


# résultats intéressants car on améliore clairement la prediction ...
# mais si on refait cette opération plusieurs fois, on voit que la "meilleure"
# méthode change souvent, esseayez de relancer le script 5 ou 6 fois et vous 
# verrez ! 


# on va donc changer notre méthode, on va faire tourner la boucle précédente plusieurs
# fois, MAIS au lieu de comparer le meilleur modèle à chaque fois on va compter
# le "gagnant ", c'est a dire celui qui a l'erreur la plus faible le plus de fois
# comme un "concours interne"


first = list()

for i in range(100) : 

	methode_error_list = list()

	for methode in 	[LinearRegression, Lasso, Ridge, BayesianRidge, LassoCV, LassoLarsCV,
					 HuberRegressor, Lars, LassoLars, RidgeCV, LinearSVR] : 

		model_errors = list()
		nb_test = 10

		for _ in range(nb_test) : 
			X_train, X_test, loyer_train, loyer_test \
				= train_test_split(df.drop(["loyer"], axis=1), df.loyer, 
					train_size=TRAIN_SIZE, test_size=TEST_SIZE)

			model = methode()
			model.fit(X_train,loyer_train)

			loyer_pred = model.predict(X_test)

			model_error = 100 * round((1 - model.score(X_test, loyer_test)) ,4)
			model_errors.append(model_error)

		# enfin faisons les moyennes : 
		modele_error = round(sum(model_errors)/len(model_errors),2)

		methode_error_list.append([model_error, methode.__name__])

	methode_error_list.sort()
	print("la meilleure méthode du round {} est {}"\
			.format(i, methode_error_list[0]), end = " * ")

	first.append(methode_error_list[0][1])

print()
plt.hist(first)
plt.show()

first = Counter(first)

print(first)


# On voit qu'on peut améliorer le modele, mais les resulstats ne sont pas tres probants



# Hypothese 4 : Essayons avec des regession polynomiales !!

deg_error_list = list()

for deg in range(7) : 

	model_errors = list()

	for _ in range(500) : 
		surface_train, surface_test, arrond_train, arrond_test, loyer_train, loyer_test \
			= train_test_split(df.surface, df.arrondissement, df.loyer, train_size=0.8, test_size=0.2)

		X = np.array(surface_train)
		y = np.array(loyer_train)

		polynomial_features = PolynomialFeatures(degree=deg, include_bias=True)
		linear_regression = LinearRegression()

		model = Pipeline([("polynomial_features", polynomial_features),
							 ("linear_regression", linear_regression)])

		model.fit(X[:, np.newaxis], y)

		Xt = np.array(surface_test)
		yt = np.array(loyer_test)

		loyer_pred = model.predict(Xt[:, np.newaxis])

		model_error = 100 * round((1 - model.score(X_test, loyer_test)) ,4)
		model_errors.append(model_error)

	# enfin faisons les moyennes : 
	modele_error = round(sum(model_errors)/len(model_errors),2)

	print("avec deux features (surface, arrondissement), apres avoir refait l'operation {} fois, en moyenne : ".format(nb_test) )
	print("l'erreur modele de degré  {} est de {}%, \n".format(deg, model_error,))

	deg_error_list.append([model_error, deg])


print(sorted(deg_error_list))



##################################################
#	PARTIE 5 : Conclusion
##################################################



##################################################
#	PARTIE 6 : Pour aller plus loin ...
##################################################


# Il ya de nombreuses opérations qui auraient pu etre effectuées en plus, 
# on aurait pu par-exemple : 
#  * rajouter les methodes de regression linéarires 'lasso_path' et 'logistic_regression_path'
#  * regarder les méthodes en prenant en compte leurs carratéristiques comme  alpha pour lasso ou 
#  epsilon 
#  * etudier l'impact de la normalisation des données avec la 'normalize'
#  * effectuer un préprocessing sur les features surface et arrondissement pour déterminer
#  leur "impact" relatifs
#  * ajuster le % entre training et testing dataset 
#  * appliquer la methode dite de cross validation pour augmenter la pertinance de nos test
#  * ne pas se fier qu'au score (r2) du modele mais regarder d'autres indicateurs comme
#  la mean_absolute_error, explained_variance_score 
#  * ...




#################################################################



# non concluant, apparement la regression lineaire simple suffit !

				#  Hypothese 5 : Essayons le modele Lasso, mais vaex diférentes valeurs de alpha


				# deg_error_list = list()

				# for a in np.arange(0,1, 0.1) : 

				# model_errors = list()
				# var_errors = list()
				# mean_errors = list()

				# for _ in range(300) : 
				# 	surface_train, surface_test, arrond_train, arrond_test, loyer_train, loyer_test \
				# 		= train_test_split(df.surface, df.arrondissement, df.loyer, train_size=0.8)

				# 	model = Lasso(alpha=a)
				# 	X_train  = pd.DataFrame(dict(surface=surface_train, arrond=arrond_train))
				# 	model.fit(X_train,loyer_train)

				# 	X_test  = pd.DataFrame(dict(surface=surface_test, arrond=arrond_test))
				# 	loyer_pred = model.predict(X_test)

				# 	model_error = 100 * round((1 - model.score(X_test, loyer_test)) ,4)
				# 	model_errors.append(model_error)

				# 	var_error = 100 * round(1 - explained_variance_score(loyer_test, loyer_pred),4)
				# 	var_errors.append(var_error)

				# 	mean_error = mean_absolute_error(loyer_test, loyer_pred)
				# 	mean_errors.append(mean_error)


				# # enfin faisons les moyennes : 
				# model_error = round(sum(model_errors)/len(model_errors),2)
				# var_error = round(sum(var_errors)/len(var_errors),2)
				# mean_error = round(sum(mean_errors)/len(mean_errors),2)
				# deg_error_list.append([model_error, var_error, mean_error, round(a,2)])

				# print(sorted(deg_error_list))
































