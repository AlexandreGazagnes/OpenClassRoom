#!/usr/bin/env pythonX
# -*- coding: utf-8 -*-



#######################################
#   Exercice OpenClassRoom
#######################################



# Ce script est l'exercice d'OpenClassRoom relatif au cours 
# "initiez vous au machine learning"
# il est relatif au fichier CSV "house.csv"


# le code ci dessous a été écrit de facon à etre le plus lisible possible
# de nombreuses possibilités de factorisation/reduction/contraction du 
# code auraient été possibles mais j'ai préféré écrire un code très facile
# à lire pour tout type de lecteur :) 



# importons les librairies

from collections import Counter

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 
sns.set()

from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor



# définissions quelques constantes 

SHOW = False
NB_TEST = 500
TRAIN_SIZE = 0.75
TEST_SIZE = 0.25



# définissions quelques fonctions de "confort"

def print_t(title):
	"""just a fancy personal print"""

	if title : 
		print("\n##########################################################")
		print("\t" + title)
		print("##########################################################\n")

def print_e(): 
	print()
	print()

def pause() : 
	print("\n\n")
	print_t("continuer?") 
	input()



# Créons le dataframe

df = pd.read_csv("house_data.csv")



############################################
#	PARTIE 1 : Exploration des données
############################################


# dans cette partie nous allons explorer les données sous toutes les coutures
# nous essayons de comprendre comment elle sont dispersées, quel rapport ont
# elle entre elles etc etc


# premiere exploration de nos données : 

print_t("DataFrame info")

print("{}".format(df.info()))
print_e()

print("head and tail : \n{}\n{}".format(df.head(), df.tail()))
print_e()

print("distribution : \n{}".format(df.describe()))
print_e()

print("dtype : \n{}".format(df.dtypes))
print_e()

print("shape : \n{}".format(df.shape))
print_e()

print("ndim : \n{}".format(df.ndim))
print_e()

print_t("DataFrame columns")
print("features : \n{}".format(df.columns))
print_e()


# regardons notamment la colonne "price" : 

print_t("Price")
print(df['price'].describe())
print_e()

print("Skewness: {:.2f}".format(df['price'].skew()))
print("Kurtosis: {:.2f}".format(df['price'].kurt()))
print_e()

sns.distplot(df['price'])
plt.title("distribution du prix")
plt.show()


# étudions la correlation entre les features et price : 

 # d'abord avec les points 
fig, ax = plt.subplots(1,2, figsize=(10,5))
for i, var in enumerate(['surface', 'arrondissement' ]) : 
	ax[i].scatter(x=df[var], y=df['price'], marker=".")
	ax[i].set_title("prix en fonction de {}".format(var))
plt.show()


# ensuite avec la matrice de correlation 
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True, annot=True, cbar=True, fmt='.4f')
plt.title("matrice de correlation entre les différents features")
plt.show()


# enfin en regardant la distribution des prix/surface et prix/arrondissement
data = pd.concat([df['price'], df['surface']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='surface', y="price", data=data)
plt.title("Distribution des prix en fonctions des différentes surfaces")
plt.show()

data = pd.concat([df['price'], df['arrondissement']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='arrondissement', y="price", data=data)
plt.title("Distribution des prix en fonctions des différents arrondissements")
plt.show()



############################################
#	PARTIE 2 : Nettoyage des données
############################################


# dans cette partie nous allons effectuer un certain nombre
# d'opérations sur notre dataframe initial afin de le "nettoyer", 
# et de permettre un travail plus aisé par la suite


# regradons s'il y a des valeurs manquantes

print_t("Recherche des valeurs manquantes")
for k in df.columns : 
	print("nombre de NaN dans la colonne {} : {}".format(k, df[k].isna().sum()))

mask = df["surface"].isna()

print("\nvoici les lignes en question : \n{}".format(df[mask]))
print_e()


# supprimons ces valeurs manquantes

df = df.dropna()


# assurons nous que le travail a été fait

print('après avoir nettoyez nos données :')
for k in df.columns : 
 	print("nombre de NaN dans la colonne {} : {}".format(k, df[k].isna().sum()))

print("notre dataframe est maintenant de dimension {}".format(df.shape))
print_e()


# Supprimons les outliners 

df =  df[df["surface"] < 180]
df =  df[df["price"] < 20000]



# apportons quelques améliorations "personnelles" à notre dataframe

# reindexer notre dataframe
df.index = range(len(df))

# unifier la langue des colonnes (on garde le francais)
df = df.rename(index=str, columns={"price": "loyer"})

# retyper les valeurs d'arrondissement (en int)
df["arrondissement"] = df["arrondissement"].astype("int16")



#####################################################
#	PARTIE 3 : Poursuivons l'exploration des données 
#####################################################


# dans cette partie nous approfondissons l'exploration et la visualisation des 
# données en essaynat de comprendre l'impact de l'arrondissement par rapport aux
# différents prix des appartements


# tentons donc une visualisation 3d

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.surface, df.arrondissement, df.loyer)
ax.set_xlabel("surface") ; ax.set_ylabel("arrondissement")
ax.set_zlabel("loyer")
ax.set_title("loyer en fonction de la surface et de l'arrondissement - vue 3d")

plt.show()


#  en jouant un peu avec les axes, on peut voir que les prix dans le 10e arr 
#  sont moins elevés que ailleurs, à surface identique
# essayons d'une autre facon...


data = df.sort_values(by="loyer")	

data["cat"] = pd.Series(pd.cut(data["loyer"], bins=10, labels = range(10)), dtype = 'int16')
area = data["cat"] * 4 * 10**2

panel = ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "black"]
data['color'] = pd.cut(data["loyer"], bins=len(panel), labels=panel)


plt.scatter(data.surface, data.arrondissement, s=area, c=data["color"],alpha=0.3)
plt.xlabel("surface") ; plt.ylabel("arrondissement")
plt.title("loyer en fonction de la surface et de l'arrondissement - vue 2d")
# plt.legend(handles=range(len(panel)), labels=panel, loc='best')
plt.show()


# il semble que le 10e arrondissement soit effectivement un peu moins cher...
# regardons alors du coté du prix/surface


df["loyer_m2"] = df["loyer"] / df["surface"]

fig = plt.figure(figsize=(15,8))

ax = fig.add_subplot(1,1,1)
ax.boxplot(	[df.loc[df.arrondissement == j, "loyer_m2"] for j in\
				df.arrondissement.unique()])
ax.set_xticklabels(df.arrondissement.unique())
ax.set_xlabel("arrondissement") ; ax.set_ylabel("loyer_m2")
ax.set_title("loyer_m2 par arrondissement")

plt.show()


# difficile d'en déduire quelque chose ! essayons autrement ...

fig, ax = plt.subplots(5,1, figsize=(6,10))
for i, arr in enumerate(df["arrondissement"].unique()):
	data = df[df["arrondissement"] == arr]
	sns.distplot(data["loyer_m2"], bins=20, ax = ax[i])
	ax[i].set_title("distribution prix/m2 pour l'arr : {}".format(arr))
plt.subplots_adjust(hspace=1)
plt.show()


# de facon contre-intuitive, c'est le 3e arr le "moins cher", puis, le 10e
# puis le 1er, le 2e et enfin le 4e...

# sur cette base nous allons ajouter un feature, non pas relatif au prix mais
# au prestige de l'arrondissement 

arr_cat_dict = {3:1, 10:2, 1:3, 2:4, 4:5}
df["cat_arr"] = df["arrondissement"].map(lambda x : arr_cat_dict[x])



##################################################
#	PARTIE 4 Rappel : Regression Lineaire simple
##################################################

print("""
# dans cette partie nous revenons rapidement sur la regression linéaire 
# simple, c'est à dire, juste sur la matrice prix/surface
# ce qui nous interesse notamment, c'est le score de cette regression
""")


# creons notre vecteur features X et notre vecteur target y

X = df["surface"]
X = X[:, np.newaxis]

y = df["loyer"]


# creons les données train et test

X_train, X_test, y_train, y_test \
	= train_test_split(X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE)


# creons et entrainons le modele sur les données train

model = LinearRegression()
model.fit(X_train, y_train)


# calculons les données prédides sur les données test 

loyer_pred = model.predict(X_test)


# caluclons les erreurs : 

model_error = 100 * round((1 - model.score(X_test, y_test)) ,4)
print("avec un seul feature (surface), apres avoir refait l'operation une seule fois: ")
print("l'erreur modele est de {}%,\n".format(round(model_error,2)))


#  pour ajuster notre calcul, faisons le plusieurs fois et prennons 
#  la moynne des erreurs --> Nous ferons ainsi pour le reste de l'exercice ! 

# initions ne nombre de test et le conteneur des erreurs

model_errors = list()
nb_test=NB_TEST


# lancons le modele plusieurs fois

for _ in range(nb_test) : 
	X_train, X_test, y_train, y_test \
		= train_test_split(X,y, train_size=TRAIN_SIZE, test_size=TEST_SIZE)

	model = LinearRegression()
	model.fit(X_train, y_train)

	loyer_pred = model.predict(X_test)

	model_error = 100 * round((1 - model.score(X_test, y_test)) ,4)
	model_errors.append(model_error)


# enfin faisons les moyennes :

model_error = round(sum(model_errors)/len(model_errors),2)
print("avec un seul feature (surface), apres avoir refait l'operation {} fois, en moyenne : "\
		.format(nb_test) )
print("l'erreur modele est de {}%, \n".format(model_error,))


input("continuer?")


##################################################
#	PARTIE 5 : A la recherche du meilleur modèle
##################################################


print("""
# Hypothese 1 : En ajoutant le feature "arrondissement", on peut améliorer le 
# modele
""")

# creons notre matrice features X et notre vecteur target y

y = df["loyer"]
X = df.drop(["loyer",  "cat_arr", "loyer_m2"], axis=1)


# initions ne nombre de test et le conteneur des erreurs

model_errors = list()
nb_test=NB_TEST


# lancons le modele plusieurs fois

for _ in range(nb_test) : 

	X_train, X_test, y_train, y_test \
		= train_test_split(X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE)

	model = LinearRegression()
	model.fit(X_train,y_train)

	loyer_pred = model.predict(X_test)

	model_error = 100 * round((1 - model.score(X_test, y_test)) ,4)
	model_errors.append(model_error)


# enfin faisons les moyennes : 

model_error = round(sum(model_errors)/len(model_errors),2)
print("avec deux features (surface, arrondissement), apres avoir refait l'operation {} fois, en moyenne : "\
		.format(nb_test) )
print("l'erreur modele est de {}%, \n".format(round(model_error,2)))


print(""")
# globalement on améliore un peu la perofmance, on passe de +/- 20 à +/- 18 %
# Essayons de faire mieux...
""")
input("Continuer?")


############################################################################
############################################################################

print("""
# Hypothese 2 : En ajoutant le feature "cat_arr", on peut améliorer le 
# modele
""")

# commencons par redéfinir notre df :

y = df["loyer"]
X = df.drop(["loyer",  "arrondissement", "loyer_m2"], axis=1)
print(X.columns)


# initions ne nombre de test et le conteneur des erreurs

model_errors = list()
nb_test=NB_TEST


# lancons le modele plusieurs fois

for _ in range(nb_test) : 
	X_train, X_test, y_train, y_test \
		= train_test_split(X, y ,train_size=TRAIN_SIZE, test_size=TEST_SIZE)

	model = LinearRegression()
	model.fit(X_train,y_train)

	loyer_pred = model.predict(X_test)

	model_error = 100 * round((1 - model.score(X_test, y_test)) ,4)
	model_errors.append(model_error)


# enfin faisons les moyennes 

model_error = round(sum(model_errors)/len(model_errors),2)
print("avec deux features (surface, arrondissement), apres avoir refait l'operation {} fois, en moyenne : "\
		.format(nb_test) )
print("l'erreur modele est de {}%, \n".format(round(model_error,2)))


print(""")
# globalement on améliore la perofmance, mais le résultat n'es pas tres bon
# Essayons de faire mieux...
""")
input("Continuer?")


############################################################################
############################################################################


print("""
# Hypothese 3 : En appliquant une transformation loarytmique basique on peut ameliorer le 
# modele
""")

# commencons par redéfinir notre df

y = df["loyer"]
X = df.drop(["loyer",  "arrondissement", "loyer_m2"], axis=1)


# appliquons la transformation

y = np.log(y)
X["surface"] = np.log(X["surface"])


# initions ne nombre de test et le conteneur des erreurs

model_errors = list()
nb_test=NB_TEST


# lancons le modele plusieurs fois

for _ in range(nb_test) : 
	X_train, X_test, y_train, y_test \
		= train_test_split(X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE)

	model = LinearRegression()
	model.fit(X_train,y_train)

	loyer_pred = model.predict(X_test)

	model_error = 100 * round((1 - model.score(X_test, y_test)) ,4)
	model_errors.append(model_error)


# enfin faisons les moyennes

model_error = round(sum(model_errors)/len(model_errors),2)
print("apres une transformation logarytmique, avec deux features (surface, cat_arr), apres avoir refait l'operation {} fois, en moyenne : "\
		.format(nb_test) )
print("l'erreur modele est de {}%, \n".format(round(model_error,2)))


print("""
# On améliore enfinn la perofmance, on passe de +/- 17 % à +/- 14 %
# Essayons de faire mieux...
""")
input("Continuer?")


############################################################################
############################################################################


# Hypothese 4 : D'autres modeles linéarires sont meilleurs que LinearRegression 


# commencons par redéfinir notre df

y = df["loyer"]
X = df.drop(["loyer",  "arrondissement", "loyer_m2"], axis=1)


# appliquons la transformation

y = np.log(y)
X["surface"] = np.log(X["surface"])



for methode in 	[LinearRegression, Lasso, Ridge, BayesianRidge, LassoCV, LassoLarsCV,
					 HuberRegressor, Lars, LassoLars, RidgeCV] : 

	model_errors = list()

	# lancons le modele plusieurs fois

	for _ in range(nb_test) : 

		X_train, X_test, y_train, y_test \
			= train_test_split(X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE)

		model = methode()
		model.fit(X_train,y_train)

		loyer_pred = model.predict(X_test)

		model_error = 100 * round((1 - model.score(X_test, y_test)) ,4)
		model_errors.append(model_error)

		# enfin faisons les moyennes : 
		modele_error = round(sum(model_errors)/len(model_errors),2)

	print("avec deux features (surface, arrondissement), apres avoir refait l'operation {} fois, en moyenne : ".format(nb_test) )
	print("l'erreur modele de la methode {} est de {}%, \n".format(methode.__name__, round(model_error,2)))


# résultats intéressants car on améliore clairement la prediction ...
# mais si on refait cette opération plusieurs fois, on voit que la "meilleure"
# méthode change souvent, esseayez de relancer le script 5 ou 6 fois et vous 
# verrez ! 


		# # on va donc changer notre méthode, on va faire tourner la boucle précédente plusieurs
		# # fois, MAIS au lieu de comparer le meilleur modèle à chaque fois on va compter
		# # le "gagnant ", c'est a dire celui qui a l'erreur la plus faible le plus de fois
		# # comme un "concours interne"


		# y = df["loyer"]
		# X = df.drop(["loyer",  "arrondissement", "loyer_m2"], axis=1)


		# first = list()

		# for i in range(100) : 

		# 	methode_error_list = list()

		# 	for methode in 	[LinearRegression, Lasso, Ridge, BayesianRidge, LassoCV, LassoLarsCV,
		# 					 HuberRegressor, Lars, LassoLars, RidgeCV, LinearSVR] : 

		# 		model_errors = list()
		# 		nb_test = 10

		# 		for _ in range(nb_test) : 
		# 			X_train, X_test, y_train, y_test \
		# 				= train_test_split(X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE)

		# 			model = methode()
		# 			model.fit(X_train,y_train)

		# 			loyer_pred = model.predict(X_test)

		# 			model_error = 100 * round((1 - model.score(X_test, y_test)) ,4)
		# 			model_errors.append(model_error)

		# 		# enfin faisons les moyennes : 
		# 		modele_error = round(sum(model_errors)/len(model_errors),2)

		# 		methode_error_list.append([model_error, methode.__name__])

		# 	methode_error_list.sort()
		# 	print("la meilleure méthode du round {} est {}"\
		# 			.format(i, methode_error_list[0]), end = " * ")

		# 	first.append(methode_error_list[0][1])

		# print()
		# plt.hist(first)
		# plt.show()

		# first = Counter(first)

		# print(first)


		# # On voit qu'on peut améliorer le modele, mais les resulstats ne sont pas tres probants

input("continuer?")

############################################################################
############################################################################

print("""
# Hypothese 4 : Essayons avec une regression  regession polynomiales sur surface
""")

# commencons par redéfinir notre df

y = df["loyer"]
X = df["surface"]
X = X[:, np.newaxis]


deg_error_list = list()

# essaons plusieurs degrés

for deg in range(7) : 

	model_errors = list()

	# lancons le modele plusieurs fois

	for _ in range(500) : 
		X_train, X_test, y_train, y_test \
			= train_test_split(X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE)


		polynomial_features = PolynomialFeatures(degree=deg, include_bias=True)
		linear_regression = LinearRegression()

		model = Pipeline([("polynomial_features", polynomial_features),
							 ("linear_regression", linear_regression)])

		model.fit(X_train, y_train)

		loyer_pred = model.predict(X_test)

		model_error = 100 * round((1 - model.score(X_test, y_test)) ,4)
		model_errors.append(model_error)

	# enfin faisons les moyennes : 

	modele_error = round(sum(model_errors)/len(model_errors),2)

	print("avec un features (surface), apres avoir refait l'operation {} fois, en moyenne : ".format(nb_test) )
	print("l'erreur modele de degré  {} est de {}%, \n".format(deg, model_error,))

	deg_error_list.append([model_error, deg])

input("continuer?")


############################################################################
############################################################################


print("""
# Hypothese 5 : Essayons avec une regression plunomiale sur surface ET cat_arr
""")


# commencons par redéfinir notre df

y = df["loyer"]
X = df.drop(["loyer",  "arrondissement", "loyer_m2"], axis=1)

deg_error_list = list()


# essqyaons plusieurs degrés de regression

for deg in range(7) : 

	model_errors = list()

	# lancons le modele plusieurs fois

	for _ in range(300) : 
		X_train, X_test, y_train, y_test \
			= train_test_split(X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE)


		polynomial_features = PolynomialFeatures(degree=deg, include_bias=True)
		linear_regression = LinearRegression()

		model = Pipeline([("polynomial_features", polynomial_features),
							 ("linear_regression", linear_regression)])

		model.fit(X_train, y_train)

		loyer_pred = model.predict(X_test)

		model_error = 100 * round((1 - model.score(X_test, y_test)) ,4)
		model_errors.append(model_error)

	# enfin faisons les moyennes 

	modele_error = round(sum(model_errors)/len(model_errors),2)

	print("avec deux features (surface, arrondissement), apres avoir refait l'operation {} fois, en moyenne : ".format(nb_test) )
	print("l'erreur modele de degré  {} est de {}%, \n".format(deg, model_error,))

	deg_error_list.append([model_error, deg])


input("continuer?")


################################################################
################################################################


# refléchissions un peu , que nous dit le feature 'arrondissement'?
# en fait pas grand chose en soit... ce qui compte un arrondissement est-il plus cherqu
# 'un autre? A surface éalele un appartement dans l'arondissement 3 vaudra moins cher que dans l'arondissment 4
# nous avons chois de donner un score arbitraire à ces arrondissements, cela a bien marché, 
# essayons de pousserla démarche en ne pensant plus à l'arrondissement en tant que tel, mais 
# au prix moyen du m2 par arrondissement, au prix median, et à la dispertion ...
# pour cela nous allon saugmenter notre espace de features ! 


# hypothse 6 il faut augmenter la feature arrondissement et tout passer en log !! 


# creons 3 nouveaux features

df["arr_mean"] = df["arrondissement"].map(lambda x : \
				df.loc[df["arrondissement"] == x, "loyer"].mean())

df["arr_med"] = df["arrondissement"].map(lambda x : \
				df.loc[df["arrondissement"] == x, "loyer"].median())


df["arr_std"] = df["arrondissement"].map(lambda x : \
				df.loc[df["arrondissement"] == x, "loyer"].std())


# supprimons les features inutiles du dataframe

df.drop(["arrondissement", "loyer_m2", "cat_arr"], axis=1, inplace=True	)
print(df.columns)


# creions notre matrice de features X et notre vecteur target y

y = df["loyer"]
X = df.drop(["loyer"], axis=1)


# appliquons la transformation "classique"

y = np.log(y)
X = np.log(X)


# initions le conteneur d'erreurs et le nombre de tests
model_errors = list()
nb_test=NB_TEST


# lancons le modele plusieurs fois

for _ in range(nb_test) : 
	X_train, X_test, y_train, y_test \
		= train_test_split(X, y ,train_size=TRAIN_SIZE, test_size=TEST_SIZE)

	model = LinearRegression()
	model.fit(X_train,y_train)

	loyer_pred = model.predict(X_test)

	model_error = 100 * round((1 - model.score(X_test, y_test)) ,4)
	model_errors.append(model_error)


# enfin faisons les moyennes 

model_error = round(sum(model_errors)/len(model_errors),2)
print("avec 4 features (surface, arr_mean, arr_std, arr_med), apres avoir refait l'operation {} fois, en moyenne : "\
		.format(nb_test) )
print("l'erreur modele est de {}%, \n".format(round(model_error,2)))


print(""")
# globalement on améliore la perofmance, on passe de +/- 20 à +/- 10 %
# Essayons de faire mieux...
""")
input("Continuer?")



# hypothese 6 bis essayons avec KNN Regressor 


model_errors = list()
nb_test=NB_TEST

for _ in range(100) : 


	errors= list()
	k_range = range(1, 50)

	for k in k_range : 

		# split into train and test
		X_train, X_test, y_train, y_test = train_test_split(X, y, 
			test_size=TEST_SIZE, train_size=TRAIN_SIZE)

		# instantiate learning model (k = 3)
		knn = KNeighborsRegressor(n_neighbors=k)

		# fitting the model
		knn.fit(X_train, y_train)

		# predict the response
		pred = knn.predict(X_test)
		# 	print(pd.DataFrame(dict(predict=pred, tested=y_test)))

		# evaluate accuracy
		errors.append(100 * round(1 - knn.score(X_test, y_test),4))


	errors = list(zip(k_range, [round(e,2)for e in errors]))
	# errors.sort(key = lambda x : x[0])
	model_errors.extend(errors)


# affichons la distribution des erreurs en fonction des K

x_coord, y_coord = zip(*model_errors)
x_coord = pd.Series(x_coord, name="k")
y_coord = pd.Series(y_coord, name="error")
data = pd.concat([x_coord, y_coord], axis=1)

f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='k', y="error", data=data)
plt.show()


print("clairement non concluant")

input("continuer?")


# hypothse 6 ter essayons d'autres methodes de reression, en profitant de la 
# Cross validation

# on essaye que 4 methodes...
for Method in [LassoCV, LassoLarsCV, LarsCV, RidgeCV ] :

	model_errors = list()
	nb_test=NB_TEST

	for _ in range(nb_test) : 
		X_train, X_test, y_train, y_test \
			= train_test_split(X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE)

		model = Method(cv=10)
		model.fit(X_train,y_train)

		loyer_pred = model.predict(X_test)

		model_error = 100 * round((1 - model.score(X_test, y_test)) ,4)
		model_errors.append(model_error)


	# enfin faisons les moyennes

	model_error = round(sum(model_errors)/len(model_errors),2)
	print("apres une transformation logarytmique, avec 4 features (surface, moy, std, med _arr), apres avoir refait l'operation {} fois, en moyenne : "\
			.format(nb_test) )
	print("l'erreur modele est de {:.2f}% pour la methode {} \n".format(model_error, Method.__name__))


	print("""
	# globalement on améliore la perofmance, on passe de +/- 10 % à +/- 8 %
	# Essayons de faire mieux...
	""")
input("Continuer?")




# ##################################################
# #	PARTIE 6 : Conclusion
# ##################################################


# nous avons réussi à diminuer le % d'erreur de 20% sur une regression lineaire
# de base à 12% en appliquant différentes transformations
# finalement, le travail qui a été le plus important est bien le travail sur les
# features et non sur le modele en tant que tel
# nous retenons donc finalement le model que  regression linéaire "simple"
# mais appliqué à des données tranfromées


# enjoy :) 



# ##################################################
# #	PARTIE 7 : Pour aller plus loin ...
# ##################################################


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

