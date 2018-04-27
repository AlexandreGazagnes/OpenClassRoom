#!/usr/bin/env python3
# -*- coding: utf-8 -*-



###########################################
#  	Exercice : Regression Lineaire
###########################################



# Le code présenté a été volontairement rédigé
# de la facon la plus "simple" et la plus
# "lisible" possible.
# Il aurait pu être très facilement factorisé
# et condensé, mais j'ai préféré privilégier
# le facteur lisibilité pour le correcteur :)



# D'abord, importons les libraires nécessaires : 

import pandas as pd 
import numpy as np 
import sklearn.linear_model as sk
import matplotlib.pyplot as plt
import seaborn as sns



# Créons le dataframe avec pandas : 

df = pd.read_csv("hubble_data.csv")



# Affichons les valeurs "brutes" : 

print(df)



# Affichons un premier graphique :

plt.scatter(df.distance, df.recession_velocity, marker="o")
plt.title("Hubble galaxy observations - datas")

plt.rcParams.update({"font.size":15})
plt.xlabel("distance")
plt.ylabel("recession velocity")
sns.set()
plt.style.use("seaborn-white")
plt.legend(loc="upper left")

plt.show()



# Créons le modèle de regression : 

model = sk.LinearRegression(fit_intercept=True)



# Nous devons modifier la forme de nos valeurs en abscisse (x) : 

reshaped_distance = df.distance[:, np.newaxis]



# On peut ensuite effectuer la régression linéaire : 

model.fit(reshaped_distance, df.recession_velocity)



# Amusons nous à imprimmer l'équation de la droite de régression : 

a = round(model.coef_[0], 2)
b = round(model.intercept_, 2)
print("la droite de régression est de forme y = a.x + b, où a = {} et b = {}"\
	  .format(a, b))



# Déterminons ensuite le scope en abscisse (x) pour la droite de régression : 

x_min, x_max = min(df.distance), max(df.distance)
x_val = np.linspace(x_min, x_max, 100)
x_val = x_val[:, np.newaxis]



# Calculons les valeurs en ordonnées (y) correspondantes  : 

y_val = model.predict(x_val)



# (Non demandé) Affichons le graphique de régression linéaire : 

plt.plot(x_val, y_val, label="recession velocity")
plt.title("Hubble galaxy observations - linear regression")

plt.rcParams.update({"font.size":15})
plt.xlabel("distance")
plt.ylabel("recession velocity")
sns.set()
plt.style.use("seaborn-white")
plt.legend(loc="upper left")

plt.show()



# Et enfin, affichons le graphique final : 

plt.scatter(df.distance, df.recession_velocity,	marker=".", \
            color="black", label="raws datas")
plt.plot(x_val, y_val,color="red", label="linear regression")
plt.title("Hubble galaxy observations - datas and linear regression")

plt.rcParams.update({"font.size":15})
plt.xlabel("distance")
plt.ylabel("recession velocity")
sns.set()
plt.style.use("seaborn-white")
plt.legend(loc="upper left")

plt.show()
