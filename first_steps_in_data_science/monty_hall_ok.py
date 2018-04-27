#!/usr/bin/env python3
# -*- coding: utf-8 -*-



###################################
# Monthy Hall python simulation
###################################



# import 

# get_ipython().run_line_magic('matplotlib', 'inline')
from random import randint
from enum import Enum
import matplotlib.pyplot as plt



# class

class Strategie(Enum) : 
    CHANGER = 1
    GARDER = 0



# function 

def play_game(strategie) : 
    portes =  [0, 1, 2]
    
    bonne_porte = randint(0,2)
    
    premier_choix = randint(0,2)
    
    portes.remove(premier_choix)
    
    if premier_choix == bonne_porte : 
        portes.remove(portes[randint(0,1)])
    else : 
        portes = [bonne_porte]
    
    deuxieme_choix = 0
    if strategie == Strategie.GARDER : 
        deuxieme_choix = premier_choix
    else : deuxieme_choix = portes[0]
        
    return deuxieme_choix == bonne_porte


def play(strategie, nb_tours) : 
    return [1 if play_game(strategie) else 0 for _ in range(nb_tours)]



# main 

if __name__ == '__main__':

	# play_game(Strategie.CHANGER)

	print(sum(play(Strategie.CHANGER, 1000)))

	plt.plot(play(Strategie.CHANGER, 100))
	plt.show()

	plt.bar([1,2], [sum(play(Strategie.CHANGER, 1000)),sum(play(Strategie.GARDER, 1000))])
	plt.show()