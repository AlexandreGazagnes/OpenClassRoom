#!/usr/bin/env python3
#-*- coding: utf8 -*-



###############################
#   map_loader.py
###############################



#########################
#   IMPORTS
#########################


import os



#########################
#   FUNCTIONS
#########################


def map_loader() : 
	"""
	"""

	with open("./maps/"+ask_map(), "r") as f : 
		return f.read()

		
def ask_map() : 
	"""
	"""

	maps_dispo = possibles_maps()
	correct_choices = list(zip(*maps_dispo.items()))[0]
	choices_str = ",".join([str(i) for i in correct_choices])
	print("Quelle labyrinthe voulez vous choisir ? \nréponse = {}".\
		format(choices_str))
	
	while True : 
		ans = input()
		try : 
			ans = int(ans)
			if ans in maps_dispo.keys() : 
				break
			else : 
				print("Désolé, réponses acceptées = {}".format(choices_str))
		except : 
			print("Désolé, réponses acceptées = {}".format(choices_str))
	return maps_dispo[ans]


def possibles_maps() :
	"""
	"""

	maps_dispo = {i: j  for i,j in enumerate(os.listdir("./maps/"))}
	print("Voici les labyrinthes disponibles : ")
	[print("\t {} : {}".format(i, j)) for i,j in maps_dispo.items()]
	print()
	return maps_dispo

