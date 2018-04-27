#!/usr/bin/env python3
#-*- coding: utf8 -*-



###############################
#   end.py
###############################


#########################
#   IMPORTS
#########################

import os 
import pickle 



#########################
#   FUNCTIONS
#########################


def delete_save() : 
	try : 
		os.remove("./save/save.pk")
	except : 
		print("No ./save/save.pk")

	


def reload_or_restart() : 
	try :
		open("./save/save.pk", "rb")

		choice = ask_if_reload()

		if choice : 
			return load_saved_game()
	except : 
			print("Pas de sauverade enregistée, on commence une nouvelle partie")
			return None
	return None




def ask_if_reload() : 
	possibles_choices = ["Recommencer une partie", "charger partie précédente"]
	choices_dict = {i:j for i,j in possibles_choices.items()}
	print("Que voulez vous faire? ")
	[print("\t {} : {}".format(i, j)) for i,j in choices_dict.items()]


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
















def save_game(maze) : 
	"""
	izoozaaidzaoidza
	izaoiadaziadaido
	azidijodojiadjoi
	"""

	with open("./save/save.pk", "wb") as f : 
		pickler = pickle.Pickler(f)
		pickler.dump(maze)


def load_saved_game() : 
	"""
	izoozaaidzaoidza
	izaoiadaziadaido
	azidijodojiadjoi
	"""

	with open("./save/save.pk", "rb") as f : 
    	depickler = pickle.Unpickler(f)
		return depickler.load()
