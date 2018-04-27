#!/usr/bin/env python3
#-*- coding: utf8 -*-



###############################
#   user_moove.py
###############################



#########################
#   FUNCTIONS
#########################


def ask_moove() :
	autorized_directions = ["S", "N", "E", "W"]
	while True :
		ans = input("moove?\n")
		direction, distance = ans[0], ans[1:]

		direction = direction.upper()
		if direction not in autorized_directions : 
			print("Désolé, mais la direction doit etre {}"\
				.format(",".join(autorized_directions)))
			continue
		try : 
			distance = int(distance)
		except : 
			print("Désolé mais la distance doit etre un entier (0,1,2,3...)")
			continue
		break
	return (direction, distance)

