#!/usr/bin/env python3
#-*- coding: utf8 -*-



###############################
###############################
#   roboc.py
###############################
###############################



# roboc.py est un programme simple et facile d'utilisation. 
# il s'agit d'un jeu de labyrinthe version débutant.
# vous controlez un petit robot qui essaye de sortir d'un labyrinthe...



#########################
#   IMPORTS
#########################


import os

from libs.save import * 
from libs.intro import * 
from libs.map_loader import * 
from libs.Maze import * 
from libs.user_moove import * 
from libs.end import * 
from libs.save import * 


#########################
#   FUNCTIONS
#########################


def main() : 

	os.system('clear')
	intro()
	maze = reload_or_restart() :
	if not maze : 
		raw_maze = map_loader()
		maze = Maze(raw_maze)
	os.system('clear')
	print(maze)

	while True : 
		moove = ask_moove()
	
		if not moove : 
			save_game(maze)
			break

		maze.moove_robot(*moove)
		win = maze.check_win()
		save_game(maze)
		os.system('clear')
		print(maze)
		
		if win : 
			print("You win")
			delete_save()
			break

	end()



#########################
#   MAIN
#########################


if __name__ == '__main__' :
	main()
