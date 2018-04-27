#!/usr/bin/env python3
#-*- coding: utf8 -*-



###############################
#   Maze.py
###############################



#########################
#   CLASS
#########################


class Maze(list) : 
	"""
	ieodjoedjzidz
	djieozdjozejdezd
	jdiezdoeodzdjz
	dkoezpopezdpzdkz
	jdoezidzoidjeo
	"""

	def __init__(self, txt) :
		""" init the maze
		positional args : 
			txt		: the str version of the labyrinth store in ./maps/
					type : str

		"""

		arr = [list(i) for i in txt.splitlines()]
		list.__init__(self, arr)
	
		self.dim = (len(self), len(self[0]))
		self.hidden_maze_point = " "

		self.check_maze()
		self.exit_position = self.find_item_position("U")
		self.robot_position = self.find_item_position("X")


	def check_maze(self) : 
		"""chechk the correctness of the maze : 
		- is there a robot and an exit?
		- is there any holes in the walls
		- are we sre there a solution to this maze?
		postional args :	- 
		
		return :			- 
		
		raise  : 			Value error if no robot or no exit
		"""

		robot_number = [i for line in self for i in line if i=="X"]
		exit_number = [i for line in self for i in line if i=="U"]

		if not (len(robot_number) == 1 and len(exit_number) == 1): 
			raise ValueError("Maze.check_maze : Le labyrinthe n'est pas conforme")


	def find_item_position(self, char) : 
		"""method for searching in a Maze a specific character lique 'U'
		for exit or 'X' for robot

		positional args : 
			char :		the character searched in the maze
						type : str, exemple : 'U' or 'X' 

		return :		tuple (x, y) of the char position in the maze 
		
		raise :			ValueError if char not founded
		"""

		for i, line in enumerate(self) : 
			for j, col in enumerate(line) : 
				if self[i][j] == char : 
					return (i,j)
		raise ValueError("Maze.find_item_position : {} non trouvé".format)


	def moove_robot(self, direction, distance) : 
		"""
		lzeded
		dezdzedze
		ddezdzdze
		"""
		
		# check first if it is possible to apply user's moove
		if self.__possible_moove(direction, distance) :

			# if OK, then "do" the moove on the maze
			self.__apply_moove(direction, distance)
		else :
			print("sorry this moove is not possible ! ") 


	def __possible_moove(self, direction, distance) : 
		""" evaulate the corectnes of a user asked moove
		positional args : 
			direction: 	the direction of the moove, North, South, West, Est
						type str in ["S", "N", "W", "E"]
		
			distance:	the number of "cases" the robot will moove 
						type int, exemple 1, 2, 3 ...	

		return  : 	True if the moove is posssible ndlr (not outside the 
					maze, not in a wall... ) else False
						type boolean True / False

		raise : 	- 
		
		"""
		
		# local values for line and columns
		line, col = self.robot_position

		# try vertical moove
		for lett, incr in [("S", 1), ("N", -1)] : 
			if direction == lett : 

				# we need to check every step of a moove, step by step 
				for _ in range(distance) : 
					try : 
						line +=incr

						# if it is wall, no moove possible, return False
						if self[line][col] == "0" : 
							return False
					except : 
						return False

		# try horizontal moove
		for lett, incr in  [("W", -1), ("E", 1)]: 

			# we need to check every step of a moove, step by step
			if direction == lett : 
				for _ in range(distance) : 
					try : 
						col +=incr

						# if it is wall, no moove possible, return False
						if self[line][col] == "0" : 
							return False
					except : 
						return False
		
		# if everythin is fine, return True  
		return True


	def __apply_moove(self, direction, distance) : 
		"""	oio,lk,l,l,
		opkpppokp
		kkoppko
		popiojijiokoji
		"""
		
		# local values for line and columns
		line, col = self.robot_position

		# apply vertical moove
		for lett, incr in [	("S", distance), ("N", -distance)]: 
			if direction == lett : 
				
				# reload the point wich was hidden
				self[line][col] = self.hidden_maze_point
				line += incr
				
				# save the new hiden point for next moove
				self.hidden_maze_point = self[line][col]

				# update robot postion 
				self.robot_position = (line, col)
				self[line][col] = "X"

		# apply horizontal moove
		for lett, incr in [("W", -distance), ("E", distance)] : 
			if direction == lett : 
				# reload the point wich was hidden
				self[line][col] = self.hidden_maze_point
				col += incr
				
				# save the new hiden point for next moove
				self.hidden_maze_point = self[line][col]

				# update robot postion 
				self.robot_position = (line, col)
				self[line][col] = "X"


	def check_win(self) : 
		""" Maze method to check if the game is won or not
		positional args : 	-
		
		return 			: 	True if player find the exit else False
						  	type boolean True/False
		
		raise 			:	-
		"""
		
		if self.robot_position == self.exit_position : 
			return True 
		return False


	def __repr__(self) : 
		""" build a string representation of the global maze
		positional agrs : 	-
		return 	:		 	a formated form of maze to be printed
							type str
		raise : 			- 
		"""
		txt = "\n"
		for line in self : 
			txt= txt + "".join(line) +"\n"
		return txt

