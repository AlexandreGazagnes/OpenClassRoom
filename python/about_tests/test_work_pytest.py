#!/usr/bin/env python3
# -*- coding: utf-8 -*-



##########################################
#	test_work_pytest
##########################################



# import 

import pytest



# define setup and teardown

def setup_function(function) : 
	print("before (function) " + str(function))


def teardown_function(function) : 
	print("after (function) " + str(function))



# try to use pytest :

def square(num) :
    """function compute and return square of a number

    positional arguments :
    num :	the number you want to square 
    		type : int
    
    return : 	
    square of argument if int, else None

    """
    return num **2 if isinstance(num, int) else None


def test_square_int() : 
	assert 121 == square(11)


def test_square_other() : 
	assert None == square(1.1)


def test_exception(): 
	with pytest.raises(ZeroDivisionError) :
		1/0


def test_print() : 
	print("use -s to print with test functions")


class Human : 
	def __init__(self, name, sex, age) : 
		self.name = name
		self.sex = sex if sex in ["h", "f", "homme", "femme", 
								  "m", "w", "man", "woman"] \
								  else None
		self.age = age if isinstance(age, int) else None

	def older(self, year) : 
		self.age = self.age + year if isinstance(year, int) else self.age


class TestHuman : 

	NAME = "alex"
	SEX = "man"
	AGE = 33

	def setup_method(function) : 
		print("before (method) " + str(function))


	def teardown_method(function) : 
		print("after (method) " + str(function))


	def test_instance(self) : 
		human = Human(self.NAME, self.SEX, self.AGE)
		assert 		(self.AGE == human.age)\
				and	(self.SEX == human.sex)


	def test_older(self) : 
		human = Human(self.NAME, self.SEX, self.AGE)
		human.older(12)
		assert self.AGE + 12 == human.age



# main 

if __name__ == '__main__':
    print(square(11))
    print(square(12.3))
    # print(1/0)

    tom = Human("tom", "man", 31)
    print(tom.__dict__)
    tom.older(3)
    print(tom.age)



#############################################################
# to use pytest just run "pytest" in your folder with your 
# CLI. Possible use of "-v" to verbose or "-s" for print
#############################################################
