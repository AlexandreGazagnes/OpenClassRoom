#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# try to use doc test  :

def square(num) :
    """function compute and return square of a number

    positional arguments :
    num :	the number you want to square 
    		type : int
    
    return : 	
    square of argument if int, else None

    doctest :
    >>> square(11)
    121
    >>> square(12.3)
    
    """
    return num **2 if isinstance(num, int) else None




if __name__ == '__main__':
    print(square(11))
    print(square(12.3))


#############################################################
# to use doctest run "python -m doctest [Filename.py]" 
# in your CLI. Possible use of "-v" to verbose
#############################################################
