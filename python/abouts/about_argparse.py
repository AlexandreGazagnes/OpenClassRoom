#!/usr/bin/env python



import argparse


def args_manager(string) : 
	if not string : 
		return None 
	else : 
		return string

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--string",  help='the string to transform')

parser.add_argument("-v", "--verbose", help="show graph in web browser",
                          action="store_true")


args = parser.parse_args()


print(args.string)
print(args_manager(args.string))
