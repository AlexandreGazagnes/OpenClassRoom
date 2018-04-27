#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################
#       iterator_generator
###############################################################


print("# using iterators  without knowlege :")

l = [0, 1, 2, 3, 4]
print(l)
print(type(l))

for i in l:
    print(i)


###############################################################
###############################################################


print("# Ok for a list, but what is/does 'range' function?")

g = range(5)
print(g)
print(type(g))

for i in g:
    print(i)


###############################################################
###############################################################


print("# Try to find out iterator works")

it = iter(l)
print(it)
print(type(it))

while True:
    try:
        print(next(it))
    except StopIteration:
        break
    except:
        print("Unknown error")
        break


###############################################################
###############################################################


print("# so let's define our own iterator object :")


class MyListIterator:

    def __init__(self, seq):
        if not isinstance(seq, list):
            raise TypeError("Ne fonctionne qu'avec des listes :)")
        self._seq = seq
        self._len = len(seq)
        self._count = 0

    def __iter__(self):
        print("create iterator first")
        return self

    def __next__(self):
        print("call next")
        if self._count < self._len:
            elem = self._seq[self._count]
            self._count += 1
            return elem
        else:
            raise StopIteration()


print(l)
print(type(l))
for i in MyListIterator(l):
    print(i)


###############################################################
###############################################################


print("# Shoter version, using yied (auto iterator creator)")


def MyListGenerator(seq):
    if not isinstance(seq, list):
        raise TypeError("Ne fonctionne qu'avec des listes :)")

    stop = len(seq) - 1
    i = -1

    while i < stop:
        i += 1
        yield seq[i]


print(l)
print(type(l))
for i in MyListGenerator(l):
    print(i)
