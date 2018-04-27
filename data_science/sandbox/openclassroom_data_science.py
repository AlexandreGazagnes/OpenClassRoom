#!/usr/bin/env python
#-*- coding: utf8 -*-


# IMPORTS
from __future__ import division

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# create df
df = pd.read_csv(
    "/home/alex/Bureau/calculs_manus/BTCUSD_1H_2017_ok.csv")

print(df.head())

print(str(len(df))+"\n")

print("{}\n".format(max(df.index)))

plt.plot(df.index, df.close)


df = pd.DataFrame({"i": range(10), "i**2": [i**2 for i in range(10)]})
print(df)


for index, ser in df.iterrows():
    print("index = {}".format(index))
    print("ser = {}".format(ser))
    ser["i"] = True if ser["i"] % 2 else False
    print("ser.i = {}".format(ser.i))


df


df["i**2"][3]
df.i[3]
df.loc[3, "i**2"]
df.loc[3, "i**2"] = 122
df


class Test:
    def __init__(self, arg, val):
        self.arg = self.double_arg(arg)

    def double_arg(self, arg):
        if isinstance(arg, int):
            return arg * 2
        return 0


alp = Test(12, 120)
alp
type(alp)

alp.__dict__


df


df.shape
len(df)

df.i.sum()


df


from stockstats import StockDataFrame
from __future__ import division

df = pd.DataFrame()

x = range(100)
low = [i/2 for i in x]
_open = [i/1.3 for i in x]
close = [i*2 for i in x]
high = [i*4 for i in x]
volume = [100 for i in x]

df = pd.DataFrame({"low": low, "open": _open, "close": close, "high": high, "volume": volume})


df = df.round(2)

st = StockDataFrame.retype(df)


st["moyenne_constant"] = 120

st


st["indicator"] = False
st


st.get("close_10_sma")


st


st.get("close_3_sma_xu_close_10_sma")

    a = 12
    b = 13

    for i in [a, b]:
        i = 122


a
a


class Test:

    def __init__(self, arg):

        __setattr__(self, str(arg), int(arg))


a = Test("12")


str(None)


isinstance(12, int)


ORDER_TYPE = (("market", "taker"), ("limit", "maker"))

var = "market"

var in ORDER_TYPE

var in (ORDER_TYPE[0] or ORDER_TYPE[1])


var1 = 12
var2 = 14


# il vaut mieux :
try:
    assert var1 == var2
except:
    raise ValueError("Probleme")

# ou
if not var1 == var2:
    raise ValueError("Probleme")


ORDER_TYPE = ORDER_TYPE[0] + ORDER_TYPE[1]
ORDER_TYPE

text = ", ".join(ORDER_TYPE)
text


class Test:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


t = Test()
t
t.__dict__

t2 = Test(douze=2, b=122, tit="djoiadidao")
t2
print(t2.__dict__)
