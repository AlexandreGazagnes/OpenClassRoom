#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import sqlite3

import sqlite3

# FIRST STEP : CREATION / CONNECTION
try:
    # create a connection with bdd (if not existing, created)
    conn = sqlite3.connect("database.db")
    # create the cursor from connection to interact with bdd
    cursor = conn.cursor()
except Exception as e:
    raise(e)


# 2* STEP : FIRST SQL REQUEST
try:
    # write your request
    req = """CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
                name TEXT, age INTEGER  )"""
    # the cursor execute your request
    cursor.execute(req)
    # don't forget to commit :)
    conn.commit()
    #  of course you have to commit from conn and not from cursor :)
except Exception as e:
    conn.close()
    raise(e)


# LETS INSERT DATAS IN OUR TABLE
try:
    user_0 = None  #  first way
    req = """INSERT INTO users(name, age) VALUES("test",122)"""
    cursor.execute(req)
    print("req 0 OK")

    user_1 = ["alex", 31]  #  2* way
    req = """INSERT INTO users(name, age) VALUES(?,?)"""
    cursor.execute(req, user_1)
    print("req 1 OK")

    user_2 = dict(name="agnes", age=34)
    req = """INSERT INTO users(name, age) VALUES(:name,:age)"""
    cursor.execute(req, user_2)
    print("req 2 OK")

    user_3 = ["antoine", 2]
    req = """INSERT INTO users(name, age) VALUES("{}", {})"""\
        .format(user_3[0], user_3[1])
    cursor.execute(req)
    print("req 3 OK")


except Exception as e:
    print("erreur, on close la connection par securité!")
    conn.close()
    raise(e)


# LETS NOW IMPORT DATAS FROM TABLE

#  first one lign :
try:
    cursor.execute("""SELECT name, age FROM users""")

    # first one lign :

    # user1 = cursor.fetchone()
    # print(user1)
    #
    # user2 = cursor.fetchone()
    # print(user2)
    #
    # or all ligns in one time
    # rows = cursor.fetchall()
    # for row in rows:
    #     print(row)
    #
    # age_max = 3
    # cursor.execute("SELECT * FROM users WHERE age <=?", (age_max,))
    # reponse = cursor.fetchall()
    # for val in reponse:
    #     print(val)

    #  cursor is an iterator, so
    #  reponse = cursor.fetchall() is not usefull :
    age_max = 3
    for val in cursor.execute("SELECT * FROM users WHERE age <=?", (age_max,)):
        print(val)

except Exception as e:
    print("erreur, on close la connection par securité!")
    conn.close()
    raise(e)


# AT THE END TRY TO CLOSE THE CONNECTION IF POSSIBLE :
try:
    conn.close()
    print("OK, all Good")
except exception as e:
    raise(e)
