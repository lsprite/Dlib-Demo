import psycopg2


def getConn():
    conn = psycopg2.connect(database="dlib", user="postgres", password="4434881751", host="127.0.0.1",
                            port="5432")
    return conn