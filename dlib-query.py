import DbManage
import numpy as np
import json

try:
    # print(type(np.array(feature).tolist()))
    # print(np.array(feature))
    conn = DbManage.getConn()
    cur = conn.cursor()
    # execute   mogrify
    sql = """SELECT * FROM face_features"""
    cur.execute(sql)
    results = cur.fetchall()
    print('查询完成')
    for row in results:
        print(row[2])
        print(type(row[2]))
        print(np.array(json.loads(bytes.decode(bytes(row[2].encode())))))
except Exception as e:
    print("Unexpected error:", e)
finally:
    cur.close()
    conn.close()
