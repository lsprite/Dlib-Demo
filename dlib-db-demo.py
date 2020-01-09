import DbManage
import numpy as np
from get_one_face_features import OneFaceFeatures
import json

goff = OneFaceFeatures()
res = goff.get_faces_feature('images/chenduling.jpg')
if res[0] is False:
    print(res[1])
else:
    print("--识别成功")
    feature = res[1]
    try:
        # print(type(np.array(feature).tolist()))
        # print(np.array(feature))
        conn = DbManage.getConn()
        cur = conn.cursor()
        # execute   mogrify
        sql = """INSERT INTO face_features (pname,feature) VALUES(%s,%s)"""
        print(np.array(feature).dtype)
        cur.execute(sql, ('chengling', json.dumps(np.array(feature).tolist())))
        conn.commit()
        print("导入成功")
    except Exception as e:
        print("Unexpected error:", e)
    finally:
        cur.close()
        conn.close()
