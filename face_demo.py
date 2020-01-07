# -*- coding: utf-8 -*-
import face_recognition
from PIL import Image, ImageDraw
import numpy as np

facial_features = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip', 'left_eye', 'right_eye',
                   'top_lip', 'bottom_lip']
# 读图片
# image = face_recognition.load_image_file("/home/pi/py_worksapce/face/nba1996.jpg")
image = face_recognition.load_image_file("nba1996.jpg")
# 人脸定位
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1)
# 人脸特征点识别
face_landmarks_list = face_recognition.face_landmarks(image, face_locations)
print len(face_landmarks_list)
# 图片处理、展示
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)
for (top, right, bottom, left), face_landmarks in zip(face_locations, face_landmarks_list):
    # 描绘脸部特征
    for facial_feature in facial_features:
        d.line(face_landmarks[facial_feature], width=5)
    # 框住人脸
    d.rectangle(xy=[(left, top), (right, bottom)], outline=(255, 0, 0))

# 改变图片大小，缩小为原来的1/2
pil_image.thumbnail(image.shape * np.array(0.5), Image.ANTIALIAS)
pil_image.show()
