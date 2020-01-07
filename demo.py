# -*- coding: utf-8 -*-
import dlib
from skimage import io
import numpy
import cv2

cnn_face_path = 'model/mmod_human_face_detector.dat'
predictor_path = 'model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'model/dlib_face_recognition_resnet_model_v1.dat'
PATH_FACE = "data/face_img_database/"

# 加载正脸检测器
# detector = dlib.get_frontal_face_detector()
detector = dlib.cnn_face_detection_model_v1(cnn_face_path)
# 加载人脸关键点检测器
predictor = dlib.shape_predictor(predictor_path)
# 加载人脸识别模型
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

img = io.imread('images/chenduling.jpg')
# 1.人脸检测(参数1表示对图片进行上采样一次，有利于检测到更多的人脸)
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
# 检测到人脸
if len(dets) != 0:
    biggest_face = dets[0]
    # 取占比最大的脸
    maxArea = 0
    for det in dets:
        w = det.right() - det.left()
        h = det.top() - det.bottom()
        if w * h > maxArea:
            biggest_face = det
            maxArea = w * h
    shape = predictor(img, det)
    # print("---shape:",shape)
    # landmark = numpy.matrix([[p.x, p.y] for p in shape.parts()])
    # print("landmark:",landmark)
    features_cap = facerec.compute_face_descriptor(img, shape)
    print(numpy.array(features_cap))
    face_height = biggest_face.bottom() - biggest_face.top()
    face_width = biggest_face.right() - biggest_face.left()
    im_blank = numpy.zeros((face_height, face_width, 3), numpy.uint8)
    try:
        for ii in range(face_height):
            for jj in range(face_width):
                im_blank[ii][jj] = img[biggest_face.top() + ii][biggest_face.left() + jj]
        cv2.imencode('.jpg', im_blank)[1].tofile(PATH_FACE + "/img_face_" + "face" + ".jpg")
        print("写入成功")
    except Exception as e:
        print("保存照片异常", e)
