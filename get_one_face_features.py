import dlib
import numpy as np
from skimage import io


class OneFaceFeatures:
    _predictor_path = 'model/shape_predictor_68_face_landmarks.dat'
    _face_rec_model_path = 'model/dlib_face_recognition_resnet_model_v1.dat'

    def __init__(self):
        # 加载正脸检测器
        self._detector = dlib.get_frontal_face_detector()
        # 加载人脸关键点检测器
        self._predictor = dlib.shape_predictor(self._predictor_path)
        # 加载人脸识别模型
        self._facerec = dlib.face_recognition_model_v1(self._face_rec_model_path)

    def get_faces_feature(self, image_path):
        img = io.imread('images/chenduling.jpg')
        # 1.人脸检测(参数1表示对图片进行上采样一次，有利于检测到更多的人脸)
        dets = self._detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        # 检测到人脸
        if len(dets) > 0:
            if len(dets) == 1:
                biggest_face = dets[0]
                shape = self._predictor(img, biggest_face)
                features_cap = self._facerec.compute_face_descriptor(img, shape)
                return [True, features_cap]
            else:
                return [False, '识别到多个人']
        else:
            return [False, '未识别到人脸']

    @staticmethod
    def face_compare(feature_1, feature_2, assess=0.4):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        print("欧式距离: ", dist)
        if dist > assess:
            return False
        else:
            return True

    @staticmethod
    def face_compare_nparray(feature_1, feature_2, assess=0.4):
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        print("欧式距离: ", dist)
        if dist > assess:
            return False
        else:
            return True
