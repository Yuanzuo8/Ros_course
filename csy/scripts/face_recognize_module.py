
import cv2
import dlib
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



class FaceRecognizer:
    def __init__(self):
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.model_path = "dlib_face_recognition_resnet_model_v1.dat"
        self.path_know = "/home/jdt/ros_ws/src/csy/people_i_know"  # 已知人脸文件夹
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.model = dlib.face_recognition_model_v1(self.model_path)
        self.know_list = self.load_known_faces()
        self.closest_name=None
        self.text_location=None
    def Eu(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b), ord=1)
    def load_known_faces(self):
        know_list = {}
        for filename in os.listdir(self.path_know):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.path_know, filename)
                name = os.path.splitext(filename)[0]
                with open(filepath, 'r') as file:
                    vector_str = file.readline()
                    vector = np.array([float(num) for num in vector_str.strip().split(",")])
                    know_list[name] = vector
        return know_list
    def save_face(self,image):
        img = image
        # 将图片转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 检测图片中的人脸
        faces = self.detector(gray)
        # 如果没有检测到人脸，则返回空的特征编码
        if len(faces) == 0:
            return None
        # 获取第一个人脸的特征点
        shape = self.predictor(gray, faces[0])
        # 计算人脸的特征编码
        face_encoding = np.array(self.model.compute_face_descriptor(img, shape))
        with open("/home/jdt/ros_ws/src/csy/people_i_know/faces/wjw.txt", "w") as file:
            for i, value in enumerate(face_encoding[:-1]):  # excluding the last element
                file.write(str(float(value)) + ',')
            file.write(str(face_encoding[-1]))

# 人脸特征提取并进行特征向量编码
    def feature_get(self,img):
        for filename in os.listdir(self.path_know):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.path_know, filename)
                name = os.path.splitext(filename)[0]
                with open(filepath, 'r') as file:
                    vector_str = file.readline()
                    vector = np.array([float(num) for num in vector_str.strip().split(",")])
                    self.know_list[name] = vector
                    # print(vector.size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        closest_name=None
        text_location=None
        for i, face in enumerate(faces):
            # cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 0), 2)
            # cv2.imshow('Detected Faces', img)
            # cv2.waitKey(1)
            
            # 获取人脸特征点
            shape = self.predictor(img, face)
            # landmarks = shape.parts()
            # print("第", i + 1, '个人脸特征点:')
            # print(shape.parts())
            # 将特征点转换为特征向量
            face_vector = self.model.compute_face_descriptor(img, shape,use_gpu=True)
            # print(face_vector)
            # 比较特征向量并进行人脸识别
            min_distance = float('inf')
            closest_name = "unknown"
            potential_name = closest_name
            for name, know_vector in self.know_list.items():
                distance = self.Eu(know_vector, face_vector)
                # print(distance)
                if distance < min_distance:
                    min_distance = distance
                    potential_name = name
                # print("distance:",distance)
            if min_distance<=4:
                self.closest_name = potential_name
            else:
                # print("人脸未知吧偶你")
                self.save_face(img)
            # 标注人物名字
            self.text_location = (face.left(), face.top() - 10)
            
        return [self.closest_name,self.text_location]

