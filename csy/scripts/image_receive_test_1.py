#!/usr/bin/python3

import rospy
import base64
import urllib
import os
import numpy as np

from sensor_msgs.msg import Image
import cv2

from cv_bridge import CvBridge
import requests
import json
import dlib
import cv2
import numpy as np
import time
import os

predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
model_path = "dlib_face_recognition_resnet_model_v1.dat"  # resent模型
model = dlib.face_recognition_model_v1(model_path)

# 存放人脸的文件夹
path_know = "/home/jdt/ros_ws/src/csy/people_i_know/faces"  # 已知人脸文件夹
detector = dlib.get_frontal_face_detector()

know_list = {}

# 距离函数
def Eu(a, b):
    return np.linalg.norm(np.array(a) - np.array(b), ord=1)
know_list
# 读取已知人脸数据
for filename in os.listdir(path_know):
    if filename.endswith(".txt"):
        filepath = os.path.join(path_know, filename)
        name = os.path.splitext(filename)[0]
        with open(filepath, 'r') as file:
            vector_str = file.readline()
            vector = np.array([float(num) for num in vector_str.strip().split(",")])
            know_list[name] = vector
            print(vector.size)

def save_face(image):
    img = image
    # 将图片转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测图片中的人脸
    faces = detector(gray)
    # 如果没有检测到人脸，则返回空的特征编码
    if len(faces) == 0:
        return None
    # 获取第一个人脸的特征点
    shape = predictor(gray, faces[0])
    # 计算人脸的特征编码
    face_encoding = np.array(model.compute_face_descriptor(img, shape))
    with open("/home/jdt/ros_ws/src/csy/people_i_know/faces/wjw.txt", "w") as file:
        for i, value in enumerate(face_encoding[:-1]):  # excluding the last element
            file.write(str(float(value)) + ',')
        file.write(str(face_encoding[-1]))

# 人脸特征提取并进行特征向量编码
def feature_get(img):
    for filename in os.listdir(path_know):
        if filename.endswith(".txt"):
            filepath = os.path.join(path_know, filename)
            name = os.path.splitext(filename)[0]
            with open(filepath, 'r') as file:
                vector_str = file.readline()
                vector = np.array([float(num) for num in vector_str.strip().split(",")])
                know_list[name] = vector
                print(vector.size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    for i, face in enumerate(faces):
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 0), 2)
        cv2.imshow('Detected Faces', img)
        cv2.waitKey(1)
        # 获取人脸特征点
        shape = predictor(img, face)
        # landmarks = shape.parts()
        print("第", i + 1, '个人脸特征点:')
        print(shape.parts())
        # 将特征点转换为特征向量
        face_vector = model.compute_face_descriptor(img, shape,)
        print(face_vector)
        # 比较特征向量并进行人脸识别
        min_distance = float('inf')
        closest_name = "unknown"
        potential_name = closest_name
        for name, know_vector in know_list.items():
            distance = Eu(know_vector, face_vector)
            print(distance)
            if distance < min_distance:
                min_distance = distance
                potential_name = name
            print("distance:",distance)
        if min_distance<=4:
            closest_name = potential_name
        else:
            save_face(img)
        # 标注人物名字
        text_location = (face.left(), face.top() - 10)
        cv2.putText(img, closest_name, text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Detected Faces', img)
    cv2.waitKey(1)



def callback_color(imgmsg):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
    feature_get(img)

      
def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/camera/rgb/image_color", Image, callback_color)
    rospy.spin()

if __name__ == '__main__':
    listener()

