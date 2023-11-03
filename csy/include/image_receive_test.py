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

##导入两个模型
predictor_path = "/home/ustinian/catkin_ws/src/image_receive/src/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
model_path = "/home/ustinian/catkin_ws/src/image_receive/src/dlib_face_recognition_resnet_model_v1.dat" # resent模型
model = dlib.face_recognition_model_v1(model_path)

##两个存放人脸的文件夹
path_know = "/home/ustinian/catkin_ws/src/image_receive/src/people_i_know" # 已知人脸文件夹
path_unknow = "/home/ustinian/catkin_ws/src/image_receive/src/unknow_people" # 未知人脸文件夹
detector = dlib.get_frontal_face_detector()

know_list = {}

##距离函数
def Eu(a, b):
    return np.linalg.norm(np.array(a) - np.array(b), ord=1)

##读取已知人脸数据
for filename in os.listdir(path_know):
    if filename.endswith(".txt"):
        filepath = os.path.join(path_know, filename)
        name = os.path.splitext(filename)[0]
        with open(filepath, 'r') as file:
            vector_str = file.readline()
            vector_list = vector_str.strip()[1:-1].split(',')
            vector = np.array([tuple(map(int, point.strip()[1:-1].split(','))) for point in vector_list])
            know_list[name] = vector


##人脸年龄估计：
def get_token():
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=CZlHcyjavzoiqM7hEsBMaNDK&client_secret=2ouD8L6XGNrmsL4v6uSwBpatPitl1As1"

    payload = ""
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'

    }

    response = requests.request("POST", url, headers=headers, data=payload)
    data = response.json()
    access_token = data["access_token"]
    print(access_token)
    return access_token


def get_detect(img):
    url = "https://aip.baidubce.com/rest/2.0/face/v3/detect?access_token=" + get_token()
    cv2.imwrite('1.jpg', img)
    image_name = "1.jpg"
    image = get_file_content_as_base64(image_name, False)
    payload = json.dumps({
        "image": image,
        "max_face_num": 10,
        "image_type": "BASE64",
        "face_field": "faceshape,facetype,age,gender,glasses"
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    detect = response.json()
    return detect


def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content


def draw(detect, image):
    face_list = detect["result"]["face_list"]
    for face in face_list:
        left = int(face["location"]["left"])
        top = int(face["location"]["top"])
        width = int(face["location"]["width"])
        height = int(face["location"]["height"])

        # 绘制检测框
        cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 10)

        # 构造标签字符串
        label = f"Age: {face['age']}, Gender: {face['gender']['type']}"

        # 绘制标签文本
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 显示带有检测框和标签的图片
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

##人脸特征提取并进行特征向量编码
def feature_get(img):
    cv2.imshow('Detected Faces', img)
    cv2.waitKey(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    for i, face in enumerate(faces):
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 0), 2)
        cv2.imshow('Detected Faces', img)
        cv2.waitKey(1)

        # 获取人脸特征点
        shape = predictor(img, face)
        landmarks = shape.parts()

        # 将特征点转换为特征向量
        vector = np.zeros(136)
        for j, landmark in enumerate(landmarks):
            vector[j*2] = landmark.x
            vector[j*2+1] = landmark.y

        # 比较特征向量并进行人脸识别
        min_distance = float('inf')
        closest_name = "unknown"
        for name, know_vector in know_list.items():
            distance = Eu(know_vector, vector)
            if distance < min_distance:
                min_distance = distance
                closest_name = name

        # 标注人物名字
        text_location = (face.left(), face.top() - 10)
        cv2.putText(img, closest_name, text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 显示带有人脸识别结果的图片
    cv2.imshow('Detected Faces', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
##订阅话题的回调函数
def callback(imgmsg):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
    ##人脸特征年龄性别识别
    # detect=get_detect(img)
    # draw(detect,img)
    feature_get(img)
    # cv2.waitKey(3)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/usb_cam/image_raw", Image, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
if __name__ == '__main__':
    listener()

