#!/usr/bin/python3

import rospy
import base64
import urllib
import os
import numpy as np

from sensor_msgs.msg import Image

import cv2
import dlib
from cv_bridge import CvBridge
import requests
import json
import dlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
import cv2
import paddlehub as hub
import PIL.Image as Image_pil
from PIL import ImageSequence
from IPython.display import display, HTML
import numpy as np
import imageio
import os
import time
import os
import rospy
import smach
import smach_ros
import wave
import pyaudio
import threading
from concurrent.futures import ThreadPoolExecutor
from face_recognize_module import FaceRecognizer
from segment_module import PeopleSegmentation
from pydub import AudioSegment
from pydub.playback import play
import rospy
from recognize_model import FaceRecognition
from api_module import GetPersonalInformation
from navigate import Navigator
from geometry_msgs.msg import PoseStamped
from face_detected import FaceDetector
import insightface

from sklearn import preprocessing
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
entrance = True
entrance_position=[1,0.306,0,0,0,1,-0.02]

next = False #是否完成此状态下一任务
is_find = False #是否找到含有大面积人体图片
img_color = None #颜色图
binary_mask=None #
contours=None
depth_image = None
distance_image = None
closest_name=None
text_location=None
seg_switch = True#人体分割是否进行
face_switch = False#人脸检测是否进行
recognize_switch = False#人脸识别是否进行
scaling_factor = 0.001  # 缩放因子，将深度值从毫米转换为米
face_recognizer=FaceRecognizer()
segmenter = PeopleSegmentation()
navigator = Navigator()
api_detect = GetPersonalInformation()
face_detect = FaceDetector()
face_insightface_recognizer=FaceRecognition()
draw_times=0
faces = None
face_list=None

current_passenger= {
    '姓名': '',
    '年龄': 0,
    '性别': '',
    '喜欢的饮料': '',
    'feature_vector': ""
}
last_passenger ={
    '姓名': '',
    '年龄': 0,
    '性别': '',
    '喜欢的饮料': '',
    'feature_vector': ""
}





def callback_color(imgmsg):
    bridge = CvBridge()
    global img_color
   
    img_color = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
    
    global is_find
    # if (seg_switch):
    global binary_mask,contours
    [is_find,binary_mask,contours]=segmenter.segment_people(img_color)
# if (face_switch):
    global faces
    faces = face_insightface_recognizer.recognition(img_color)
    
    draw()
def callback_depth(img_depth):
    bridge = CvBridge()
    global depth_image
    global distance_image 
    global scaling_factor
    depth_image = bridge.imgmsg_to_cv2(img_depth, desired_encoding='passthrough')
    
    distance_image = depth_image * scaling_factor
def search_oldest_customer():
    global face_list
    global img_color
    max_age = float('-inf')
    oldest_face = None#找年龄最大成员

    for face in face_list:
        age = face['age']
        if age > max_age:
            max_age = age
            oldest_face = face
    
    current_passenger['年龄']=age
    current_passenger['性别']=oldest_face['gender']['type']
    current_passenger['姓名']="unknown"
    # play_wav("欢迎光临！请问您叫什么名字？.wav")
    current_passenger['喜欢的饮料']="可乐"
    left, top = int(oldest_face['location']['left']), int(oldest_face['location']['top'])
    width, height = int(oldest_face['location']['width']), int(oldest_face['location']['height'])
    
    # face_features = face_recognizer.extract_face_features(img_color)
    # print("人脸特征向量：", face_features)
    # boundary_points = [(int(left), int(top)), (int(right), int(bottom))]
    
    # rect = dlib.rectangle(left, top, right, bottom)
    # gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # shape = face_recognizer.predictor(gray, rect)
    # face_descriptor = face_recognizer.model.compute_face_descriptor(img_color, shape)
    # face_vector = np.array(face_descriptor)
    # current_passenger['feature_vector']=face_vector
    print(current_passenger['feature_vector'])
  
  

    
def draw():
    
    global img_color
    global contours
    global text_location
    global closest_name
    global faces
    # print("contours2:",contours)
    if img_color is not None:
        if faces:
            for face in faces:
                # 开始人脸识别
                embedding = np.array(face.embedding).reshape((1, -1))
                embedding = preprocessing.normalize(embedding)
                user_name = "unknown"
                for com_face in face_insightface_recognizer.faces_embedding:
                    r = face_insightface_recognizer.feature_compare(embedding, com_face["feature"], face_insightface_recognizer.threshold)
                    if r:
                        user_name = com_face["user_name"]
                bbox = np.array(face.bbox).astype(np.int32)
                cv2.rectangle(img_color, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(img_color, f"{user_name}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
        if contours is not None:
            #同时时刻进行人脸识别不过每个人都打上不知道的标签
            cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)
            if closest_name is not None:
                # 计算横坐标（x）的均值和纵坐标（y）的最小值
                contour_array = contours[0]
                # 创建空列表来存储横坐标和纵坐标
                x_coordinates = []
                y_coordinates = []
                # 遍历contour_array数组，提取横坐标和纵坐标，并分别添加到列表中
                for contour in contour_array:
                    x_coordinates.append(contour[0][0])
                    y_coordinates.append(contour[0][1])
                # 计算横坐标的均值和纵坐标的最小值
                x_mean = np.mean(x_coordinates)
                y_min = np.min(y_coordinates)
                # 设置text_location为均值x_mean和最小值y_min的坐标
                text_location = (int(x_mean), int(y_min))
                cv2.putText(img_color, closest_name, text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            global face_list
            if face_list is not None:
                for face in face_list:
                    left = int(face["location"]["left"])
                    top = int(face["location"]["top"])
                    width = int(face["location"]["width"])
                    height = int(face["location"]["height"])

                    # 绘制检测框
                    cv2.rectangle(img_color, (left, top), (left + width, top + height), (0, 255, 0), 2)

                    # 构造标签字符串
                    label = f"Age: {face['age']}, Gender: {face['gender']['type']}"

                    # 绘制标签文本
                    cv2.putText(img_color, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Detected Contours', img_color)
        cv2.waitKey(1)
def play_wav(file_path):
    audio = AudioSegment.from_wav(file_path)

    # 播放音频
    play(audio)
class Welcome(smach.State):
    def __init__(self):#状态机第一个状态
        smach.State.__init__(self, outcomes=['transition_to_register'])

    def execute(self, userdata):
        global next
        next = False
        global is_find
        global entrance
        rospy.loginfo('Go to Entrance')
        if(entrance):#首先前往门口迎接
            navigator.navigate_to_point(entrance_position)
            while(not navigator.odom_callback):#当成功到达门口后
                
                rospy.sleep(0.2)
        rospy.loginfo('Welcome')#将欢迎状态打印
        # print("img_color:",img_color)
        
        while(img_color is None or is_find is False):
            rospy.sleep(0.1)#直到获取到的图片非空并且人体有大半区域出现在屏幕中
        
        if(img_color is not None and next is False and is_find is not False):
            
            
            print("is_find:",is_find)
            if is_find:
                global face_switch #打开人脸检测开关
                face_switch = True
                
                global face_list#对图片中进行人脸检测
                rospy.sleep(1)#对照片中的人脸进行检测
                detect=api_detect.get_detect(img_color)#调用api获取年龄
                face_list = detect['result']['face_list']
                # print(face_list)
                while(face_list is  None):
                    #其实这里应该有一个未检测到人脸的处理
                    continue
                if(face_list is not None):
                    search_oldest_customer()#找最老的顾客
               
               
                
                next = True
        if next is True:
            return 'transition_to_register'
        return 'transition_to_register'      
                
           
        
class register(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['transition_to_Welcome'])
        
    def execute(self, userdata):
        global next
        global closest_name
        global is_find
        next = False
        rospy.loginfo('register')
        while(img_color is None or is_find is False):
            continue
        if(img_color is not None and next is False):
            # while(closest_name is None):
            #     global text_location
            #     [closest_name,text_location]=face_recognizer.feature_get(img_color)
            #     print(closest_name)
            

# 此时 oldest_face 变量中存储了年龄最大的人脸信息

          
            
            #替换其标签为其名字
            #给他编码
            #这时候删除其他所有人的检测标签,图像中仅仅对其一个人进行检测并显示其姓名喜好
            #然后借助于人脸检测
            next = True
        while True:
            rospy.sleep(0.01)
            
        
        if next is True:
            return 'transition_to_Welcome'


    # print(img.shape)

    
    
# 订阅话题的回调函数


   

    


         
def main():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/camera/rgb/image_color", Image, callback_color)
    rospy.sleep(0.5)
    sm = smach.StateMachine(outcomes=['ros_shutdown'])
    with sm:
        smach.StateMachine.add('Welcome', Welcome(), transitions={'transition_to_register': 'register'})
        smach.StateMachine.add('register', register(), transitions={'transition_to_Welcome': 'Welcome'})

    sis = smach_ros.IntrospectionServer('smach_server', sm, '/SM_ROOT')
    sis.start()

    outcome = sm.execute()

    sis.stop() 
    if outcome == 'ros_shutdown':
        rospy.signal_shutdown('ROS state machine example complete')
    rospy.spin()
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    main()

