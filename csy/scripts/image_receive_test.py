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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
rospy.init_node('listener', anonymous=True)
is_find = False
img_color = None
depth_image = None
distance_image = None
scaling_factor = 0.001  # 缩放因子，将深度值从毫米转换为米
segmenter = PeopleSegmentation()
face_recognizer=FaceRecognizer()
def play_wav(file_path):
    # 打开.wav文件
    wav_file = wave.open(file_path, 'rb')

    # 创建一个PyAudio对象
    audio = pyaudio.PyAudio()

    # 打开音频流
    stream = audio.open(format=audio.get_format_from_width(wav_file.getsampwidth()),
                        channels=wav_file.getnchannels(),
                        rate=wav_file.getframerate(),
                        output=True)

    # 读取并播放音频数据
    chunk = 1024
    data = wav_file.readframes(chunk)
    while data:
        stream.write(data)
        data = wav_file.readframes(chunk)

    # 关闭流和音频对象
    stream.stop_stream()
    stream.close()
    audio.terminate()
class Welcome(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['transition_to_Welcome2'])

    def execute(self, userdata):
        
        rospy.loginfo('Welcome')
        #记得取消注释
        # print(img_color)
        # if (img_color is not None):
            
        #     global is_find
        #     if is_find:
        #         play_wav("欢迎光临！请问您叫什么名字？.wav")
                
                
                
            # else:
            #     rospy.Subscriber("/camera/rgb/image_color", Image, callback_color)
           
        return 'transition_to_Welcome2'
class Welcome2(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['transition_to_Welcome'])

    def execute(self, userdata):
        rospy.loginfo('Welcome2')
        
        if img_color is not None:
            face_recognizer.feature_get(img_color)
        #     return 'transition_to_Welcome'
        return 'transition_to_Welcome'

def callback_color(imgmsg):
    bridge = CvBridge()
    global img_color
    # print("接收图片")
    img_color = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
    if img_color is not None:
        global is_find
        # is_find=segmenter.segment_people(img_color)
    # print(img.shape)
def image_subscriber():
    
    rospy.Subscriber("/camera/rgb/image_color", Image, callback_color)
    rospy.spin()
# 订阅话题的回调函数


   
def callback_depth(img_depth):
    bridge = CvBridge()
    global depth_image
    global distance_image 
    global scaling_factor
    depth_image = bridge.imgmsg_to_cv2(img_depth, desired_encoding='passthrough')
    
    distance_image = depth_image * scaling_factor
    

def execute_state_machine():
    sm = smach.StateMachine(outcomes=['ros_shutdown'])
    with sm:
        smach.StateMachine.add('Welcome', Welcome(), transitions={'transition_to_Welcome2': 'Welcome2'})
        smach.StateMachine.add('Welcome2', Welcome2(), transitions={'transition_to_Welcome': 'Welcome'})

    sis = smach_ros.IntrospectionServer('smach_server', sm, '/SM_ROOT')
    sis.start()

    outcome = sm.execute()

    sis.stop()      
def main():
    
    
    
    
    # spin() simply keeps python from exiting until this node is stopped
    
    # t1 = threading.Thread(target=image_subscriber)
    # t2 = threading.Thread(target=execute_state_machine)
    # t1.start()
    # t2.start()
    
    # t1.join()
    # t2.join()
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.submit(image_subscriber)
        executor.submit(execute_state_machine)
    rospy.signal_shutdown('ROS state machine example complete')
    
    
    

if __name__ == '__main__':
    main()

