#!/usr/bin/python3
import sys
import statistics
sys.path.append('/home/jdt/ros_ws/src/api_register/scripts/') 
sys.path.append('/home/jdt/ros_ws/src/csy/scripts') 
sys.path.append('/home/jdt/ros_ws/src/body_segment_module/scripts/') 
sys.path.append('/home/jdt/ros_ws/src/face_recognize/scripts/')#把这些包的环境变量加到系统变量里避免找不到
import math
import os
import hashlib
import base64
import hmac
import json
import tf.transformations as tftr
from urllib.parse import urlencode
import re
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
import cv2
import subprocess
import rospy
import paddlehub as hub
import PIL.Image as Image_pil
from PIL import ImageSequence
from IPython.display import display, HTML
import numpy as np
import smach
import smach_ros
from concurrent.futures import ThreadPoolExecutor
from face_recognize_module import FaceRecognizer
from segment_module import PeopleSegmentation
from pydub import AudioSegment
from pydub.playback import play
from recognize_model import FaceRecognition
from api_module import GetPersonalInformation
from navigate import Navigator
from geometry_msgs.msg import PoseStamped
from face_detected import FaceDetector
import rospy
from std_msgs.msg import String
import codecs
import requests
import json
import SparkApi
from api_register.srv import Api_Resister_Service,Api_Resister_ServiceRequest,Api_Resister_ServiceResponse
from face_recognize.srv import FaceRecognitionService,FaceRecognitionServiceResponse, FaceRecognitionServiceRequest
from face_recognize.srv import FaceregisterService, FaceregisterServiceResponse,FaceregisterServiceRequest
from std_msgs.msg import String
from body_segment_test import body_segment
from face_register_model_test import face_register
from face_recognize_model_test import face_recoginze
from api_register_service_test import api_register
from sklearn import preprocessing

"""以下是对话大模型相关内容"""
import SparkApi
#以下密钥信息从控制台获取
appid = "b7b8a5a4"     #填写控制台中获取的 APPID 信息
api_secret = "N2MxMjY2MmY5N2QxMTJhNjE4NjI4NmQ5"   #填写控制台中获取的 APISecret 信息
api_key ="0e9a50e73b428340bb4e76ec40af2318"    #填写控制台中获取的 APIKey 信息
#用于配置大模型版本，默认“general/generalv2”
domain = "general"   # v1.5版本
#domain = "generalv2"    # v2.0版本
#云端环境的服务地址
Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
#Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址
text =[]
"""以上是对话大模型相关内容"""


"""标志:响应成功与否""" 
face_register_response = None
face_recognize_response = None
body_segment_response = None
api_register_response = None
"""标志:响应成功与否""" 

"""环境配置""" 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
"""环境配置""" 

"""全局变量:一些标志与存储信息""" 
entrance = True
destination = None
encoded_message = None  # 全局变量，用于存储语音消息
name = None
en_name=None
hobby = None
"""位置标志""" 

"""位置信息:这里还需要测试几个点位""" 
entrance_position=[0.224229335785,0.349239826202,0,0,0,0.611702262265,0.791088074958]
sofa_position=[-3.75832366943,-0.225122451782,0,0,0,0.93474325117,0.355323872532]
chair_position=[ -3.17674040794,-1.31715536118,0,0,0,0.921442345979,-0.388515125878]
wash_room_position = [-0.983902573586,-2.10156536102,0.0,0.0,0.0,0.604484478545,0.796616918724]
"""位置信息:这里还需要测试几个点位""" 



scaling_factor = 0.001  # 缩放因子，将深度值从毫米转换为米
navigator = Navigator()
#自我感觉底盘运动以及播报功能没必要再封装成函数了,这两个功能都应该是什么时候调用什么时候启动,并且不会有什么冲突问题

"""存储两位客人的信息"""
current_passenger= {
    'username': '',
    'age': 0,
    'gender': '',
    'hobby': '',
    'faces': ""
}
last_passenger ={
    '姓名': '',
    '年龄': 0,
    '性别': '',
    '喜欢的饮料': '',
    'feature_vector': ""
}
"""存储两位客人的信息"""
#主控只负责调用,有限状态及的生成,以及语音播报和导航





"""合成语音函数"""
def audio_text(text,output_path):
    # 设置API URL
    url = 'https://tts.ai-lab.top'

    # 设置请求头信息
    headers = {'Content-Type': 'application/json'}

    # 设置API Token
    token = 'f20bd7cf1fa464f15295bbe4458e6e11'

    # 设置请求参数
    data = {
        'token': token,
        'speaker': '枫原万叶',
        'text': text,
        'sdp_ratio': 0.2,
        'noise': 0.5,
        'noisew': 0.9,
        'length': 1.0
    }

    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # 解析响应数据
    result = json.loads(response.content)
    # 输出音频文件链接和响应消息
    print('音频文件链接:', result['audio'])
    print('响应消息:', result['message'])
    response1 = requests.get(result['audio'])
    output_path = output_path
    audio_data = response1.content
    with open(output_path, 'wb') as file:
        file.write(response1.content)

"""情感倾向分析函数"""
def analyze_emotion(text):
    url = "https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?access_token=24.73d27a593107f64f556dab41241029bf.2592000.1700107805.282335-41193627&charset=UTF-8"

    payload = json.dumps({
        "text": text
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    data = json.loads(response.text)
    negative_prob = data['items'][0]['negative_prob']
    positive_prob = data['items'][0]['positive_prob']
    if negative_prob > positive_prob:
        result = 0
    else:
        result = 1
    return result

"""文本关键字查找函数"""
def search_text(query, content):
    global name
    global hobby
    url = "https://aip.baidubce.com/rpc/2.0/nlp/v1/txt_monet?access_token=24.510bb932c3106394318b85e3ff8289ff.2592000.1700061590.282335-41193627"
    payload = json.dumps({
        "content_list": [
            {
                "content": content,
                "query_list": [
                    {
                        "query": query
                    }
                ]
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    result = json.loads(response.text)
    name = result["results_list"][0]["results"][0]["items"][0]["text"]
    hobby = result["results_list"][0]["results"][0]["items"][0]["text"]
    print(name)
    print(hobby)


"""播放语音函数"""
def play_wav(file_path):
    audio = AudioSegment.from_wav(file_path)
    play(audio)

"""语音听写函数"""
def voice_words_callback(msg):
    global received_data
    global encoded_message
    encoded_message = msg.data
    print("Listening start!")
    rospy.loginfo(f"Received message: {encoded_message}")
    if encoded_message:
        rospy.loginfo("Received a non-empty message, stopping subscriber and iat_publish...")
def voice_words_subscriber():
    rospy.Subscriber("/voiceWords", String, voice_words_callback)
    return encoded_message
class Detect_and_Recognize(smach.State):
    def __init__(self,name,next):#导航状态#这里的next是输入参数用以表征execute方法返回什么内容时,进入到下一个状态
        smach.State.__init__(self, outcomes=[next])
        self.position=name#name为待寻找人物的名称
    def execute(self, userdata):#usrdata是整个状态机共享的数据
        navigator.navigate_to_point(self.position)
    #死逻辑,即不需要大模型,由代码显示调用状态机完成一些列固定任务


"""以下是gpt回答处理相关的函数"""
def getText(role,content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text

def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length

def checklen(text):
    while (getlength(text) > 8000):
        del text[0]
    return text

def gpt(question):
    text.clear
    question = checklen(getText("user", question))
    SparkApi.answer =""
    #print("星火:",end = "")
    SparkApi.main(appid,api_key,api_secret,Spark_url,domain,question)
    getText("assistant",SparkApi.answer)
    print(SparkApi.answer)
    return SparkApi.answer

def split_string(sentence):
    pattern = r'(动作|参数)(\d+):(\S+)'
    matches = re.findall(pattern, sentence)
    action_count = 0
    
    # 字典映射动作和函数
    action_functions = {
        '导航': navigate,
        '检测识别': detect,
        '播放音乐':music,
        '聊天':chat,
    }
    
    for match in matches:
        if match[0] == '动作':
            action_count += 1
            action_number = match[1]
            action_content = match[2]
            print(f"动作{action_number}: {action_content}")
            
            # 判断动作是否在映射字典中
            if action_content in action_functions:
                parameter_matches = re.findall(r"参数"+action_number+":(\S+)", sentence)
                if parameter_matches:
                    parameter = parameter_matches[0]
                else:
                    parameter = ""
                action_functions[action_content](parameter)
        # elif match[0] == '参数':
        #     parameter_number = match[1]
        #     parameter_content = match[2]
        #     print(f"参数{parameter_number}: {parameter_content}")
    return action_count
"""以上是gpt回答处理相关的函数"""

def angle_2_tftr(angle):
    # 假设desk_position为包含平面角度的列表
    roll, pitch, yaw = 0,0,angle

    # 将角度转换为弧度
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    
    # 使用tf库将欧拉角转换为四元数
    quaternion = tftr.quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)
    return quaternion

def tftr_2_angle(quaternion):
    roll, pitch, yaw = tftr.euler_from_quaternion(quaternion)

    # 将欧拉角从弧度转换为度
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    return roll_deg,pitch_deg,yaw_deg
"""以下是不同任务类别调用的函数"""
def navigate(parameter):
    print(f"导航函数调用，参数: {parameter}")
    play_wav("gpt导航启动.wav")
    if("门口" in parameter or "起点" in parameter or "初始" in parameter):
        navigator.navigate_to_point(entrance_position)
        while(not(navigator.is_goal_reached())):#当成功到达后
            rospy.sleep(2)
            print(navigator.is_goal_reached())
    if("厕所" in parameter):
        navigator.navigate_to_point(wash_room_position)
        while(not(navigator.is_goal_reached())):#当成功到达后
            rospy.sleep(2)
            print(navigator.is_goal_reached())
    if("沙发" in parameter):
        navigator.navigate_to_point(sofa_position)
        while(not(navigator.is_goal_reached())):#当成功到达后
            rospy.sleep(2)
            print(navigator.is_goal_reached())
    if("凳子" in parameter or "椅子" in parameter):
        navigator.navigate_to_point(chair_position)
        while(not(navigator.is_goal_reached())):#当成功到达后
            rospy.sleep(2)
            print(navigator.is_goal_reached())
    audio_text(parameter+",已到达!","gpt已到达.wav")
    play_wav("gpt已到达.wav")


def detect(parameter):
    print(f"检测识别函数调用，参数: {parameter}")
    while(True):
        face_reconginze_response = face_recoginze()
        shift(1)
        print("场景中存在的人物有"+face_reconginze_response.user_find)
        if parameter in face_reconginze_response.user_find:  
            audio_text("已找到"+parameter,"gpt寻找目标.wav")
            play_wav("gpt寻找目标.wav")
    
    
def music(parameter):
    print(f"播放音乐函数调用，参数: {parameter}")
    if("演员" in parameter):
        play_wav("演员.wav")
    if("体面" in parameter):
        play_wav("体面.wav")
        
def chat(parameter):
    global en_name
    print("聊天任务启动,聊天的内容是:{parameter}")
    text_chat="现在,你是一位陪客人聊天的迎宾机器人.请你和客人聊聊"+parameter
    result=gpt(text_chat)
    audio_text(result,"聊天1.wav")
    play_wav("聊天1.wav")
    voice_words_subscriber()
    # 运行rosrun命令
    iat_publish_process = subprocess.Popen(["rosrun", "robot_voice", "iat_publish"])
    print("start success")
    rospy.sleep(15)
    iat_publish_process.terminate()
    while("结束" not in voice_words_subscriber() and "不聊" not in voice_words_subscriber()):
        result1=gpt(voice_words_subscriber())
        audio_text(result1+"请问还要继续聊天吗?","聊天2.wav")
        play_wav("聊天2.wav")
        voice_words_subscriber()
        # 运行rosrun命令
        iat_publish_process = subprocess.Popen(["rosrun", "robot_voice", "iat_publish"])
        print("start success")
        rospy.sleep(15)
        iat_publish_process.terminate()
    audio_text(en_name+"老师,很开心能够与您聊天,我们下次见!","结束聊天.wav")
    play_wav("结束聊天.wav")
"""以上是不同任务类别调用的函数"""


def shift(sort):
    l=320/math.tan(math.radians(30))
    if sort==0:
        while(True):
            #获取当前位置
            orientation,position=navigator.get_current_position()
            if(orientation!=None):#当前朝向信息获取后
                
                #当前角度计算(角度制)
                _,_,current_angle=tftr_2_angle([orientation.x,orientation.y,orientation.z,orientation.w])
                
                #获取与目标相对位置
                body_segment_response = body_segment()
                target_angle = current_angle
                # print(body_segment_response.cX)
                # print(body_segment_response.cY)
                mean_distance=body_segment_response.mean_distance
                #计算出当前到人体距离
                if(body_segment_response.cX>340):#角度计算,如果当前人体相对于摄像头偏右,向右转,角度变小一点
                    target_angle=current_angle-math.degrees(math.atan((body_segment_response.cX-320)/l))
                    audio_text("朝右转","shift.wav")
                elif(body_segment_response.cX<280):
                    audio_text("朝左转","shift.wav")
                    target_angle=current_angle+math.degrees(math.atan((320-body_segment_response.cX)/l))
                else:
                    break
                target_position_x = position.x
                target_position_y = position.y
                target_position_z = 0 
                target_orientation=angle_2_tftr(target_angle)
                target=[target_position_x,target_position_y,target_position_z,target_orientation[0],target_orientation[1],target_orientation[2],target_orientation[3]]
                navigator.navigate_to_point(target)
                rospy.sleep(0.1)
    else:
        
        orientation,position=navigator.get_current_position()
        if(orientation!=None):#当前朝向信息获取后
            
            #当前角度计算(角度制)
            _,_,current_angle=tftr_2_angle([orientation.x,orientation.y,orientation.z,orientation.w])
            target_angle=current_angle+10
            target_position_x = position.x
            target_position_y = position.y
            target_position_z = 0 
            target_orientation=angle_2_tftr(target_angle)
            target=[target_position_x,target_position_y,target_position_z,target_orientation[0],target_orientation[1],target_orientation[2],target_orientation[3]]
            navigator.navigate_to_point(target)
            navigator.navigate_to_point(sofa_position)
            while(not(navigator.is_goal_reached())):#当成功到达后
                rospy.sleep(0.1)
                print(navigator.is_goal_reached())


def follow():
    dzhiding=10
    l=320/math.tan(math.radians(30))
    while(True):
        #获取当前位置
        orientation,position=navigator.get_current_position()
        if(orientation!=None):#当前朝向信息获取后
            
            #当前角度计算(角度制)
            _,_,current_angle=tftr_2_angle([orientation.x,orientation.y,orientation.z,orientation.w])
            
            #获取与目标相对位置
            body_segment_response = body_segment()
            print(body_segment_response)
            target_angle = current_angle
            # print(body_segment_response.cX)
            # print(body_segment_response.cY)
            icishu=0
            mean_distanceresult=[]
            while(1):
             if(icishu<10):
                mean_distance=body_segment_response.mean_distance
                mean_distanceresult.append(mean_distance)
                icishu=icishu+1
             else:
                 break
            mean_distance=statistics.median(mean_distanceresult)
            
            print(mean_distance)
            #计算出当前到人体距离
            if(body_segment_response.cX>340):#角度计算,如果当前人体相对于摄像头偏右,向右转,角度变小一点
                target_angle=current_angle-math.degrees(math.atan((body_segment_response.cX-320)/l))
                audio_text("朝右转","shift.wav")
            elif(body_segment_response.cX<280):
                audio_text("朝左转","shift.wav")
                target_angle=current_angle+math.degrees(math.atan((320-body_segment_response.cX)/l))
            else:
                break
            target_position_x = position.x+dzhiding*math.sin(target_angle)
            target_position_y = position.y+dzhiding*math.cos(target_angle)
            target_position_z = 0 
            target_orientation=angle_2_tftr(target_angle)
            target=[position.x,position.y,position.z,target_orientation[0],target_orientation[1],target_orientation[2],target_orientation[3]]
            rospy.sleep(1)
            navigator.navigate_to_point(target)
            while(not(navigator.is_goal_reached())):#当成功到达门口后
                rospy.sleep(0.5)
            rospy.sleep(0.1)
            idaoda=0
            target=[target_position_x,target_position_y,target_position_z,target_orientation[0],target_orientation[1],target_orientation[2],target_orientation[3]]
            navigator.navigate_to_point(target)
            while(not(navigator.is_goal_reached())):#当成功到达门口后
                rospy.sleep(0.5)
                idaoda=idaoda+1
                if(i>10):
                    break
            # rospy.sleep(0.1)
    


"""状态1:欢迎访客的到来:前往门口迎接访客-询问姓名-人脸识别-成功迎接"""
class Welcome(smach.State):
    def __init__(self):#状态机第一个状态,负责回到初始位置以及检测人体
        smach.State.__init__(self, outcomes=['transition_to_register'])
    def execute(self, position):
        global en_name
        global entrance
        rospy.loginfo('Go to Entrance')    
        # if(entrance):#首先前往门口迎接
        #     navigator.navigate_to_point(entrance_position)
        #     rospy.loginfo(navigator.is_goal_reached())#函数不能当成属性用
        #     while(not(navigator.is_goal_reached())):#当成功到达门口后
        #         rospy.sleep(2)
        #         print(navigator.is_goal_reached())
        rospy.loginfo('Welcome')#将欢迎状态打印
        play_wav("欢迎光临！请问您叫什么名字？.wav")
        rospy.sleep(1)  # 增加一些延迟以确保播放音频
        voice_words_subscriber()
        # 运行rosrun命令
        iat_publish_process = subprocess.Popen(["rosrun", "robot_voice", "iat_publish"])
        print("start success")
        rospy.sleep(15)
        iat_publish_process.terminate()
        search_text("姓名",voice_words_subscriber())
        en_name=name
        text1="请问您的姓名是"+name+"吗?"
        audio_text(text1,"确认姓名.wav")
        play_wav("确认姓名.wav")
        rospy.sleep(1)#休眠时间根据需要修改
        voice_words_subscriber()
        # 运行rosrun命令
        iat_publish_process = subprocess.Popen(["rosrun", "robot_voice", "iat_publish"])
        print("start success")
        rospy.sleep(15)
        iat_publish_process.terminate()
        if analyze_emotion(voice_words_subscriber()):
            audio_text("好的,欢迎你"+name,"重复姓名1.wav")
            play_wav("重复姓名1.wav")
        else:
            play_wav("抱歉,请您再说一遍您的名字.wav")
            voice_words_subscriber()
            # 运行rosrun命令
            iat_publish_process = subprocess.Popen(["rosrun", "robot_voice", "iat_publish"])
            print("start success")
            rospy.sleep(15)
            iat_publish_process.terminate()
            search_text("姓名",voice_words_subscriber())
            text2="好的,您的姓名是"+name+".欢迎你!"
            audio_text(text2,"重复姓名2.wav")
            play_wav("重复姓名2.wav")
            rospy.sleep(1)#休眠时间根据需要修改
        while (True):#调用识别与分割服务,直到识别与分割服务都返回true,此时证明人来了
            face_reconginze_response = face_recoginze()
            body_segment_response = body_segment()
            if(face_reconginze_response.result=="success" and body_segment_response.result=="success"):
                rospy.loginfo("成功迎接客人")
                rospy.sleep(5)#休眠时间根据需要修改
                return 'transition_to_register'
            else:
                rospy.loginfo("等待客人到来")
                rospy.sleep(1)
"""状态1:欢迎访客的到来:前往门口迎接访客-询问姓名-人脸识别-成功迎接"""

    
"""状态2:注册 询问访客喜欢喝的饮料-注册访客的人脸-年龄性别估计-判断引导的对应的位置"""      
class register(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['transition_to_Navigate_sofa_desk'])
        
    def execute(self, userdata):
        global hobby
        rospy.loginfo('register')
        #这里的hobby和username应该通过语音听写获得
        #并且需要增加确认功能避免一次听写出错
        hobby  = "可乐"
        username = name
        global api_register_response#调用注册功能
        while(True):
            api_register_response=api_register(hobby,username)
            current_passenger['username']=username
            current_passenger['age']=int(api_register_response.age)
            current_passenger['hobby']=hobby
            current_passenger['gender']=api_register_response.gender
            print(type(current_passenger['gender']))
            if(api_register_response.result=="success"):
                rospy.loginfo('注册成功')
                play_wav("欢迎第一位访客.wav")
                play_wav("对姐姐的夸赞.wav")
                rospy.sleep(5)
                audio_text(name+"老师,请问您喜欢喝什么饮料?","询问饮料.wav")
                play_wav("询问饮料.wav")
                voice_words_subscriber()
                # 运行rosrun命令
                iat_publish_process = subprocess.Popen(["rosrun", "robot_voice", "iat_publish"])
                print("start success")
                rospy.sleep(15)
                iat_publish_process.terminate()
                search_text("饮料",voice_words_subscriber())
                text_answerdrink="好的,明白了!你爱喝"+hobby
                audio_text(text_answerdrink,"确认饮料.wav")
                play_wav("确认饮料.wav")
                rospy.sleep(1)#休眠时间根据需要修改
                if(current_passenger['age']>=40):
                    global destination
                    destination = "sofa"
                    play_wav("老人导引.wav")
                else:
                    destination = "desk"
                    play_wav("年轻人导引.wav")
                break
            rospy.sleep(1)
            rospy.loginfo('register failed')
        return 'transition_to_Navigate_sofa_desk'
"""状态2:注册 询问访客喜欢喝的饮料-注册访客的人脸-年龄性别估计-判断引导的对应的位置"""      

"""状态3:引导 判断访客到对应的位置"""      
class Navigate_sofa_desk(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['transition_to_Introduce'])
        
    def execute(self, userdata):
        rospy.loginfo('Navigate_sofa_desk')
        play_wav("请跟着我，我将为你指引一些可用的座位。.wav");
        rospy.sleep(5)
        global destination
        if(destination=="sofa"):
            navigator.navigate_to_point(sofa_position)
        else:
            navigator.navigate_to_point(chair_position)
        #同时也可以显示一些导航到哪个位置的一些内容
        while(not navigator.is_goal_reached()):
            rospy.sleep(0.5)
            rospy.loginfo("正在前往"+destination)
        #也可以播报已到达哪里
        rospy.loginfo('顺利到达'+destination)
        rospy.sleep(5)
        shift(0)
        return 'transition_to_Introduce'
"""状态3:引导 判断访客到对应的位置"""   

"""状态4:介绍 寻找人物-人物在屏幕中心位置-介绍人物"""   
class Introduce(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['transition_to_Back'])
        
    def execute(self, userdata):
        rospy.loginfo('即将介绍人物,先寻找')
        rospy.sleep(5)
        while(True):
            global en_name
            global face_recognize_response
            face_recognize_response=face_recoginze()
            if(face_recognize_response.position=="聚焦人物在视野中央"):#先去识别当目标人物在视野中央
                #这里应该是播放语音
                rospy.loginfo("人物已找到,现在开始介绍")
                rospy.sleep(5)
                print("此人为 年龄 性别 喜欢喝的饮料")
                text_introduce="主人,您好!我把人给你带来了.这位客人是"+en_name+".他爱喝的饮料是"+hobby+"!祝你们度过美好的一天!"
                audio_text(text_introduce,"介绍客人.wav")
                play_wav("介绍客人.wav")
                break
            else:
                #获取当前位置和朝向后转动底盘
                #我这里直接用休眠代替了
                print(face_recognize_response.position)
                rospy.sleep(1)
        rospy.loginfo('介绍完毕,即将返回')
        play_wav("非常高兴认识你.wav")
        rospy.sleep(3) 
        play_wav("其他帮助.wav")
        text_org="我需要你帮助我对以下指令内容进行提取，并提取出其中的导航任务,人脸检测与识别任务,播放音乐任务,聊天任务以及相应任务的参数,然后将其按照我向你输入的答案输出格式为我输出。以下是一些案例:注意一定要识别其中的动作在逻辑上的先后次序以及相应动作的参数.请带我前往沙发    答案:动作1:导航 参数1:沙发    陪我聊聊天气    答案:动作1:聊天 参数1:天气    请带我去厨房寻找金东涛  答案:动作1:导航 参数1:沙发 动作2:检测识别 参数2:金东涛  请去学校找陶沐村,但在此之前现在原地寻找金东涛 答案:动作1:导航 参数1:原地 动作2:检测识别 参数2:金东涛 动作3:导航 参数3:学校 动作4:检测识别 参数4:陶沐村   请去沙发放薛之谦的丑八怪 答案:动作1:导航 参数1:沙发 动作2:放音乐 参数2:丑八怪   去椅子上找瓶水  答案:动作1:导航 参数1:椅子 动作2:检测识别 参数2:饮料。现在，我将输入我的指令。"
        voice_words_subscriber()
        # 运行rosrun命令
        iat_publish_process = subprocess.Popen(["rosrun", "robot_voice", "iat_publish"])
        print("start success")
        rospy.sleep(15)
        iat_publish_process.terminate()
        text_cur=voice_words_subscriber()
        text=text_org+text_cur
        result = gpt(text)
        action_count = split_string(result)
        print(f"共有{action_count}个动作")
        return 'transition_to_Back'  
"""状态4:介绍 寻找人物-人物在屏幕中心位置-介绍人物"""


"""状态5:返回 人物返回到门口位置"""   
class Back(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['transition_to_Welcome'])
        
    def execute(self, userdata):
        rospy.loginfo('return the entrance')
        navigator.navigate_to_point(entrance_position)
        while(not navigator.is_goal_reached()):#当成功到达门口后
            rospy.sleep(5)
            rospy.loginfo("一次行程结束")
        return 'transition_to_Welcome'
"""状态5:返回 人物返回到门口位置"""   

def main():
    # rospy.init_node('main_control', anonymous=True)
    # rospy.sleep(0.5)
    # # 主要执行逻辑
    rospy.init_node('main_control', anonymous=True)
    rospy.sleep(0.5)
    # 创建一个状态机
    sm = smach.StateMachine(outcomes=['ros_shutdown'])#最后的结束状态为shutdown
    # 使用add方法添加状态到状态机容器当中
    with sm:
        smach.StateMachine.add('Welcome', Welcome(), transitions={'transition_to_register': 'register'})
        smach.StateMachine.add('register', register(), transitions={'transition_to_Navigate_sofa_desk': 'Navigate_sofa_desk'})
        smach.StateMachine.add('Navigate_sofa_desk', Navigate_sofa_desk(), transitions={'transition_to_Introduce': 'Introduce'})
        smach.StateMachine.add('Introduce', Introduce(), transitions={'transition_to_Back': 'Back'})
        smach.StateMachine.add('Back', Back(), transitions={'transition_to_Welcome': 'Welcome'})
    # 创建并启动内部监测服务器
    sis = smach_ros.IntrospectionServer('smach_server', sm, '/SM_ROOT')
    sis.start()
    # 开始执行状态机
    outcome = sm.execute()
    sis.stop() 
    if outcome == 'ros_shutdown':
        rospy.signal_shutdown('ROS state machine example complete')
    rospy.spin()
follow()