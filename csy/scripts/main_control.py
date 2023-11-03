#!/usr/bin/python3
import os
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
import rospy
from std_msgs.msg import String
import codecs
import sys
import requests
import json
import tf.transformations as tftr
import math
sys.path.append('/home/jdt/ros_ws/src/api_register/scripts/') 
sys.path.append('/home/jdt/ros_ws/src/csy/scripts') 
sys.path.append('/home/jdt/ros_ws/src/body_segment_module/scripts/') 
sys.path.append('/home/jdt/ros_ws/src/face_recognize/scripts/')#把这些包的环境变量加到系统变量里避免找不到
from api_register.srv import Api_Resister_Service,Api_Resister_ServiceRequest,Api_Resister_ServiceResponse
from face_recognize.srv import FaceRecognitionService,FaceRecognitionServiceResponse, FaceRecognitionServiceRequest
from face_recognize.srv import FaceregisterService, FaceregisterServiceResponse,FaceregisterServiceRequest
from std_msgs.msg import String
from body_segment_test import body_segment
from face_register_model_test import face_register
from face_recognize_model_test import face_recoginze
from api_register_service_test import api_register
from sklearn import preprocessing
face_register_response = None
face_recognize_response = None
body_segment_response = None
api_register_response = None
navigator = Navigator()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
entrance = True
destination = None
#这是机器人初始位置
entrance_position=[0.224229335785,0.349239826202,0,0,0,0.611702262265,0.791088074958]
#以下位置都还没测试,都使用的是机器人的初始位置
sofa_position=[0.733,0.155,0,0,0,1,0.006]
chair_position=[ -3.17674040794,-1.31715536118,0,0,0,0.921442345979,-0.388515125878]

#这里还需要测试几个点位
scaling_factor = 0.001  # 缩放因子，将深度值从毫米转换为米

#自我感觉底盘运动以及播报功能没必要再封装成函数了,这两个功能都应该是什么时候调用什么时候启动,并且不会有什么冲突问题
current_angle = 0
#存储两位客人的信息
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
#主控只负责调用,有限状态及的生成,以及语音播报和导航


encoded_message = None  # 全局变量，用于存储语音消息
name = None
hobby = None
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
    print(result)
    name = result["results_list"][0]["results"][0]["items"][0]["text"]
    hobby = result["results_list"][0]["results"][0]["items"][0]["text"]
    print(name)
    print(hobby)

def play_wav(file_path):
    audio = AudioSegment.from_wav(file_path)
    play(audio)


def voice_words_callback(msg):
    global received_data
    global encoded_message
    encoded_message = msg.data
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


class Welcome(smach.State):
    def __init__(self):#状态机第一个状态,负责回到初始位置以及检测人体
        smach.State.__init__(self, outcomes=['transition_to_register'])
    def execute(self, position):
        global entrance
        rospy.loginfo('Go to Entrance')
        if(entrance):#首先前往门口迎接
            navigator.navigate_to_point(entrance_position)
            while(not navigator.odom_callback):#当成功到达门口后
                rospy.sleep(0.5)
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
                    audio_text("小心台阶,现在,我将带您去沙发稍作休息.","老人导引.wav")
                    play_wav("老人导引.wav")
                else:
                    destination = "desk"
                    audio_text("请跟我来,我将带您去桌子旁等候.","年轻人导引.wav")
                    play_wav("年轻人导引.wav")
                break
            rospy.sleep(1)
            rospy.loginfo('register failed')
        return 'transition_to_Navigate_sofa_desk'


class Navigate_sofa_desk(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['transition_to_Introduce'])
        
    def execute(self, userdata):
        rospy.loginfo('Navigate_sofa_desk')
        play_wav("请跟着我，我将为你指引一些可用的座位。.wav")
        rospy.sleep(5)
        global destination
        if(destination=="sofa"):
            navigator.navigate_to_point(sofa_position)
        else:
            navigator.navigate_to_point(desk_position)
        #同时也可以显示一些导航到哪个位置的一些内容
        while(not navigator.odom_callback):
            rospy.sleep(0.5)
            rospy.loginfo("正在前往"+destination)
        #也可以播报已到达哪里
        rospy.loginfo('顺利到达'+destination)
        rospy.sleep(5)
        return 'transition_to_Introduce'

class Introduce(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['transition_to_Back'])
        
    def execute(self, userdata):
        rospy.loginfo('即将介绍人物,先寻找')
        rospy.sleep(5)
        while(True):
            global face_recognize_response
            face_recognize_response=face_recoginze()
            if(face_recognize_response.position=="聚焦人物在视野中央"):#先去识别当目标人物在视野中央
                #这里应该是播放语音
                rospy.loginfo("人物已找到,现在开始介绍")
                rospy.sleep(5)
                print("此人为 年龄 性别 喜欢喝的饮料")
                break
            else:
                #获取当前位置和朝向后转动底盘
                #我这里直接用休眠代替了
                print(face_recognize_response.position)
                rospy.sleep(1)
        rospy.loginfo('介绍完毕,即将返回')
        play_wav("非常高兴认识你.wav")
        rospy.sleep(3) 
        return 'transition_to_Back'  



class Back(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['ros_shutdown'])
        
    def execute(self, userdata):
        rospy.loginfo('return the entrance')
        navigator.navigate_to_point(entrance_position)
        while(not navigator.odom_callback):#当成功到达门口后
            rospy.sleep(5)
            rospy.loginfo("一次行程结束")
        return 'ros_shutdown'   


         
def main():
    rospy.init_node('main_control', anonymous=True)
    rospy.sleep(0.5)
    
    rospy.sleep(2)
    # sm = smach.StateMachine(outcomes=['ros_shutdown'])#最后的结束状态为shutdown
    # with sm:
    #     smach.StateMachine.add('Welcome', Welcome(), transitions={'transition_to_register': 'register'})
    #     smach.StateMachine.add('register', register(), transitions={'transition_to_Navigate_sofa_desk': 'Navigate_sofa_desk'})
    #     smach.StateMachine.add('Navigate_sofa_desk', Navigate_sofa_desk(), transitions={'transition_to_Introduce': 'Introduce'})
    #     smach.StateMachine.add('Introduce', Introduce(), transitions={'transition_to_Back': 'Back'})
    #     smach.StateMachine.add('Back', Back(), transitions={'ros_shutdown': 'ros_shutdown'})
    # sis = smach_ros.IntrospectionServer('smach_server', sm, '/SM_ROOT')
    # sis.start()
    # outcome = sm.execute()
    # sis.stop() 
    # if outcome == 'ros_shutdown':
    #     rospy.signal_shutdown('ROS state machine example complete')
    navigator.navigate_to_point(chair_position)#一定要有初始位置
    while(not(navigator.is_goal_reached())):#当成功到达门口后
                rospy.sleep(2)
                print(navigator.is_goal_reached())
    l=320/math.tan(math.radians(30))
    while(True):
        #获取当前位置
        orientation,position=navigator.get_current_position()
        if(orientation!=None):#当前朝向信息获取后
            
            #当前角度计算(角度制)
            _,_,current_angle=tftr_2_angle([orientation.x,orientation.y,orientation.z,orientation.w])
            
            #获取与目标相对位置
            body_segment_response = body_segment()
            target_aourngle = current_angle
            print(body_segment_response.cX)
            print(body_segment_response.cY)
            mean_distance=body_segment_response.mean_distance
            #计算出当前到人体距离
            if(body_segment_response.cX>340):#角度计算,如果当前人体相对于摄像头偏右,向右转,角度变小一点
                target_angle=current_angle-math.degrees(math.atan((body_segment_response.cX-320)/l))
            elif(body_segment_response.cX<280):
                target_angle=current_angle+math.degrees(math.atan((320-body_segment_response.cX)/l))
                #将角度纠正一下
            #纠正位置
            # if(mean_distance>=0.6):#当距离大于一定值时,修改位置,朝向目标移动
            #     target_position_x = position.x+mean_distance*math.cos(math.radians(current_angle))#目标位置
            #     target_position_y = position.y+mean_distance*math.cos(math.radians(current_angle))
            #     target_position_z = 0
            # elif(mean_distance<=0.6 and mean_distance>=0.4):
            target_position_x = position.x
            target_position_y = position.y
            target_position_z = 0
            # else:#当距离小于一定值时,修改位置,原理目标移动
            #     target_position_x = position.x-mean_distance*math.cos(math.radians(current_angle))#目标位置
            #     target_position_y = position.y-mean_distance*math.cos(math.radians(current_angle))
            #     target_position_z = 0
                #纠正朝向
            
            target_orientation=angle_2_tftr(target_angle)
            target=[target_position_x,target_position_y,target_position_z,target_orientation[0],target_orientation[1],target_orientation[2],target_orientation[3]]
            navigator.navigate_to_point(target)
            
            

            print("current_angel:",current_angle)
            # print("current_position:",position)
            print("target_angle",target_angle)
            # print("target_position_x",target_position_x)
            # print("target_position_y",target_position_y)
            # print("target_position_z",target_position_z)
            rospy.sleep(0.1)
        
    rospy.spin()
    
    
    
if __name__ == '__main__':
    main()

