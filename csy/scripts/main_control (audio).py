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
import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
import gradio as gr
import librosa
import webbrowser
import time
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence,_symbol_to_id, get_bert
from text.cleaner import clean_text
from scipy.io import wavfile


device = "cuda:0" if torch.cuda.is_available() else "cpu"
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

language_marks = {
    "简体中文": "[ZH]",
}
lang = ['简体中文']



face_register_response = None
face_recognize_response = None
body_segment_response = None
api_register_response = None
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
entrance = True
destination = None
#这是机器人初始位置
entrance_position=[0.733,0.155,0,0,0,1,0.006]
#以下位置都还没测试,都使用的是机器人的初始位置
sofa_position=[0.733,0.155,0,0,0,1,0.006]
desk_position=[0.733,0.155,0,0,0,1,0.006]
#这里还需要测试几个点位
scaling_factor = 0.001  # 缩放因子，将深度值从毫米转换为米
navigator = Navigator()
#自我感觉底盘运动以及播报功能没必要再封装成函数了,这两个功能都应该是什么时候调用什么时候启动,并且不会有什么冲突问题

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

def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    print([f"{p}{t}" for p, t in zip(phone, tone)])
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str)

    assert bert.shape[-1] == len(phone)

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)

    return bert, phone, tone, language


dev='cuda'
def infer(text, sdp_ratio, noise_scale, noise_scale_w,length_scale,sid):
    bert, phones, tones, lang_ids = get_text(text,"ZH", hps,)
    print(sid)
    with torch.no_grad():
        x_tst=phones.to(dev).unsqueeze(0)
        tones=tones.to(dev).unsqueeze(0)
        lang_ids=lang_ids.to(dev).unsqueeze(0)
        bert = bert.to(dev).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(dev)
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(dev)
        audio = net_g.infer(x_tst, x_tst_lengths, speakers, tones, lang_ids,bert, sdp_ratio=sdp_ratio
                           , noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
        #del stn_tst,tones,lang_ids,bert, x_tst, x_tst_lengths, sid
        return audio


def synthesize_audio(text, output_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./G_latest.pth", help="directory to your fine-tuned model")
    parser.add_argument("--config_dir", default="./configs\config.json", help="directory to your model config file")
    parser.add_argument("--share", default=False, help="make link public (used in colab)")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config_dir)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers, 
        **hps.model).to(dev)
    _ = net_g.eval()
    _ = utils.load_checkpoint(args.model_dir, net_g, None,skip_optimizer=True)

    speaker_ids = hps.data.spk2id
    speakers = list(hps.data.spk2id.keys())
    #inf = infer(net_g, hps, speaker_ids)

    text = "你知道吗，我最喜欢的季节是冬天。"
    sdp_ratio = 0.2
    noise_scale = 0.5
    noise_scale_w = 0.9
    length_scale = 1.0
    audio_data = infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, 'skadi')
    output_path = "output.wav"
    sampling_rate = hps.data.sampling_rate  # 采样率
    import soundfile as sf
    sf.write(output_path, audio_data, sampling_rate)



def search_text(query, content):
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
    print(name)

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
        play_wav("欢迎光临！请问您叫什么名字？.wav");
        rospy.sleep(1)  # 增加一些延迟以确保播放音频
        voice_words_subscriber()
        # 运行rosrun命令
        iat_publish_process = subprocess.Popen(["rosrun", "robot_voice", "iat_publish"])
        print("start success")
        rospy.sleep(15)
        iat_publish_process.terminate()
        search_text("姓名",voice_words_subscriber())
        rospy.sleep(10)#休眠时间根据需要修改
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
        rospy.loginfo('register')
        #这里的hobby和username应该通过语音听写获得
        #并且需要增加确认功能避免一次听写出错
        hobby  = "coke"
        username = "tmc"
        global api_register_response#调用注册功能
        while(True):
            api_register_response=api_register(hobby,username)
            current_passenger['username']=username
            current_passenger['age']=int(api_register_response.age)
            current_passenger['hobby']=hobby
            current_passenger['gender']=api_register_response.gender
            if(api_register_response.result=="success"):
                rospy.loginfo('注册成功')
                play_wav("欢迎第一位访客.wav");
                rospy.sleep(5)
                if(current_passenger['age']>=40):
                    global destination
                    destination = "sofa"
                else:
                    destination = "desk"
                break
            rospy.sleep(1)
            rospy.loginfo('register failed')
        return 'transition_to_Navigate_sofa_desk'


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
        play_wav("非常高兴认识你.wav");
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
    sm = smach.StateMachine(outcomes=['ros_shutdown'])#最后的结束状态为shutdown
    with sm:
        smach.StateMachine.add('Welcome', Welcome(), transitions={'transition_to_register': 'register'})
        smach.StateMachine.add('register', register(), transitions={'transition_to_Navigate_sofa_desk': 'Navigate_sofa_desk'})
        smach.StateMachine.add('Navigate_sofa_desk', Navigate_sofa_desk(), transitions={'transition_to_Introduce': 'Introduce'})
        smach.StateMachine.add('Introduce', Introduce(), transitions={'transition_to_Back': 'Back'})
        smach.StateMachine.add('Back', Back(), transitions={'ros_shutdown': 'ros_shutdown'})
    sis = smach_ros.IntrospectionServer('smach_server', sm, '/SM_ROOT')
    sis.start()
    outcome = sm.execute()
    sis.stop() 
    if outcome == 'ros_shutdown':
        rospy.signal_shutdown('ROS state machine example complete')
    rospy.spin()
    
    
    
if __name__ == '__main__':
    main()

