#register打错了达成了resister不管了
import requests
import cv2
import json
import base64
import urllib.parse
import rospy
import sys
sys.path.append('/home/jdt/ros_ws/src/body_segment_module/scripts/') 
sys.path.append('/home/jdt/ros_ws/src/face_recognize/scripts/')
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from face_recognize.srv import FaceRecognitionService,FaceRecognitionServiceRequest,FaceRecognitionServiceResponse
from face_recognize.srv import FaceregisterService,FaceregisterServiceRequest,FaceregisterServiceResponse
from body_segment_module.srv import body_segmentService,body_segmentServiceRequest,body_segmentServiceResponse
from api_register.srv import Api_Resister_Service,Api_Resister_ServiceRequest,Api_Resister_ServiceResponse
from body_segment_test import body_segment
from face_register_model_test import face_register
from face_recognize_model_test import face_recoginze
import numpy as np
from api_register.srv import Api_Resister_Service, Api_Resister_ServiceResponse,Api_Resister_ServiceRequest
# 引入自定义的消息类型和服务类型

class GetPersonalInformation:
    def __init__(self):
        rospy.init_node("api_register_node")
        rospy.Subscriber('/camera/rgb/image_color', Image, self.get_color_image)
        rospy.Service('api_register',Api_Resister_Service, self.api_register)
        self.image = None
        self.api_register_responce = Api_Resister_ServiceResponse()
    def get_token(self):
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
    def get_color_image(self,imgmsg): 
        bridge = CvBridge()
        img_color = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
        self.image = img_color
    def search_oldest_customer(self,face_lists,image):   
        max_age = float('-inf')
        oldest_face = None#找年龄最大成员
        for face in face_lists:
            age = face['age']
            if age > max_age:
                max_age = age
                oldest_face = face
        left = max(int(oldest_face["location"]["left"])-10,0)
        top = max(int(oldest_face["location"]["top"])-10,0)
        width = int((oldest_face["location"]["width"])*1.1)
        height = int((oldest_face["location"]["height"])*1.1)
        label = f"Age: {oldest_face['age']}, Gender: {face['gender']['type']}"
        # 绘制检测框
        #在切割时保证图片大小不变避免输入网络时出错或者检测不了,经测试不是640*480输入到网络里检测不出东西
        face_image = image[top:top+height, left:left+width]
        padding_top = max(0, (image.shape[0] - face_image.shape[0]) // 2)
        padding_bottom = max(0, image.shape[0] - face_image.shape[0] - padding_top)
        padding_left = max(0, (image.shape[1] - face_image.shape[1]) // 2)
        padding_right = max(0, image.shape[1] - face_image.shape[1] - padding_left)
        face_image_padded = np.pad(face_image, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), mode='constant')
        _, face_bytes = cv2.imencode(".jpg", face_image_padded)
        cv2.rectangle(image, (left, top), (min(left + width,640), min(top + height,480)), (255, 255, 0), 20)
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 5)
        # cv2.imshow("result",image)
        # cv2.waitKey(0)
        face_bytes = face_bytes.tobytes()
        for face in face_lists:
            left = max(int(face["location"]["left"])-30,0)
            top = max(int(face["location"]["top"])-30,0)
            width = int((face["location"]["width"])*1.2)
            height = int((face["location"]["height"])*1.2)
            # 绘制检测框
            cv2.rectangle(image, (left, top), (min(left + width,640), min(top + height,480)), (0, 255, 0), 10)
            # 构造标签字符串
            label = f"Age: {face['age']}, Gender: {face['gender']['type']}"
            # 绘制标签文本
            cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2),
        return face_bytes,oldest_face
    
    
    
    def get_detect(self, img):
        url = "https://aip.baidubce.com/rest/2.0/face/v3/detect?access_token=" + self.get_token()
        cv2.imwrite('1.jpg', img)
        image_name = "1.jpg"
        image = self.get_file_content_as_base64(image_name, False)
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
        # print(response.text)
        detect = response.json()
        return detect
    
    
    
    def api_register(self,request):
        face_register_response = None
        while(True):
            #调用其他服务
            face_reconginze_response = face_recoginze()
            body_segment_response = body_segment()
            if(face_reconginze_response.result=="success" and body_segment_response.result=="success"):
                if body_segment_response.ratio>=0.4:   
                    image = self.image
                    result=self.get_detect(image)
                    print(result)
                    if(len(result)!=0):
                        face_lists=result['result']['face_list']
                        bytes,oldest=self.search_oldest_customer(face_lists,image) 
                        username = request.username
                        gender = str(oldest['gender']['type'])
                        # print(oldest)
                        hobby = request.hobby
                        # print(str(oldest['age']))
                        # print(type(str(oldest['age'])))
                        face_register_response=(face_register(bytes,username,gender,hobby,str(oldest['age'])))
                        if face_register_response.result == "success":
                            print("注册成功啦")
                            break
                        else:
                            print("重新注册")
                    else:
                        continue
            else:
                print("请调整位置")
        self.api_register_responce.result=face_register_response.result
        self.api_register_responce.faces =  face_register_response.faces
        self.api_register_responce.age=str(oldest['age'])
        self.api_register_responce.gender=str(oldest['gender'])       
        return self.api_register_responce
               
               
                    
    def get_file_content_as_base64(self, path, urlencoded=False):
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


if __name__ == "__main__":
    try:
        api_register_node = GetPersonalInformation()
        rospy.loginfo('Api register service is running.')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    