#!/usr/bin/python3

import os
import cv2
import rospy
from std_msgs.msg import String

from cv_bridge import CvBridge
import json
import zlib
import sys
from PIL import Image as Image2
from sensor_msgs.msg import Image

from PIL import ImageDraw,ImageFont
import os
import pickle

import gzip
#responce和request前面的内容应该和service文件一致
from face_recognize.srv import FaceRecognitionService,FaceRecognitionServiceRequest,FaceRecognitionServiceResponse
from face_recognize.srv import FaceregisterService,FaceregisterServiceRequest,FaceregisterServiceResponse
#从对应文件夹下的srv文件导入两种服务消息类型及其对应的请求与相应
import insightface
import numpy as np
from sklearn import preprocessing

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class FaceRecognitionServer:
    def __init__(self):
        self.gpu_id = 0
        self.face_db = 'face_db'
        self.threshold = 1.10
        self.det_thresh = 0.50
        self.det_size = (640, 640)
        self.model = insightface.app.FaceAnalysis(root='./',
                                                  allowed_modules=None,
                                                  providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)
        self.faces_embedding = list()
        self.current_user_embedding = None
        self.load_faces(self.face_db)
        self.image=None
        self.username_string = " "
        self.current_user = None
        self.current_user_username = None
        self.current_user_age = None
        self.current_user_gender = None
        self.current_user_favor = None
        
        self.position_information = None
        rospy.init_node('face_recognition_server')
        #订阅颜色图并实时处理
        rospy.Subscriber("/camera/rgb/image_color", Image, self.handle_face_recognition)
        
        #调用这个服务是,将检测结果返回
        rospy.Service('face_recognition', FaceRecognitionService, self.handle_face_recognition_service)
        #调用这个服务时,将人员注册并将结果返回
        rospy.Service('face_register', FaceregisterService, self.handle_face_register_service)
        self.face_recognition_responce = FaceRecognitionServiceResponse()
        self.face_register_responce = FaceregisterServiceResponse()
        self.face_recognition_responce.user_find = " "
        
    

        
    

    def handle_face_recognition(self,image):#回调显示
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(image, 'bgr8')
        self.image=image
        faces = self.model.get(image)
       
        find_person = False
        if(faces!=None):
            num_people = len(faces)
        else:
            num_people = 0
        is_draw = [True] * num_people 
        font = ImageFont.truetype('simhei.ttf', 30, encoding='utf-8')
        self.username_string = ""
        if num_people != 0:
            for com_face in self.faces_embedding:#先将与数据库中进行匹配上的进行绘制
                #将数据库中的每个拿出和场景中的人对比
                min_dist = 1024
                min_idx = -1
                user_name = "unknown"
                for idx,face in enumerate(faces):#比较当前向量与数据库中已知向量检测人员是否存在,如果数据库中某个人存在,找到与数据库中特征向量最对应的一个
                    embedding = np.array(face.embedding).reshape((1, -1))
                    embedding = preprocessing.normalize(embedding)
                    dis=self.feature_compare(embedding, com_face["feature"], 1.20)
                    if dis < min_dist:
                        min_dist = dis
                        min_idx =  idx
                        user_name=com_face["user_name"]
                if min_dist <= 1.20 and self.current_user_username!=user_name:#如果数据库中的元素匹配上,但并不是当前的,按照检测结果绘制
                    user_name = com_face["user_name"]
                    self.username_string += user_name + " "
                    bbox = np.array(faces[min_idx].bbox).astype(np.int32)
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    image = Image2.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(image)
                    draw.text((bbox[0], bbox[1] - 30),user_name,(0,255,255),font=font)
                    image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
                    age = faces[min_idx].age
                    gender = faces[min_idx].gender
                    print
                    cv2.putText(image, f"Age: {age}", (bbox[0], bbox[1] +10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(image, f"Gender: {gender}", (bbox[0], bbox[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    is_draw[min_idx] = False
                elif min_dist<=1.20 and self.current_user_username==user_name:#如果有匹配的,并且被匹配的是当前服务人物,按照存储的当前人物信息绘制
                    user_name = com_face["user_name"]
                    self.username_string += user_name + " "
                    find_person = True
                    bbox = np.array(faces[min_idx].bbox).astype(np.int32)
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 5)
                    image = Image2.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(image)
                    
                    draw.text((bbox[0], bbox[1] - 30),self.current_user_username,(0,255,255),font=font)
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    cv2.putText(image, f"Age: {self.current_user_age}", (bbox[0], bbox[1] +15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255 , 0), 3)
                    cv2.putText(image, f"Gender: {self.current_user_gender}", (bbox[0], bbox[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 3)
                    # cv2.putText(image, f"Favorite: {self.current_user_favor}", (bbox[0], bbox[1] + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 3)
                    center_x =640 // 2
                    center_y = 480 // 2
                    box_center_x = (bbox[0] + bbox[2]) // 2
                    box_center_y = (bbox[1] + bbox[3]) // 2
                    if abs(center_x - box_center_x) <=100 :
                        self.position_information ="聚焦人物在视野中央"
                    else:
                        self.position_information ="聚焦人物不在视野中央"
                    is_draw[min_idx] = False
            #再绘制未匹配上的
            for idx,element in enumerate(is_draw):
                if element:
                    user_name = "unknown"
                    bbox = np.array(faces[idx].bbox).astype(np.int32)
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    cv2.putText(image, f"{user_name}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    age = faces[idx].age
                    gender = faces[idx].gender
                    cv2.putText(image, f"Age: {age}", (bbox[0], bbox[1] +10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(image, f"Gender: {gender}", (bbox[0], bbox[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if find_person == False:
            self.position_information ="聚焦人物不存在"
        print(self.position_information)
        cv2.imshow("face_recognize",image)
        cv2.waitKey(1)
        return faces
    
    
    def request_face_recognition(self):#请求显示
        image=self.image   
        faces = self.model.get(image)
        return faces
    
    
    def register(self, user_name,binary_oldest_image):
        self.current_user = user_name
        #将字节流解码
        image=binary_oldest_image
        image= np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        print(image)
        #这个faces指的是人脸特征向量编码
        faces = self.model.get(image)
        # print("faces\n\n\n\n\n\n:",faces)
        # print(len(faces))
        # print("faces:",faces)
        if(faces!=None):
            print(len(faces))
        else:
            return 'failed',None
        if len(faces) != 1:
            return 'failed',None
        # 判断人脸是否存在
        embedding = np.array(faces[0].embedding).reshape((1, -1))
        #对人脸列表中第一个人的特征向量进行处理(其实只有一个人)
        embedding = preprocessing.normalize(embedding)
        self.current_user_embedding = embedding
        #对该向量归一化
        min_dist = 100000
        min_idx  = -1
        if len(self.faces_embedding)>0:#如果原数据库有内容
            for idx,com_face in enumerate(self.faces_embedding):#比较当前向量与数据库中已知向量检测人员是否存在
                #比对
                dist = self.feature_compare(embedding, com_face["feature"], self.threshold)
                if dist<min_dist:
                    min_dist = dist
                    min_idx  = idx                      
            if min_dist<1.12:
                os.remove(os.path.join(self.face_db, '%s.png' % com_face["user_name"]))
                self.faces_embedding[min_idx]["user_name"] = user_name
                self.faces_embedding[min_idx]["feature"] = embedding
                cv2.imencode('.png', image)[1].tofile(os.path.join(self.face_db, '%s.png' % user_name))
                # self.faces_embedding.append({
                #     "user_name": user_name,
                #     "feature": embedding
                # })
                rospy.sleep(0.5)
                #每次注册时将新人物特征存储下来后重新加载数据库中的图片
                self.load_faces(self.face_db)
                rospy.loginfo("此人已存在,现替换姓名及人脸向量以及原有图片")


                return "success",faces
            # 符合注册条件保存图片，同时把特征添加到人脸特征库中
            else:
                cv2.imencode('.png', image)[1].tofile(os.path.join(self.face_db, '%s.png' % user_name))
                # self.faces_embedding.append({
                #     "user_name": user_name,
                #     "feature": embedding
                # })
                rospy.sleep(0.5)
                #每次注册时将新人物特征存储下来后重新加载数据库中的图片
                self.load_faces(self.face_db)
                rospy.loginfo("新人员注册成功")


                return "success",faces
        else:#如果原数据库无内容,直接注册
            cv2.imencode('.png', image)[1].tofile(os.path.join(self.face_db, '%s.png' % user_name))
            # self.faces_embedding.append({
            #     "user_name": user_name,
            #     "feature": embedding
            # })
            rospy.sleep(0.5)
            #每次注册时将新人物特征存储下来后重新加载数据库中的图片
            self.load_faces(self.face_db)
            rospy.loginfo("新人员注册成功")

            return "success",faces
        
        
        
    def handle_face_recognition_service(self,request):
        faces=self.request_face_recognition()
        result_list = []
        #由于该数据他不能直接encode也不能直接序列化,将复杂列表变简单,numpy变为list,int64替换掉,float32替换掉
        #只有当检测到的图片中含有人脸时,才会将结果输出
        if(len(faces)!=0):#检测到了人
            #由于原来的检测结果较为复杂需要经过处理才能传回
            for element in faces:
                new_element = {}
                for key, value in element.items():
                    if isinstance(value, np.ndarray):
                        new_element[key] = value.astype(float).tolist()
                    else:
                        new_element[key] = value
                    if isinstance(value, np.float32):
                            new_element[key] = float(value)
                    elif isinstance(value, np.int64):
                        new_element[key] = int(value)
                result_list.append(new_element)
            faces=result_list
            #检测结果传回去,包含检测的特征信息,图片以及人脸是否在视野中央
            serialized_data = json.dumps(faces)
            serialized_data =serialized_data.encode()
            faces =zlib.compress(serialized_data)
            self.face_recognition_responce.faces = faces
            retval, buffer = cv2.imencode('.jpg', self.image)
            byte_data = buffer.tobytes()
            self.face_recognition_responce.picture = byte_data
            
            self.face_recognition_responce.position = self.position_information
            self.face_recognition_responce.result = "success"
            self.face_recognition_responce.user_find = self.username_string
            return self.face_recognition_responce
        else:#没检测到人
            self.face_recognition_responce.result = "failed"
            self.face_recognition_responce.user_find = " "
            return self.face_recognition_responce
    def handle_face_register_service(self,req):
        #使用接受到的二进制字节流进行人脸注册
        binary_oldest_image=req.binary_oldest_image
        self.current_user_username  =(req.username)
        self.current_user_age=req.age
        self.current_user_gender=req.gender
        self.current_user_favor=req.favor
        image=binary_oldest_image
        image= np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        result,faces=self.register(self.current_user_username,binary_oldest_image)
        print("registering")
        result_list = []
        if(len(faces)!=0):
            for element in faces:
                new_element = {}
                for key, value in element.items():
                    if isinstance(value, np.ndarray):
                        new_element[key] = value.astype(float).tolist()
                    else:
                        new_element[key] = value
                    if isinstance(value, np.float32):
                            new_element[key] = float(value)
                    elif isinstance(value, np.int64):
                        new_element[key] = int(value)
                result_list.append(new_element)
            #将face格式转换变为可以发送的格式
            faces=result_list
            #检测结果传回去,包含检测的特征信息,图片以及人脸是否在视野中央
            serialized_data = json.dumps(faces)
            serialized_data =serialized_data.encode()
            faces =zlib.compress(serialized_data)
            self.face_register_responce.faces = faces
        else:
            faces = None
        self.face_register_responce.result = result
        return self.face_register_responce
     
     
        
    def load_faces(self, face_db_path):
        #每次加载特征时将原内容清空
        self.faces_embedding = list()
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        for root, dirs, files in os.walk(face_db_path):
            for file in files:
                input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
                user_name = (file.split(".")[0])
                face = self.model.get(input_image)[0]
                embedding = np.array(face.embedding).reshape((1, -1))
                embedding = preprocessing.normalize(embedding)
                self.faces_embedding.append({
                    "user_name": user_name,
                    "feature": embedding
                })

                

    @staticmethod
    def feature_compare(feature1, feature2, threshold):
        diff = np.subtract(feature1, feature2)
        dist = np.sum(np.square(diff), 1)
        return dist 
         


if __name__ == '__main__':
    try:
        face_recognition_server = FaceRecognitionServer()
        rospy.loginfo('Face recognition service is running.')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
