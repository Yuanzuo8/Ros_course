import rospy
from std_msgs.msg import String
from face_recognize.srv import FaceregisterService, FaceregisterServiceResponse,FaceregisterServiceRequest
import json
import zlib
import cv2
import numpy as np
def face_register(byte_data,username,gender,hobby,age):#注册时传参为字节流
    
    rospy.wait_for_service('face_register')#服务名称
    face_register_service = rospy.ServiceProxy('face_register', FaceregisterService)
    face_register_request = FaceregisterServiceRequest()
    # 设置人脸图像和用户名
    #先解码看一看效果
    # image=byte_data
    # image= np.frombuffer(image, dtype=np.uint8)
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # cv2.imwrite("test.jpg",image)
    face_register_request.binary_oldest_image = byte_data
    face_register_request.username = username
    face_register_request.age = age
    face_register_request.favor = hobby
    face_register_request.gender = gender
    face_register_response = face_register_service(face_register_request)
    result = face_register_response.result
   
    return face_register_response
    
#测试
if __name__ == "__main__":
    rospy.init_node("face_register_client")
    img = cv2.imread("test.jpg")
    retval, buffer = cv2.imencode('.jpg', img)
    byte_data = buffer.tobytes()
    face_register(byte_data,"test")

