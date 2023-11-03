import rospy
from face_recognize.srv import FaceRecognitionService,FaceRecognitionServiceResponse, FaceRecognitionServiceRequest
from face_recognize.srv import FaceregisterService, FaceregisterServiceRequest
import json
import zlib
from std_msgs.msg import String


def face_recoginze():
    rospy.wait_for_service('face_recognition')
    face_recognition_service = rospy.ServiceProxy('face_recognition', FaceRecognitionService)
    # face_register_service = rospy.ServiceProxy('face_register', FaceregisterService)
    # 调用人脸识别服务
    face_recognition_request = FaceRecognitionServiceRequest()
    face_recognition_response = face_recognition_service(face_recognition_request)
    
    return face_recognition_response



if __name__ == "__main__":
    rospy.init_node("face_recognize_client")
    face_recoginze()
#调用函数