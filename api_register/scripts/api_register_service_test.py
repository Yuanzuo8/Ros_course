







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

import rospy
from face_recognize.srv import FaceRecognitionService,FaceRecognitionServiceResponse, FaceRecognitionServiceRequest
from face_recognize.srv import FaceregisterService, FaceregisterServiceRequest
import json
import zlib
from std_msgs.msg import String
from api_register.srv import Api_Resister_Service, Api_Resister_ServiceResponse,Api_Resister_ServiceRequest

def api_register(hobby,username):
    rospy.wait_for_service('api_register')
    api_register_service = rospy.ServiceProxy('api_register', Api_Resister_Service)
   
    api_register_request = Api_Resister_ServiceRequest()
    api_register_request.hobby=hobby
    api_register_request.username=username
    api_register_response = api_register_service(api_register_request)
    return api_register_response
if __name__ == "__main__":
  
    hobby  = "coke"
    username = "jdt"
    api_register(hobby,username)