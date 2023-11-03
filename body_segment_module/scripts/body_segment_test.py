import rospy
from std_msgs.msg import String
from body_segment_module.srv import body_segmentService,body_segmentServiceRequest,body_segmentServiceResponse
import json
import zlib
import cv2
import pickle
def body_segment():
    rospy.wait_for_service('body_segment')#服务名称
    body_segment_service = rospy.ServiceProxy('body_segment', body_segmentService)
    body_segment_request = body_segmentServiceRequest()
    # 设置人脸图像和用户名
    body_segment_response = body_segment_service(body_segment_request)
    result = body_segment_response.result
    print(result)
    # contours = pickle.loads(body_segment_response.contours)
    # print(contours)
    # print(type(contours))
    
    return body_segment_response
if __name__ == "__main__":
    rospy.init_node('body_segment_client')
    ratio =0
    while ratio <=0.4:
        ratio=body_segment().ratio