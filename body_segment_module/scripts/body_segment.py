import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import paddlehub as hub
import os
from stereo_msgs.msg import DisparityImage
from body_segment_module.srv import body_segmentService,body_segmentServiceRequest,body_segmentServiceResponse
import pickle
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
image_depth = None
sum_depth = 0
class PeopleSegmentationROS:
    def __init__(self):
        self.module = hub.Module(name="deeplabv3p_xception65_humanseg")
        self.segmented_image = None
        self.contours = tuple()
        self.binary_mask = None
        self.bridge = CvBridge()
        self.ratio=None
        self.result = None
        self.disparity_map = None
        self.depth = 0
        self.cX = 640
        self.cY = 240
        self.mean_distance = 0
        rospy.init_node('body_segment_server')
        # Subscribe to the input ROS topic (e.g., '/image_raw')
        rospy.Subscriber('/camera/rgb/image_color', Image, self.segment_people_callback)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.receive_depth_image)
        rospy.Subscriber('/camera/depth_registered/disparity', DisparityImage, self.disparity_callback)
        rospy.Service('body_segment', body_segmentService, self.body_segment_service)
        # Create a publisher for the segmentation mask
        # self.segmented_mask_pub = rospy.Publisher('/segmented_mask', Image, queue_size=10)
        # # Create a publisher for the contour information
        # self.contours_pub = rospy.Publisher('/contours', String, queue_size=10)
        self.body_segment_service_response = body_segmentServiceResponse()
    def receive_depth_image(self,image):
        bridge = CvBridge()
        global image_depth
        image_depth = bridge.imgmsg_to_cv2(image, desired_encoding="32FC1")
        
    def disparity_callback(self,msg):
        bridge = CvBridge()
        
        self.disparity_map = bridge.imgmsg_to_cv2(msg.image, desired_encoding="passthrough")

       
    def get_distance(self,center_x,center_y):
        # 根据视差值计算深度值
        if(self.disparity_map is not None):
            focal_length = 525.0  # 焦距（单位：像素）
            baseline = 0.054  # 左右摄像机的基线（单位：米）
            disparity = self.disparity_map[center_y, center_x]
            if(disparity<=0.01):
                print("中心点的深度值大小：", self.depth)
                return  self.depth
            else:
                depth = (focal_length * baseline) / disparity
                if(depth != math.inf):#在不是无穷大的时候,赋予新的值
                    self.depth = (focal_length * baseline) / disparity
                return  self.depth
        else:
            return 0.4#检测不到人时,距离就默认为不需要调整
            
                  
    def segment_people_callback(self, img_msg):
        try:
            # Convert ROS image message to OpenCV image
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
            # Segment people in the image
            results = self.module.segmentation(images=[cv_img], use_gpu=True)
            for result in results:
                self.segmented_image = np.array(result['data'], dtype=np.uint8)
                _, self.binary_mask = cv2.threshold(self.segmented_image, 1, 255, cv2.THRESH_BINARY)
                self.contours, _ = cv2.findContours(self.binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if self.contours :
            #同时时刻进行人脸识别不过每个人都打上不知道的标签
                # cv2.drawContours(cv_img, self.contours, -1, (0, 255, 0), 2)
                # cv2.imshow("body_segment",image_depth)  
                # cv2.waitKey(1)
                # masked_image = np.where(self.segmented_image >= 1, image_depth, 0)  # 获取掩膜图像中的像素值
                # pixel_sum = np.sum(masked_image)   
                
                # 找轮廓中心点
                center_points = []
                distances = []
                for contour in self.contours:
                    # 计算轮廓的矩
                    M = cv2.moments(contour)
                    # 获取轮廓的重心坐标
                    if M["m00"] != 0:
                        self.cX = int(M["m10"] / M["m00"])
                        self.cY = int(M["m01"] / M["m00"])
                        center_points.append((self.cX, self.cY))
                        # 在轮廓图像上绘制中心点
                        cv2.circle(cv_img, (self.cX, self.cY), 3, (0, 255, 0), -1)
                    distance=self.get_distance(self.cX, self.cY)
                    distances.append(distance)
                self.mean_distance = np.mean(distances)
                cv2.drawContours(cv_img, self.contours, -1, (0, 255, 0), 2)
                cv2.imshow("body_segment",cv_img)  
                cv2.waitKey(1)
                white_pixels = np.sum(self.segmented_image >= 1)
                total_pixels = self.segmented_image.size
                ratio = white_pixels / total_pixels
                # print("距离为:",float(pixel_sum/white_pixels))
                self.result = "success"
                
                
                
                if ratio >= 0.35:
                    rospy.loginfo("People segmented successfully!")
                else:
                    rospy.loginfo("People didn't fully attend.")
                self.ratio=ratio
                print(self.ratio)
            else:
                cv2.imshow("body_segment",cv_img)  
                rospy.loginfo("No people detected in the image.")
                cv2.waitKey(1)
                self.contours = tuple()
                self.result = "failed"
                self.ratio = 0
        except CvBridgeError as e:
            rospy.logerr(e)
    def body_segment_service(self,request):
        if(self.contours!=tuple()):
            serialized_message = pickle.dumps(self.contours)
            self.body_segment_service_response.contours=serialized_message
        else:
            self.body_segment_service_response.contours = tuple()
        self.body_segment_service_response.result = self.result
        self.body_segment_service_response.ratio = self.ratio
        self.body_segment_service_response.cX = self.cX
        self.body_segment_service_response.cY = self.cY
        self.body_segment_service_response.mean_distance = self.mean_distance
        return self.body_segment_service_response  
       
if __name__ == "__main__":
    try:
        segmentation_node = PeopleSegmentationROS()
        rospy.loginfo('Body segement service is running.')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    
    
