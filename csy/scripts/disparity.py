import rospy
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge
import numpy as np

def image_callback(msg):
    bridge = CvBridge()
    disparity_map = bridge.imgmsg_to_cv2(msg.image, desired_encoding="passthrough")

    # 获取图像尺寸
    height, width = disparity_map.shape[:2]

    # 获取中心点坐标
    center_x = width // 2
    center_y = height // 2

    # 根据视差值计算深度值
    focal_length = 525.0  # 焦距（单位：像素）
    baseline = 0.054  # 左右摄像机的基线（单位：米）
    disparity = disparity_map[center_y, center_x]
    depth = (focal_length * baseline) / disparity

    print("中心点的深度值大小：", depth)

def main():
    rospy.init_node('depth_subscriber')
    rospy.Subscriber('/camera/depth_registered/disparity', DisparityImage, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()

