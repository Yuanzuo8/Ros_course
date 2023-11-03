#!/usr/bin/python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
depth_image = None
color_image = None

# Define camera parameters as numpy arrays
camera_matrix = np.array([[512.813704, 0.000000, 319.979970], [0.000000, 513.251183, 233.708619], [0.000000, 0.000000, 1.000000]])
distortion = np.array([-0.007907, -0.035312, -0.003307, -0.002599, 0.000000])
rectification = np.array([[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000]])
projection = np.array([[508.524393, 0.000000, 318.353220, 0.000000], [0.000000, 511.451526, 232.473578, 0.000000], [0.000000, 0.000000, 1.000000, 0.000000]])

def depth_callback(data):
    global depth_image
    depth_image = bridge.imgmsg_to_cv2(data)

def color_callback(data):
    global color_image
    color_image = bridge.imgmsg_to_cv2(data)

    # Transform depth image to match color image
    depth_image_rectified = cv2.undistortPoints(depth_image, camera_matrix, distortion, None, projection)

rospy.init_node("kinect_node")

rospy.Subscriber("/camera/depth/image_raw", Image, depth_callback)
rospy.Subscriber("/camera/rgb/image_color", Image, color_callback)

cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
cv2.namedWindow("Color", cv2.WINDOW_NORMAL)

while not rospy.is_shutdown():
    # Wait for depth and color images to be available
    while depth_image is None or color_image is None:
        rospy.sleep(0.1)
    
    # Display depth and color images
    cv2.imshow("Depth", depth_image_rectified)
    cv2.imshow("Color", color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

