import rospy
from geometry_msgs.msg import PoseStamped
from actionlib_msgs.msg import GoalStatusArray
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import geometry_msgs.msg as gm
import math
import tf.transformations as tftr
import tf
import tf2_ros
from tf2_ros import TransformListener

class Navigator:
    def __init__(self):
        #一定要初始化节点否则会显示AttributeError: 'TransformListener' object has no attribute 'tf_sub'

        rospy.init_node('main_control', anonymous=True)
        rospy.Subscriber('/odom', Odometry, self.odom_callback) 
        self.goal = None
        self.goal_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.angle_difference = 1000
        self.distance_to_goal = 1000
       
        self.buffer = tf2_ros.Buffer()#创建一个用于存储坐标变换信息的数据结构
        self.angle_threshold = 14 #设置角度阈值
        self.listener = tf2_ros.TransformListener(self.buffer)#创建用于监听和查询坐标变换
        self.orientation = None#表示当前朝向
        self.position = None#表示当前位置
        
       

    def navigate_to_point(self,entrance_position):
        # 初始化ROS节点
        # 根据传参创建目标点消息
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = entrance_position[0]
        goal.pose.position.y = entrance_position[1]
        goal.pose.position.z = entrance_position[2]
        goal.pose.orientation.x = entrance_position[3]
        goal.pose.orientation.y = entrance_position[4]
        goal.pose.orientation.z = entrance_position[5]
        goal.pose.orientation.w = entrance_position[6]  # 默认朝向
        #将目标参数存入类的属性里面
        self.goal=goal
        # 发布目标点消息
        self.goal_publisher.publish(goal)
        #显示发布的目标点信息
        rospy.loginfo('Published navigation goal: (p_x={}, p_y={}, p_z={}, o_x={}, o_y={}, o_z={}, o_w={})'.format(
    goal.pose.position.x, goal.pose.position.x, goal.pose.position.z, goal.pose.orientation.x,
    goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w))

    
        """时刻计算当前位置和目标位置以及二者之前的差距
        """
    def odom_callback(self, msg):
        #下面这一行代码获取的是底盘的坐标信息,没有经过坐标变换,没啥用
        self.current_pose = msg.pose.pose
        if self.goal is not None:
            #在有目标的情况下
            #获取从base_link到map的坐标变化
            transform_stamped = self.buffer.lookup_transform('map', 'base_link', rospy.Time(0))
            self.position = transform_stamped.transform.translation
            self.orientation = transform_stamped.transform.rotation
            #计算当前位置在map下的位置
            #得到目标位置与实际位置之间的差距
            self.distance_to_goal = math.sqrt(((self.position.x - self.goal.pose.position.x) ** 2 +
                                (self.position.y - self.goal.pose.position.y) ** 2))
            # print(self.orientation)
            # print(self.goal.pose.orientation)
            #计算当前
            #计算当前实际朝向角
            _,_,current_yaw = self.calculate_yaw([ self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w])  # 计算当前朝向的偏航角度
            #计算目标朝向角
            _,_,goal_yaw = self.calculate_yaw([self.goal.pose.orientation.x,self.goal.pose.orientation.y,self.goal.pose.orientation.z,self.goal.pose.orientation.w])  # 计算目标朝向的偏航角度
            #计算两个朝向角的差距
            self.angle_difference = abs(current_yaw - goal_yaw)
            rospy.sleep(0.1)
            # print("current:",current_yaw)
            # print("goal:",goal_yaw)
            
        
    """获取目标点位是否到达"""
    def is_goal_reached(self):
        print("distance_to_goal:",self.distance_to_goal)
        print("angle_difference:",self.angle_difference)
        if (self.distance_to_goal < 0.18) and (self.angle_difference < self.angle_threshold):  # 调整此阈值以控制达到目标点的距离
                rospy.loginfo('Reached navigation goal!')
                return True
        else:
           
            return False
    #获取当前位置
    def get_current_position(self):
        # print(self.orientation)
        return self.orientation,self.position#返回当前实际的朝向与位置

    def calculate_yaw(self,quaternion):
        #用四元数计算欧拉角
        roll, pitch, yaw = tftr.euler_from_quaternion(quaternion)
        # 将欧拉角从弧度转换为度(其实只有yaw有用)
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        return roll_deg,pitch_deg,yaw_deg

