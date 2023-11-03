import os
from sensor_msgs.msg import Image
import cv2
import insightface
import numpy as np
from sklearn import preprocessing
from cv_bridge import CvBridge
import rospy
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
faces = None
class FaceRecognition:
    def __init__(self, gpu_id=0, face_db='face_db', threshold=1.24, det_thresh=0.50, det_size=(640, 640)):
        """
        人脸识别工具类
        :param gpu_id: 正数为GPU的ID，负数为使用CPU
        :param face_db: 人脸库文件夹
        :param threshold: 人脸识别阈值
        :param det_thresh: 检测阈值
        :param det_size: 检测模型图片大小
        """
        self.gpu_id = gpu_id
        self.face_db = face_db
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.det_size = det_size

        # 加载人脸识别模型，当allowed_modules=['detection', 'recognition']时，只单纯检测和识别
        self.model = insightface.app.FaceAnalysis(root='./',
                                                  allowed_modules=None,
                                                  providers=['CUDAExecutionProvider'])
        #不需要加载人脸加载好库,仅仅需要把模型
        self.model.prepare(ctx_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)
        # 人脸库的人脸特征
        self.faces_embedding = list()
        # 加载人脸库中的人脸
        self.load_faces(self.face_db)

    # 加载人脸库中的人脸
    def load_faces(self, face_db_path):
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        for root, dirs, files in os.walk(face_db_path):
            for file in files:
                input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
                user_name = file.split(".")[0]
                face = self.model.get(input_image)[0]
                embedding = np.array(face.embedding).reshape((1, -1))
                embedding = preprocessing.normalize(embedding)
                self.faces_embedding.append({
                    "user_name": user_name,
                    "feature": embedding
                })

    # 人脸识别
    def recognition(self, image):
        # bridge = CvBridge()
        # image = bridge.imgmsg_to_cv2(image, "bgr8")
        faces = self.model.get(image)#对图像中每个人物进行识别
        return faces
        # for face in faces:
        #     # 开始人脸识别
        #     embedding = np.array(face.embedding).reshape((1, -1))
        #     embedding = preprocessing.normalize(embedding)
        #     user_name = "unknown"
        #     for com_face in self.faces_embedding:
        #         r = self.feature_compare(embedding, com_face["feature"], self.threshold)
        #         if r:
        #             user_name = com_face["user_name"]
        #     bbox = np.array(face.bbox).astype(np.int32)
        #     cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        #     cv2.putText(image, f"{user_name}, {face.age}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        # cv2.imshow("detected:",image) 
        # cv2.waitKey(1)
        

    @staticmethod
    def feature_compare(feature1, feature2, threshold):
        diff = np.subtract(feature1, feature2)
        dist = np.sum(np.square(diff), 1)
        if dist < threshold:
            return True
        else:
            return False

    def register(self, image, user_name):
        
        faces = self.model.get(image)
        print(len(faces))
        print("faces:",faces)
        # if len(faces) != 1:
        #     return '图片检测不到人脸'
        # 判断人脸是否存在
        embedding = np.array(faces[0].embedding).reshape((1, -1))
        #对人脸列表中第一个人的特征向量进行处理
        embedding = preprocessing.normalize(embedding)
        #对该向量归一化
        is_exits = False
        for com_face in self.faces_embedding:#比较当前向量与数据库中已知向量检测人员是否存在
            r = self.feature_compare(embedding, com_face["feature"], self.threshold)
            if r:
                is_exits = True
        if is_exits:
            return '该用户已存在'
        # 符合注册条件保存图片，同时把特征添加到人脸特征库中
        cv2.imencode('.png', image)[1].tofile(os.path.join(self.face_db, '%s.png' % user_name))
        self.faces_embedding.append({
            "user_name": user_name,
            "feature": embedding
        })
        return "success"

    # 检测人脸
    # def detect(self, image):
    #     global faces
    #     faces = self.model.get(image)
        
    #     results = list()
    #     for face in faces:
    #         result = dict()
    #         # 获取人脸属性
    #         result["bbox"] = np.array(face.bbox).astype(np.int32).tolist()
    #         result["kps"] = np.array(face.kps).astype(np.int32).tolist()
    #         result["landmark_3d_68"] = np.array(face.landmark_3d_68).astype(np.int32).tolist()
    #         result["landmark_2d_106"] = np.array(face.landmark_2d_106).astype(np.int32).tolist()
    #         result["pose"] = np.array(face.pose).astype(np.int32).tolist()
    #         result["age"] = face.age
    #         gender = '男'
    #         if face.gender == 0:
    #             gender = '女'
    #         result["gender"] = gender
    #         # 开始人脸识别
    #         embedding = np.array(face.embedding).reshape((1, -1))
    #         embedding = preprocessing.normalize(embedding)
    #         result["embedding"] = embedding
    #         results.append(result)
    #     return results


if __name__ == '__main__':
    # img = cv2.imdecode(np.fromfile('tmc10.jpg', dtype=np.uint8), -1)
    face_recognitio = FaceRecognition()
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/camera/rgb/image_color", Image, face_recognitio.recognition)
    # # 人脸注册
    # result = face_recognitio.register(img, user_name='tmcn')
    # print("result:",result)

    # 人脸识别
    
    # for result in results:
    #     print("识别结果：{}".format(result))

    # results = face_recognitio.detect(img)
    # for result in results:
    #     print('人脸框坐标：{}'.format(result["bbox"]))
    #     print('人脸五个关键点：{}'.format(result["kps"]))
    #     print('人脸3D关键点：{}'.format(result["landmark_3d_68"]))
    #     print('人脸2D关键点：{}'.format(result["landmark_2d_106"]))
    #     print('人脸姿态：{}'.format(result["pose"]))
    #     print('年龄：{}'.format(result["age"]))
    #     print('性别：{}'.format(result["gender"]))
    rospy.spin()