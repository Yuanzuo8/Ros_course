import paddle
import paddlehub as hub
import cv2
import os
import matplotlib.pyplot  as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class FaceDetector:
    def __init__(self, model_name="ultra_light_fast_generic_face_detector_1mb_640"):
        paddle.set_device('gpu')  # 设置PaddlePaddle使用的GPU设备
        self.face_detector = hub.Module(name=model_name)

    def detect_faces(self, img_color):
        paddle.set_device('gpu')  # 设置PaddlePaddle使用的GPU设备
        
        plt.imshow(img_color)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        result = self.face_detector.face_detection(images=[img_color], use_gpu=True)
        faces = result[0].get("data")
        return faces

    # @staticmethod
    # def draw_boxes(image_path, faces):
    #     image = cv2.imread(image_path)

    #     for face in faces:
    #         left, top, right, bottom = face['left'], face['top'], face['right'], face['bottom']
    #         cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    #     cv2.imshow('Faces', image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
