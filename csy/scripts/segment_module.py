import cv2
import numpy as np
import paddlehub as hub
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class PeopleSegmentation:
    def __init__(self):
        self.module = hub.Module(name="deeplabv3p_xception65_humanseg")
        self.segmented_image = None
        self.contours=None
        self.binary_mask=None
    def segment_people(self, img):
        results = self.module.segmentation(images=[img], use_gpu=True)
        
        for result in results:
            self.segmented_image = np.array(result['data'], dtype=np.uint8)
            _, self.binary_mask = cv2.threshold(self.segmented_image, 1, 255, cv2.THRESH_BINARY)
            self.contours, _ = cv2.findContours(self.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            # print(self.segmented_image)
            
        white_pixels = np.sum(self.segmented_image >= 100)

        # 计算图片像素总数
        total_pixels = self.segmented_image.size

        # 计算人员面积占图片面积的比例
        ratio = white_pixels / total_pixels

        # 比较比例和0.5，并返回True或False
        seg=True
        notseg=False
        # print(ratio)
        if ratio >= 0.4:
            return [seg,self.binary_mask,self.contours]
        else:
            return [notseg,self.binary_mask,self.contours]