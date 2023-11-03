import requests
import cv2
import json
import base64
import urllib.parse

class GetPersonalInformation:
    def __init__(self):
        pass

    def get_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=CZlHcyjavzoiqM7hEsBMaNDK&client_secret=2ouD8L6XGNrmsL4v6uSwBpatPitl1As1"

        payload = ""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        data = response.json()
        access_token = data["access_token"]
        print(access_token)
        return access_token

    def get_detect(self, img):
        url = "https://aip.baidubce.com/rest/2.0/face/v3/detect?access_token=" + self.get_token()
        cv2.imwrite('1.jpg', img)
        image_name = "1.jpg"
        image = self.get_file_content_as_base64(image_name, False)
        payload = json.dumps({
            "image": image,
            "max_face_num": 10,
            "image_type": "BASE64",
            "face_field": "faceshape,facetype,age,gender,glasses"
            
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        # print(response.text)
        detect = response.json()
        return detect

    def get_file_content_as_base64(self, path, urlencoded=False):
        """
        获取文件base64编码
        :param path: 文件路径
        :param urlencoded: 是否对结果进行urlencoded
        :return: base64编码信息
        """
        with open(path, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
        return content
