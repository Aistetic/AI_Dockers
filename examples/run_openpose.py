"""
Author: dizhong zhu
Date: 16/06/2022

you might need to specify the url and port number yourself
"""

import requests
import cv2
import io
import numpy as np
from PIL import Image


def run_openpose(images, debug_path=None):
    """
    :param images: The images in size [N,H,W,C]
    :param debug_path: You must specify the debug_path as absolute value according to your docker volumn mapping
    :return:
    """
    openpose_url = "http://{}:{}/openpose".format('localhost', '9999')

    multiple_images = []
    for i, image in enumerate(images):
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        img_byte_io = io.BytesIO(im_buf_arr)
        multiple_images.append(("images", img_byte_io))


    data = {"debug_path": debug_path} if debug_path is not None else {}

    response = requests.post(openpose_url, files=multiple_images, data=data)

    if response.status_code != 200:
        raise Exception('Pose detection failed')
    else:
        output = response.json()
        landmarks = output["landmarks"]
        mask = output['masks']

        for k,v in landmarks.items():
            landmarks[k] = np.stack(v)[mask]
        # landmarks['pose_landmarks'] = np.stack(landmarks['pose_landmarks'])[mask]
        # landmarks['face_landmarks'] = np.stack(landmarks['face_landmarks'])[mask]

        images = images[mask]

    return landmarks, images


if __name__ == '__main__':
    image = Image.open('examples/datas/Happy-single-1.jpg')
    image = np.array(image)[None, ...]

    run_openpose(image, '/Debug/2')
