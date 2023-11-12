"""
Author: dizhong zhu
Date: 14/10/2022
"""
import shutil

from openpose import pyopenpose as op
from flask import Flask, request, jsonify, json, Response
from types import SimpleNamespace
import os
from pathlib import Path
import numpy as np
import cv2
import argparse
import yaml
from collections import OrderedDict
from fnmatch import fnmatch
from PIL import Image
import time
import sys

server = Flask(__name__)


def draw_landmarks(img, landmarks, color=(0, 255, 0), radius=2):
    for i in range(len(landmarks)):
        # print(len(landmarks))
        cv2.circle(img, (int(landmarks[i][0]), int(landmarks[i][1])), radius, color, -1)
        cv2.putText(img, str(i), (int(landmarks[i][0]), int(landmarks[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img


def render_landmarks(images, landmarks, debug_path):
    Path(debug_path).mkdir(parents=True, exist_ok=True)
    # get the length of the keypoints
    for key, val in landmarks.items():
        num_data = len(val)
        break

    for i in range(num_data):
        pose_img = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        for key, val in landmarks.items():
            # print('what what ??????????//')
            pose_img = draw_landmarks(pose_img, val[i])

        cv2.imwrite(os.path.join(debug_path, f'{i}_keypoints.jpg'), pose_img)

        # plot the joints on the image


def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)


def run_openpose(images, images_name, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process image
    poseKeypoints = []
    faceKeypoints = []
    image_masks = np.ones(shape=images.shape[0], dtype=np.bool)

    for i in range(images.shape[0]):
        datum = op.Datum()
        datum.cvInputData = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        if sys.getrefcount(datum.poseKeypoints) < 1:
            print(sys.getrefcount(datum.poseKeypoints))

        if datum.poseKeypoints is None or datum.poseKeypoints.size <= 1:  # if we cannot detect any pose from this image, then discard it
            image_masks[i] = False
            poseKeypoint = []  # np.zeros(shape=(25, 3), dtype=np.float32)
        else:
            poseKeypoint = datum.poseKeypoints[0].astype(np.float16).tolist()
        poseKeypoints.append(poseKeypoint)

        if datum.faceKeypoints is None or datum.faceKeypoints.size <= 1:
            faceKeypoint = []  # np.zeros(shape=(70, 3), dtype=np.float32)
        else:
            faceKeypoint = datum.faceKeypoints[0].astype(np.float16).tolist()

        faceKeypoints.append(faceKeypoint)

        if output_dir is not None:
            # cv2.imwrite(os.path.join(output_dir, "{0}.jpg".format(i)), datum.cvOutputData)
            json_data = OrderedDict(
                {
                    "version": 1.7,
                    "people": [
                        OrderedDict(
                            {
                                "person_id": [-1],
                                "pose_keypoints_2d": poseKeypoint,
                                "face_keypoints_2d": faceKeypoint,
                                "hand_left_keypoints_2d": [],
                                "hand_right_keypoints_2d": [],
                                "pose_keypoints_3d": [],
                                "face_keypoints_3d": [],
                                "hand_left_keypoints_3d": [],
                                "hand_right_keypoints_3d": [],
                            }
                        )
                    ],
                }
            )
            with open(os.path.join(output_dir, f"{images_name[i]}_keypoints.json"), "w") as f:
                json.dump(json_data, f)

    landmarks = {
        "pose_landmarks": poseKeypoints,
        "face_landmarks": faceKeypoints,
    }

    return landmarks, image_masks.tolist()


def parse_args():
    parser = argparse.ArgumentParser(description="Run algorithm on real images", allow_abbrev=False)
    parser.add_argument("-render_pose", default="1", help="whether render the pose in image, 0: off, 1: on. Off will improve the speed", )
    parser.add_argument('-model_folder', default="/app/model/Openpose_model", help="path to the folder containing the pre-trained models")
    parser.add_argument('-number_people_max', default="1", help="maximum number of people to detect")
    parser.add_argument('-disable_multi_thread', default="True", help="disable multi thread")
    parser.add_argument('-maximize_positives', default="True", help="maximize positives")
    parser.add_argument('-face', default="False", help="Turn on the face detection")
    parser.add_argument('-hand', default="False", help="Turn on the hand detection")
    parser.add_argument('-input_folder', type=str, help="The synthesis data root folder", )

    args = parser.parse_args()

    openpose_params = {
        "render_pose": args.render_pose,
        "model_folder": args.model_folder,
        "number_people_max": args.number_people_max,
        "disable_multi_thread": args.disable_multi_thread,
        "maximize_positives": args.maximize_positives,
        "face": args.face,
        "hand": args.hand,
        'display': 0,
        # 'write_json': './keypoints',
        # 'write_images': './keypoints_rendered'
    }

    return openpose_params, args


def init_openpose_model(openpose_params, bRender=True):
    params = openpose_params

    # if bRender is True:
    #     params['render_pose'] = 1
    # else:
    #     params['render_pose'] = 0
    #     del params['write_images']

    # starting OpenPose wrappers for each image, it might be slow, but avoid multi thread problem
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    return opWrapper


if __name__ == '__main__':
    openpose_params, args = parse_args()

    # opWrapper_rendered = init_openpose_model(openpose_params)
    opWrapper = init_openpose_model(openpose_params, bRender=False)

    input_folder = args.input_folder

    subjects = sorted(os.listdir(input_folder))
    max_num = len(subjects)

    flag = False

    count = 140000
    # if os.path.exists('record.npz'):
    #     with np.load('record.npz') as data:
    #         count = data['count']

    for i, id in enumerate(subjects):
        if i < count: continue

        print(f'processing {i}/{max_num} subject', flush=True)
        img_folder = os.path.join(input_folder, id, 'images')
        save_dir = os.path.join(input_folder, id, 'keypoints')
        # Path(save_dir).mkdir(parents=True, exist_ok=True)

        listfiles = sorted(os.listdir(img_folder))
        images = []
        images_name = []
        for k, f in enumerate(listfiles):
            if fnmatch(f, '*.jpg') or fnmatch(f, '*.JPG') or fnmatch(f, '*.png') or fnmatch(f, '*.PNG'):
                img = Image.open(os.path.join(img_folder, f))
                images_name.append(Path(f).stem)
                images.append(img)

        images = np.stack(images)
        landmarks, _ = run_openpose(images, images_name, save_dir)

        if i <= 10:
            render_dir = os.path.join(input_folder, id, 'keypoints_rendered')
            render_landmarks(images, landmarks, render_dir)

        # np.savez('record.npz', count=i)

        # break
        # if i <= 0:
        #     copy_and_overwrite('./keypoints_rendered', os.path.join(input_folder, id, 'keypoints_rendered'))
        # else:
        #     if flag is False:
        #         opWrapper.stop()
        #         opWrapper = init_openpose_model(openpose_params, bRender=False)
        #         flag = True

        # copy_and_overwrite('./keypoints', os.path.join(input_folder, id, 'keypoints'))
