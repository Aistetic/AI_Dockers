"""
Author: dizhong zhu
Date: 09/08/2021
"""

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

server = Flask(__name__)
opWrapper = None


@server.route("/openpose", methods=["post"])
def app():
    # parse the input
    ret, error_info, res = parse_input(request)
    if ret is False:
        print(error_info)
        return jsonify({"body": error_info}), 201

    images, debug_path = res
    landmarks, masks = run_openpose(images, debug_path)

    response = {
        "landmarks": landmarks,
        "masks": masks,
    }
    return jsonify(response), 200


def parse_input(request):
    files = request.files.to_dict(flat=False)
    data = request.form.to_dict()
    if 'images' not in files.keys():
        error_info = 'There is no images in the input file'
        return False, error_info, None

    if 'debug_path' not in data.keys():
        debug_path = None
    else:
        debug_path = data['debug_path']

    # save the images in the memory
    images = []
    for i, file in enumerate(files["images"]):
        nparry = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparry, cv2.IMREAD_COLOR)
        images.append(img)

    images = np.stack(images)

    return True, '', (images, debug_path)


def run_openpose(images, debug_path):
    output_dir = None
    if debug_path is not None:
        output_dir = debug_path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process image
    poseKeypoints = []
    faceKeypoints = []
    leftHandKeypoints = []
    rightHandKeypoints = []
    image_masks = np.ones(shape=images.shape[0], dtype=np.bool)

    for i in range(images.shape[0]):
        datum = op.Datum()
        datum.cvInputData = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        if datum.poseKeypoints is None or datum.poseKeypoints.size < 1:  # if we cannot detect any pose from this image, then discard it
            image_masks[i] = False
            poseKeypoint = np.zeros(shape=(1, 25, 3), dtype=np.float32)
        else:
            poseKeypoint = datum.poseKeypoints
        poseKeypoints.append(poseKeypoint.tolist())

        if datum.faceKeypoints is None or datum.faceKeypoints.size < 1:
            faceKeypoint = np.zeros(shape=(1, 70, 3), dtype=np.float32)
        else:
            faceKeypoint = datum.faceKeypoints
        faceKeypoints.append(faceKeypoint.tolist())

        if datum.handKeypoints is None or datum.handKeypoints[0].size < 1:
            leftHandKeypoint = np.zeros(shape=(1, 21, 3), dtype=np.float32)
            rightHandKeypoint = np.zeros(shape=(1, 21, 3), dtype=np.float32)
        else:
            leftHandKeypoint = datum.handKeypoints[0]
            rightHandKeypoint = datum.handKeypoints[1]
        leftHandKeypoints.append(leftHandKeypoint.tolist())
        rightHandKeypoints.append(rightHandKeypoint.tolist())

        if output_dir is not None:
            if len(datum.cvOutputData) > 0:
                cv2.imwrite(os.path.join(output_dir, "{0}.jpg".format(i)), datum.cvOutputData)

            peoples = []
            # conver the keypoints to json format
            for k in range(len(poseKeypoint)):
                peoples.append(OrderedDict(
                    {
                        "person_id": [k],
                        "pose_keypoints_2d": poseKeypoint[k].tolist(),
                        "face_keypoints_2d": faceKeypoint[k].tolist(),
                        "hand_left_keypoints_2d": leftHandKeypoint[k].tolist(),
                        "hand_right_keypoints_2d": rightHandKeypoint[k].tolist(),
                        "pose_keypoints_3d": [],
                        "face_keypoints_3d": [],
                        "hand_left_keypoints_3d": [],
                        "hand_right_keypoints_3d": [],
                    }
                ))

            json_data = OrderedDict(
                {
                    "version": 1.7,
                    "num_people": len(peoples),
                    "people": peoples,
                }
            )
            with open(os.path.join(output_dir, "{0}_keypoints.json".format(i)), "w") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

    landmarks = {
        "pose_landmarks": poseKeypoints,
        "face_landmarks": faceKeypoints,
        "left_hand_landmarks": leftHandKeypoints,
        "right_hand_landmarks": rightHandKeypoints,
    }

    return landmarks, image_masks.tolist()


def parse_args():
    parser = argparse.ArgumentParser(description="Run algorithm on real images", allow_abbrev=False)
    parser.add_argument("-render_pose", default="0", help="whether render the pose in image, 0: off, 1: on. Off will improve the speed", )
    parser.add_argument('-model_folder', default="/app/model/Openpose_model", help="path to the folder containing the pre-trained models")
    parser.add_argument('-number_people_max', default="1", help="maximum number of people to detect")
    parser.add_argument('-disable_multi_thread', default="False", help="disable multi thread")
    parser.add_argument('-maximize_positives', default="True", help="maximize positives")
    parser.add_argument('-face', default="False", help="Turn on the face detection")
    parser.add_argument('-hand', default="False", help="Turn on the hand detection")

    args = parser.parse_args()

    openpose_params = {
        "render_pose": args.render_pose,
        "model_folder": args.model_folder,
        "number_people_max": args.number_people_max,
        "disable_multi_thread": args.disable_multi_thread,
        "maximize_positives": args.maximize_positives,
        "face": args.face,
        "hand": args.hand,
    }

    return openpose_params


def init_openpose_model(params):
    params = params

    # starting OpenPose wrappers for each image, it might be slow, but avoid multi thread problem
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    return opWrapper


if __name__ == '__main__':
    openpose_params = parse_args()
    opWrapper = init_openpose_model(openpose_params)

    server.run(debug=False, host="0.0.0.0", port=8080)
