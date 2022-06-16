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
    if debug_path is not None and cfg.Debug:
        # create debug folder
        # output_dir = os.path.join(debug_path, "openpose_keypoints")
        output_dir = debug_path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process image
    poseKeypoints = []
    faceKeypoints = []
    image_masks = np.ones(shape=images.shape[0], dtype=np.bool)

    for i in range(images.shape[0]):
        datum = op.Datum()
        datum.cvInputData = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        if datum.poseKeypoints is None or datum.poseKeypoints.size <= 1:  # if we cannot detect any pose from this image, then discard it
            image_masks[i] = False
            poseKeypoint = np.zeros(shape=(25, 3), dtype=np.float32)
        else:
            poseKeypoint = datum.poseKeypoints[0]
        poseKeypoints.append(poseKeypoint.tolist())

        if datum.faceKeypoints is None or datum.faceKeypoints.size <= 1:
            faceKeypoint = np.zeros(shape=(70, 3), dtype=np.float32)
        else:
            faceKeypoint = datum.faceKeypoints[0]

        faceKeypoints.append(faceKeypoint.tolist())

        if output_dir is not None:
            cv2.imwrite(os.path.join(output_dir, "{0}.jpg".format(i)), datum.cvOutputData)
            json_data = OrderedDict(
                {
                    "version": 1.7,
                    "people": [
                        OrderedDict(
                            {
                                "person_id": [-1],
                                "pose_keypoints_2d": poseKeypoint.tolist(),
                                "face_keypoints_2d": faceKeypoint.tolist(),
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
            with open(os.path.join(output_dir, "{0}_keypoints.json".format(i)), "w") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

    landmarks = {
        "pose_landmarks": poseKeypoints,
        "face_landmarks": faceKeypoints,
    }

    return landmarks, image_masks.tolist()


def parse_args():
    parser = argparse.ArgumentParser(description="Run algorithm on real images", allow_abbrev=False)
    parser.add_argument("-config", default="./config.yaml", help="The configuration to run openpose")

    args = parser.parse_args()
    return args


def read_cofigure(path):
    cfg = SimpleNamespace()

    with open(path) as file:
        documents = yaml.safe_load(file)

        # openpose
        cfg.openpose = SimpleNamespace()
        cfg.openpose.params = documents["openpose"]

        # others
        if documents["Debug"]["mode"] == "on":
            cfg.Debug = True
            cfg.openpose.params["render_pose"] = 1
        else:
            cfg.Debug = False
            cfg.openpose.params["render_pose"] = 0

        return cfg


def init_openpose_model(cfg):
    params = cfg.openpose.params

    # starting OpenPose wrappers for each image, it might be slow, but avoid multi thread problem
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    return opWrapper


if __name__ == '__main__':
    args = parse_args()

    cfg = read_cofigure(args.config)
    opWrapper = init_openpose_model(cfg)

    server.run(debug=False, host="0.0.0.0", port=8080)
