import argparse
import os
import sys

import cv2 as cv
import numpy as np
import torch

import evaluation_util
from datamodule import Normalization


def to36M(bones, body_parts):
    H36M_JOINTS_17 = [
        "Hip",
        "RHip",
        "RKnee",
        "RFoot",
        "LHip",
        "LKnee",
        "LFoot",
        "Spine",
        "Thorax",
        "Neck/Nose",
        "Head",
        "LShoulder",
        "LElbow",
        "LWrist",
        "RShoulder",
        "RElbow",
        "RWrist",
    ]

    adjusted_bones = []
    for name in H36M_JOINTS_17:
        if not name in body_parts:
            if name == "Hip":
                adjusted_bones.append(
                    (bones[body_parts["RHip"]] + bones[body_parts["LHip"]]) / 2
                )
            elif name == "RFoot":
                adjusted_bones.append(bones[body_parts["RAnkle"]])
            elif name == "LFoot":
                adjusted_bones.append(bones[body_parts["LAnkle"]])
            elif name == "Spine":
                adjusted_bones.append(
                    (
                        bones[body_parts["RHip"]]
                        + bones[body_parts["LHip"]]
                        + bones[body_parts["RShoulder"]]
                        + bones[body_parts["LShoulder"]]
                    )
                    / 4
                )
            elif name == "Thorax":
                adjusted_bones.append(
                    (+bones[body_parts["RShoulder"]] + bones[body_parts["LShoulder"]])
                    / 2
                )
            elif name == "Head":
                thorax = (
                    +bones[body_parts["RShoulder"]] + bones[body_parts["LShoulder"]]
                ) / 2
                adjusted_bones.append(thorax + (bones[body_parts["Nose"]] - thorax) * 2)
            elif name == "Neck/Nose":
                adjusted_bones.append(bones[body_parts["Nose"]])
            else:
                raise Exception(name)
        else:
            adjusted_bones.append(bones[body_parts[name]])

    return adjusted_bones


def parts(args):
    if args.dataset == "COCO":
        BODY_PARTS = {
            "Nose": 0,
            "Neck": 1,
            "RShoulder": 2,
            "RElbow": 3,
            "RWrist": 4,
            "LShoulder": 5,
            "LElbow": 6,
            "LWrist": 7,
            "RHip": 8,
            "RKnee": 9,
            "RAnkle": 10,
            "LHip": 11,
            "LKnee": 12,
            "LAnkle": 13,
            "REye": 14,
            "LEye": 15,
            "REar": 16,
            "LEar": 17,
            "Background": 18,
        }

        POSE_PAIRS = [
            ["Neck", "RShoulder"],
            ["Neck", "LShoulder"],
            ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"],
            ["LShoulder", "LElbow"],
            ["LElbow", "LWrist"],
            ["Neck", "RHip"],
            ["RHip", "RKnee"],
            ["RKnee", "RAnkle"],
            ["Neck", "LHip"],
            ["LHip", "LKnee"],
            ["LKnee", "LAnkle"],
            ["Neck", "Nose"],
            ["Nose", "REye"],
            ["REye", "REar"],
            ["Nose", "LEye"],
            ["LEye", "LEar"],
        ]
    else:
        assert args.dataset == "MPI"
        BODY_PARTS = {
            "Head": 0,
            "Neck": 1,
            "RShoulder": 2,
            "RElbow": 3,
            "RWrist": 4,
            "LShoulder": 5,
            "LElbow": 6,
            "LWrist": 7,
            "RHip": 8,
            "RKnee": 9,
            "RAnkle": 10,
            "LHip": 11,
            "LKnee": 12,
            "LAnkle": 13,
            "Chest": 14,
            "Background": 15,
        }

        POSE_PAIRS = [
            ["Head", "Neck"],
            ["Neck", "RShoulder"],
            ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"],
            ["Neck", "LShoulder"],
            ["LShoulder", "LElbow"],
            ["LElbow", "LWrist"],
            ["Neck", "Chest"],
            ["Chest", "RHip"],
            ["RHip", "RKnee"],
            ["RKnee", "RAnkle"],
            ["Chest", "LHip"],
            ["LHip", "LKnee"],
            ["LKnee", "LAnkle"],
        ]
    return BODY_PARTS, POSE_PAIRS


class OpenPose(object):
    """
    This implementation is based on https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py
    """

    def __init__(self, args):
        self.net = cv.dnn.readNetFromCaffe(args.proto2d, args.model2d)
        # if args.inf_engine:
        #     self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)

    def predict(self, args, frame):

        inWidth = args.width
        inHeight = args.height

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        inp = cv.dnn.blobFromImage(
            frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False
        )
        self.net.setInput(inp)
        out = self.net.forward()

        BODY_PARTS, POSE_PAIRS = parts(args)

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            points.append((x, y) if conf > args.thr else None)
        return points


def create_pose(model, points):
    x = points[:, 0::2]
    y = points[:, 1::2]
    points = torch.from_numpy(points).float()
    with torch.no_grad():
        z_pred = model(points).view(17, -1)

    pose = np.stack((x, y, z_pred), axis=-1)
    pose = np.reshape(pose, (len(points), -1))

    return pose


def main(args):
    model = evaluation_util.load_model(args.ckpt)
    cap = cv.VideoCapture(args.input if args.input else 0)

    hasFrame, frame = cap.read()
    if not hasFrame:
        exit(0)
    points = OpenPose(args).predict(args, frame)
    points = [vec for vec in points]
    points = [np.array(vec) for vec in points]
    BODY_PARTS, POSE_PAIRS = parts(args)
    points = to36M(points, BODY_PARTS)
    # points = np.reshape(points, [1, -1]).astype("f")
    # NOTE: eliminate untargetted joint
    points = [point if point.shape != () else (0.0, 0.0) for point in points]
    points = np.stack(points)
    points_norm = Normalization.normalize_2d(points).reshape(17, 2)
    pose = create_pose(model, points_norm)

    out_directory = "demo_out"
    os.makedirs(out_directory, exist_ok=True)
    # out_img = evaluation_util.create_img(points[0], frame)
    out_img = evaluation_util.create_img(points, frame)
    cv.imwrite(os.path.join(out_directory, "openpose_detect.jpg"), out_img)
    deg = 15
    for d in range(0, 360 + deg, deg):
        img = evaluation_util.create_projection_img(pose, np.pi * d / 180.0)
        cv.imwrite(os.path.join(out_directory, "rot_{:03d}_degree.png".format(d)), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="./test/input_sample.png",
        help="Path to image or video. Skip to capture frames from camera",
    )
    parser.add_argument(
        "--proto2d",
        default="./test/openpose_pose_coco.prototxt",
        help="Path to .prototxt of 2d pose estimation model",
    )
    parser.add_argument(
        "--model2d",
        default="./test/pose_iter_440000.caffemodel",
        help="Path to .caffemodel of 2d pose estimation model",
    )
    parser.add_argument(
        "--thr", default=0.1, type=float, help="Threshold value for pose parts heat map"
    )
    parser.add_argument(
        "--width", default=368, type=int, help="Resize input to specific width."
    )
    parser.add_argument(
        "--height", default=368, type=int, help="Resize input to specific height."
    )
    parser.add_argument("--dataset", type=str, default="COCO")
    parser.add_argument(
        "--ckpt", type=str, default="./test/checkpoints/pretrained_model.ckpt"
    )

    args = parser.parse_args()
    main(args)
