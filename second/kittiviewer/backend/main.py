"""This backend now only support lidar. camera is no longer supported.
"""

import base64
import datetime
import io as sysio
import json
import pickle
import time
from pathlib import Path

import fire
import torch
import numpy as np
import skimage
from flask import Flask, jsonify, request
from flask_cors import CORS
from google.protobuf import text_format
from skimage import io
import cv2

from second.data import kitti_common as kitti
from second.data.all_dataset import get_dataset_class
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.pytorch.train import build_network, example_convert_to_torch
from second.light_curtain.policy import get_lc_policy_class
from second.light_curtain.utils import LC_PROCESS

app = Flask("second")
CORS(app)

class SecondBackend:
    def __init__(self):
        self.root_path = None 
        self.image_idxes = None
        self.dt_annos = None
        self.dataset = None
        self.net = None
        self.device = None
        self.lc_policy = None
        self.init_lidar_num_beams = 1

class LazyStack:
    def __init__(self):
        self.STACK = []
        self.PTR = -1
    
    def __len__(self):
        return len(self.STACK)
    
    def clear(self):
        self.STACK.clear()
        self.PTR = -1
    
    def next_needs_input(self):
        """Call this and check before calling next()"""
        return len(self.STACK) == self.PTR + 1
    
    def next(self, input=None):
        # Never returns None.
        if self.next_needs_input():
            assert input is not None, "called next() without passing input when needed"
            self.STACK.append(input)
            self.PTR += 1
        else:
            assert input is None, "called next() passing input when it was not needed"
            self.PTR += 1
    
    def prev(self):
        if self.PTR > 0:
            self.PTR -= 1
    
    def curr(self):
        # Will return None when stack is empty.
        if self.PTR == -1:
            return None
        else:
            return self.STACK[self.PTR]

BACKEND = SecondBackend()

# Stack to store results of successive LC_PROCESS.
CUM_LC_STACK = LazyStack()

def error_response(msg):
    response = {}
    response["status"] = "error"
    response["message"] = "[ERROR]" + msg
    print("[ERROR]" + msg)
    return response


@app.route('/api/readinfo', methods=['POST'])
def readinfo():
    global BACKEND, CUM_LC_STACK
    CUM_LC_STACK.clear()
    instance = request.json
    root_path = Path(instance["root_path"])
    response = {"status": "normal"}
    BACKEND.root_path = root_path
    info_path = Path(instance["info_path"])
    dataset_class_name = instance["dataset_class_name"]
    BACKEND.dataset = get_dataset_class(dataset_class_name)(
                        root_path=str(root_path),
                        info_path=str(info_path)
                      )
    BACKEND.image_idxes = list(range(len(BACKEND.dataset)))
    response["image_indexes"] = BACKEND.image_idxes
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/read_detection', methods=['POST'])
def read_detection():
    global BACKEND
    instance = request.json
    det_path = Path(instance["det_path"])
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if Path(det_path).is_file():
        with open(det_path, "rb") as f:
            dt_annos = pickle.load(f)
    else:
        dt_annos = kitti.get_label_annos(det_path)
    BACKEND.dt_annos = dt_annos
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@app.route('/api/get_pointcloud', methods=['POST'])
def get_pointcloud():
    global BACKEND, CUM_LC_STACK
    CUM_LC_STACK.clear()
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    image_idx = instance["image_idx"]
    enable_int16 = instance["enable_int16"]
    
    idx = BACKEND.image_idxes.index(image_idx)
    query = {
        "lidar": {
            "idx": idx,
            "colored": True  # colored if possible!
        },
        "cam": {},
        "init_lidar": {
            "num_beams": BACKEND.init_lidar_num_beams
        }
    }
    sensor_data = BACKEND.dataset.get_sensor_data(query)

    # img_shape = image_info["image_shape"] # hw
    if 'annotations' in sensor_data["lidar"]:
        annos = sensor_data["lidar"]['annotations']
        gt_boxes = annos["boxes"].copy()
        response["locs"] = gt_boxes[:, :3].tolist()
        response["dims"] = gt_boxes[:, 3:6].tolist()
        rots = np.concatenate([np.zeros([gt_boxes.shape[0], 2], dtype=np.float32), -gt_boxes[:, 6:7]], axis=1)
        response["rots"] = rots.tolist()
        response["labels"] = annos["names"].tolist()
    # response["num_features"] = sensor_data["lidar"]["points"].shape[1]
    points = sensor_data["lidar"]["points"]
    if points.shape[1] == 6:
        response["num_features"] = 6
    else:
        response["num_features"] = 3
        points = points[:, :3]
    if enable_int16:
        int16_factor = instance["int16_factor"]
        points *= int16_factor
        points = points.astype(np.int16)
    pc_str = base64.b64encode(points.tobytes())
    response["pointcloud"] = pc_str.decode("utf-8")
    print("send response with pointcloud size {}!".format(len(pc_str)))

    init_lidar_points = sensor_data["init_lidar"]["points"]
    if enable_int16:
        int16_factor = instance["int16_factor"]
        init_lidar_points *= int16_factor
        init_lidar_points = init_lidar_points.astype(np.int16)
    pc_str = base64.b64encode(init_lidar_points.tobytes())
    response["sb_pointcloud"] = pc_str.decode("utf-8")
    print("send response single-beam lidar cloud with size {}!".format(len(pc_str)))

    # if "score" in annos:
    #     response["score"] = score.tolist()
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/get_image', methods=['POST'])
def get_image():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")    
    image_idx = instance["image_idx"]
    idx = BACKEND.image_idxes.index(image_idx)
    query = {
        "lidar": {
            "idx": idx
        },
        "cam": {}
    }
    sensor_data = BACKEND.dataset.get_sensor_data(query)
    if "cam" in sensor_data and "data" in sensor_data["cam"] and sensor_data["cam"]["data"] is not None:
        image_str = sensor_data["cam"]["data"]
        response["image_b64"] = base64.b64encode(image_str).decode("utf-8")
        response["image_b64"] = 'data:image/{};base64,'.format(sensor_data["cam"]["datatype"]) + response["image_b64"]
        print("send an image with size {}!".format(len(response["image_b64"])))
    else:
        response["image_b64"] = ""
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/build_network', methods=['POST'])
def build_network_():
    global BACKEND, CUM_LC_STACK
    CUM_LC_STACK.clear()
    instance = request.json
    cfg_path = Path(instance["config_path"])
    ckpt_path = Path(instance["checkpoint_path"])
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if not cfg_path.exists():
        return error_response("config file not exist.")
    if not ckpt_path.exists():
        return error_response("ckpt file not exist.")
    config = pipeline_pb2.TrainEvalPipelineConfig()

    with open(cfg_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_network(config.model.second).to(device).float().eval()
    net.load_state_dict(torch.load(ckpt_path))
    eval_input_cfg = config.eval_input_reader
    BACKEND.dataset = input_reader_builder.build(
        eval_input_cfg,
        config.model.second,
        training=False,
        voxel_generator=net.voxel_generator,
        target_assigner=net.target_assigner,
        net=net).dataset
    BACKEND.net = net
    BACKEND.config = config
    BACKEND.device = device
    BACKEND.lc_policy = \
        get_lc_policy_class("DPOptimizedPolicy")(BACKEND.dataset.lc_device)
    BACKEND.init_lidar_num_beams = eval_input_cfg.cum_lc_wrapper.init_lidar_num_beams
    BACKEND.sparsify_config = eval_input_cfg.cum_lc_wrapper.sparsify_return
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    print("build_network successful!")
    return response


def _fill_inference_response_with_lc_proc_res(response):
    lc_net_input = CUM_LC_STACK.curr()["prev_sensor_data"]["lidar"]["points"]  # input data for current predictions
    pred = CUM_LC_STACK.curr()["net_pred"][0]  # current predictions of bboxes
    
    box3d = pred["box3d_lidar"].detach().cpu().numpy()
    locs = box3d[:, :3]
    dims = box3d[:, 3:6]
    rots = np.concatenate([np.zeros([locs.shape[0], 2], dtype=np.float32), -box3d[:, 6:7]], axis=1)
    response["dt_locs"] = locs.tolist()
    response["dt_dims"] = dims.tolist()
    response["dt_rots"] = rots.tolist()
    response["dt_labels"] = pred["label_preds"].detach().cpu().numpy().tolist()
    response["dt_scores"] = pred["scores"].detach().cpu().numpy().tolist()

    confidence_map = CUM_LC_STACK.curr()["confidence_map"]  # current confidence map that produces current detections

    # Heatmap: confidence.
    confidence_heatmap = _create_confidence_heatmap(confidence_map)
    image_str = cv2.imencode('.png', confidence_heatmap)[1].tostring()
    response["confidenceMap_b64"] = base64.b64encode(image_str).decode("utf-8")
    response["confidenceMap_b64"] = 'data:image/{};base64,'.format('png') + response["confidenceMap_b64"]
    print("send a heatmap with size {}!".format(len(response["confidenceMap_b64"])))

    # Heatmap: entropy.
    entropy_heatmap = _create_entropy_heatmap(confidence_map)
    image_str = cv2.imencode('.png', entropy_heatmap)[1].tostring()
    response["entropyMap_b64"] = base64.b64encode(image_str).decode("utf-8")
    response["entropyMap_b64"] = 'data:image/{};base64,'.format('png') + response["entropyMap_b64"]
    print("send a heatmap with size {}!".format(len(response["entropyMap_b64"])))

    # Light curtain -- curtain that generated the latest data for the current input.
    # So, look at generated light curtain from previous timestep.
    if CUM_LC_STACK.PTR < 1:
        lc_cloud = np.zeros([0, 4], dtype=np.float32)
    else:
        lc_cloud = CUM_LC_STACK.STACK[CUM_LC_STACK.PTR - 1]["lc_cloud"]
    # lc_cloud = lc_proc_res["lc_cloud"]
    # Convert lc_cloud to string.
    instance = request.json
    # enable_int16 = instance["enable_int16"]
    enable_int16 = True
    if enable_int16:
        # int16_factor = instance["int16_factor"]
        int16_factor = 100
        lc_cloud = (int16_factor * lc_cloud).astype(np.int16)
        lc_net_input = (int16_factor * lc_net_input).astype(np.int16)
    pc_str = base64.b64encode(lc_cloud.tobytes())
    response["lc_cloud"] = pc_str.decode("utf-8")
    pc_str = base64.b64encode(lc_net_input.tobytes())
    response["lc_net_input"] = pc_str.decode("utf-8")


@app.route('/api/inference_next_by_idx', methods=['POST'])
def inference_next_by_idx():
    global BACKEND, CUM_LC_STACK
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    image_idx = instance["image_idx"]

    if CUM_LC_STACK.next_needs_input():
        lc_proc_res = CUM_LC_STACK.curr()
        if lc_proc_res is None:
            # Stack is empty. Create input for next with just single beam lidar.
            query = {
                "lidar": {
                    "idx": image_idx
                },
                "depth": {},
                "init_lidar": {
                    "num_beams": BACKEND.init_lidar_num_beams
                }
            }
            prev_sensor_data = BACKEND.dataset.get_sensor_data(query)
            init_lidar_points = prev_sensor_data["init_lidar"]["points"]  # (N, 3)
            init_lidar_points = np.hstack((init_lidar_points,
                                           np.ones([len(init_lidar_points), 1], dtype=np.float32)))  # (N, 4)
            prev_sensor_data["lidar"]["points"] = init_lidar_points
        else:
            # Use this lc_proc_res to produce input for next.
            prev_sensor_data = lc_proc_res["next_sensor_data"]

        input = LC_PROCESS(prev_sensor_data,
                           BACKEND.net,
                           BACKEND.dataset,
                           BACKEND.lc_policy,
                           BACKEND.sparsify_config)
        CUM_LC_STACK.next(input)
    else:
        CUM_LC_STACK.next()
    
    # We are now at the next lc_proc_res.
    lc_proc_res = CUM_LC_STACK.curr()

    _fill_inference_response_with_lc_proc_res(response)

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    print(f"CUM_LC_STACK PTR = {CUM_LC_STACK.PTR}.")

    return response

@app.route('/api/inference_prev_by_idx', methods=['POST'])
def inference_prev_by_idx():
    global BACKEND, CUM_LC_STACK
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    image_idx = instance["image_idx"]

    # Give a response only if Kittiviewer needs to be updated.
    if len(CUM_LC_STACK) >= 1:
        CUM_LC_STACK.prev()

        # We are now at the prev lc_proc_res.
        lc_proc_res = CUM_LC_STACK.curr()

        _fill_inference_response_with_lc_proc_res(response)

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    print(f"CMU_LC_STACK PTR = {CUM_LC_STACK.PTR}.")
    return response

def _create_confidence_heatmap(confidence_map):
    # Take the mean of confidences for the 0° and 90° anchors
    conf_scores = confidence_map[:, :, 2:]  # (Y, X, K)
    conf_scores = conf_scores.mean(axis=2)  # (Y, X)

    # Rescale between 0 and 1.
    # conf_scores = conf_scores - conf_scores.min()
    # conf_scores = conf_scores / conf_scores.max()

    heatmap = cv2.applyColorMap((conf_scores * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    return heatmap

def _create_entropy_heatmap(confidence_map):
    p = confidence_map[:, :, 2:]  # (Y, X, K)
    p = p.clip(1e-5, 1-1e-5)  # (Y, X, K)
    entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)  # (Y, X, K)
    entropy = entropy.mean(axis=2)  # (Y, X)

    heatmap = cv2.applyColorMap((entropy * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    return heatmap

def main(port=16666):
    app.run(host='127.0.0.1', threaded=True, port=port)

if __name__ == '__main__':
    fire.Fire()
