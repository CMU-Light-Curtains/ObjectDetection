import base64
import cv2
import json
import numpy as np
from flask import Response
import time
import functools


def vizstream(app, stream, astype):
    if astype == 'scene_cloud':
        url = '/api/stream_scene_cloud'
        data2msg = data2msg_scene_cloud
    elif astype == 'camera_image':
        url = '/api/stream_camera_image'
        data2msg = data2msg_camera_image
    elif astype == 'lidar_cloud':
        url = '/api/stream_lidar_cloud'
        data2msg = data2msg_lidar_cloud
    elif astype == 'lc_cloud':
        url = '/api/stream_lc_cloud'
        data2msg = data2msg_lc_cloud
    elif astype == 'dt_boxes':
        url = '/api/stream_dt_boxes'
        data2msg = data2msg_dt_boxes
    elif astype == 'entropy_map':
        url = '/api/stream_entropy_map'
        data2msg = data2msg_entropy_map
    else:
        raise Exception(f'astype={astype} not valid')

    def generator():
        sent_timestamp = None
        while True:
            if len(stream) == 0:
                pass
            elif sent_timestamp != stream[-1].timestamp:
                sent_timestamp = stream[-1].timestamp
                
                data = stream[-1].data
                msg = data2msg(data)
                yield f"data:{msg}\n\n"

            time.sleep(0.010)  # 10ms
    
    @app.route(url, methods=['GET', 'POST'])
    @functools.wraps(data2msg)
    def route_fn():
        return Response(generator(), mimetype="text/event-stream")

########################################################################################################################
# region data2msg functions
########################################################################################################################

def data2msg_scene_cloud(data, enable_int16=True, int16_factor=100):
    points = data["points"]  # (N, 6)
    if enable_int16:
        points = points * int16_factor
        points = points.astype(np.int16)

    points = points[::3, :]
    pc_str = base64.b64encode(points.tobytes()).decode("utf-8")
    return pc_str


def data2msg_camera_image(data):        
    image_str = data["cam"]["image_str"]
    image_dtype = data["cam"]["datatype"]
    image_b64 = base64.b64encode(image_str).decode("utf-8")
    image_b64 = f"data:image/{image_dtype};base64,{image_b64}"
    return image_b64


def data2msg_lidar_cloud(data, enable_int16=True, int16_factor=100):
    points = data  # (N, 3)
    if enable_int16:
        points = points * int16_factor
        points = points.astype(np.int16)

    pc_str = base64.b64encode(points.tobytes()).decode("utf-8")
    return pc_str


def data2msg_lc_cloud(data, enable_int16=True, int16_factor=100):    
    points = data["lc_cloud"]  # (N, 4)
    if enable_int16:
        points = points * int16_factor
        points = points.astype(np.int16)

    pc_str = base64.b64encode(points.tobytes()).decode("utf-8")
    return pc_str


def data2msg_dt_boxes(data):
    dt_boxes = data["detections"]
    json_str = json.dumps(dt_boxes)
    return json_str


def data2msg_entropy_map(data):
    confidence_map = data["confidence_map"]
    entropy_heatmap = _create_entropy_heatmap(confidence_map)
    
    image_str = cv2.imencode('.png', entropy_heatmap)[1].tostring()
    image_b64 = base64.b64encode(image_str).decode("utf-8")
    image_b64 = f"data:image/png;base64,{image_b64}"
    return image_b64


#endregion
########################################################################################################################
#region Helper functions
########################################################################################################################

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

#endregion
########################################################################################################################
