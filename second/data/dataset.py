import pathlib
import pickle
import time
from functools import partial

import numpy as np

from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti

REGISTERED_DATASET_CLASSES = {}

def register_dataset(cls, name=None):
    global REGISTERED_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exist class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls

def get_dataset_class(name):
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


class Dataset(object):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    NumPointFeatures = -1
    def __getitem__(self, index):
        """This function is used for preprocess.
        you need to create a input dict in this function for network inference.
        format: {
            anchors
            voxels
            num_points
            coordinates
            if training:
                labels
                reg_targets
            [optional]anchors_mask, slow in SECOND v1.5, don't use this.
            [optional]metadata, in kitti, image index is saved in metadata
        }
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_sensor_data(self, query):
        """Dataset must provide a unified function to get data.
        Args:
            query: int or dict. this param must support int for training.
                if dict, should have this format (no example yet): 
                {
                    sensor_name: {
                        sensor_meta
                    }
                }
                if int, will return all sensor data. 
                (TODO: how to deal with unsynchronized data?)
        Returns:
            sensor_data: dict. 
            if query is int (return all), return a dict with all sensors: 
            {
                sensor_name: sensor_data
                ...
                metadata: ... (for kitti, contains image_idx)
            }
            
            if sensor is lidar (all lidar point cloud must be concatenated to one array): 
            e.g. If your dataset have two lidar sensor, you need to return a single dict:
            {
                "lidar": {
                    "points": ...
                    ...
                }
            }
            sensor_data: {
                points: [N, 3+]
                [optional]annotations: {
                    "boxes": [N, 7] locs, dims, yaw, in lidar coord system. must tested
                        in provided visualization tools such as second.utils.simplevis
                        or web tool.
                    "names": array of string.
                }
            }
            if sensor is camera (not used yet):
            sensor_data: {
                data: image string (array is too large)
                [optional]annotations: {
                    "boxes": [N, 4] 2d bbox
                    "names": array of string.
                }
            }
            metadata: {
                # dataset-specific information.
                # for kitti, must have image_idx for label file generation.
                image_idx: ...
            }
            [optional]calib # only used for kitti
        """
        raise NotImplementedError

    def evaluation(self, dt_annos, output_dir):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError

    def create_sim_lidar_mask(self, cam_w, cam_h, cam_matrix, beam_angles, ares, debug=False):
        """
        Args:
            cam_w: (int) width of depth camera.
            cam_h: (int) height of depth camera.
            cam_matrix: (np.ndarray, np.float32, shape=(3, 3)) camera intrinsics matrix.
            beam_angles: (np.ndarray, np.float32, shape=(P,)) vertical beam angles in degrees.
                            eg. a 64-beam LiDAR will have 64 angles.
            ares: (float) angular resolution in degrees.
        Returns:
            mask: (np.ndarray, np.uint8, shape=(H, W)) mask of depth map that contains LiDAR points.
        """
        # Compute extreme θ values of the camera.
        # Coordinates of the camera extremeties, in pixels.
        xy_p = np.array([[0.0,   cam_h / 2],
                        [cam_w, cam_h / 2]], dtype=np.float32)  # (2, 2)    
        xy1_p = np.hstack([xy_p, np.ones([2, 1], dtype=np.float32)])  # (2, 3)
        # First convert pixel coordinates to camera coordinates (inverse of camera matrix).
        xy1_c = xy1_p @ np.linalg.inv(cam_matrix).T
        # Since we set y_p to height / 2, y_c should be 0.
        # The depth value of all these points is 1.
        assert np.all(np.abs(xy1_c[:, 1]) < 1e-6)
        xz_c = xy1_c[:, [0, 2]]  # (2, 2): all the z values are 1.
        θs = np.rad2deg(np.arctan2(xz_c[:, 1], xz_c[:, 0]))  # (2,), in degrees
        cam_θ_max, cam_θ_min = θs
        if debug:
            print(f"SimLiDAR: cam_θ_min: {cam_θ_min:.2f}")
            print(f"SimLiDAR: cam_θ_max: {cam_θ_max:.2f}")

        # Grid of lidar rays in terms of φ and θ.
        φs = beam_angles  # (P,)
        θs = np.arange(cam_θ_min, cam_θ_max, ares)  # (T)
        φ, θ = np.meshgrid(φs, θs)  # (T, P)
        φ, θ = φ.ravel(), θ.ravel()  # φ and θ are both (R=T*P,)

        # Cartesian coordinates of LiDAR points in camera frame.
        φ_rad, θ_rad = np.deg2rad(φ), np.deg2rad(θ)
        y = np.sin(φ_rad)  # (R,)
        x = np.cos(φ_rad) * np.cos(θ_rad)  # (R,)
        z = np.cos(φ_rad) * np.sin(θ_rad)  # (R,)

        # Convert cartesian coordinates to pixel coordinates.
        xyz_c = np.hstack([e.reshape(-1, 1) for e in [x, y, z]])  # (R, 3)

        xyz_p = xyz_c @ cam_matrix.T  # (R, 3)
        xyz_p = xyz_p / xyz_p[:, 2].reshape(-1, 1)
        x_p, y_p = xyz_p[:, 0], xyz_p[:, 1]  # x_p and y_p are both (R,)

        # Convert pixel coordinates in floats to ints.
        x_p, y_p = x_p.astype(np.int), y_p.astype(np.int)  # # x_p and y_p are both (R,)
        keep = (x_p >= 0) & (y_p >= 0) & (x_p < cam_w) & (y_p < cam_h)  # (R,)
        if debug:
            print(f"SimLiDAR: {100 * keep.mean():.1f}% of LiDAR rays are within the depth map")
        x_p, y_p = x_p[keep], y_p[keep]

        mask = np.zeros([cam_h, cam_w], dtype=np.uint8)
        mask[y_p, x_p] = 1
        if debug:
            print(f"SimLiDAR: {100 * mask.mean():.1f}% of the depth map pixels are LiDAR rays")
        return mask
