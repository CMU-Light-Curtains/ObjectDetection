import numpy as np
import math

def point_cloud_to_depth_map(points, cam_matrix, width, height, debug=False):
    """
    Note: (1) Input point cloud should be in CAMERA FRAME.
          Project points to CAMERA FRAME first using camera extrinsics.
          (2) Camera frame is as follows:
              - The focal point is (0, 0, 0).
              - +ve X and +ve Y axes go from left-to-right and top-to-bottom
                parallel the film respectively.
              - The imaging plane is placed parallel to the XY plane and 
                the +ve Z axis points towards the scene. 
    Args:
        points: (np.ndarray, np.float32, shape=(N, 3+)) points.
        cam_matrix: (np.ndarray, np.float32, shape=(3, 3)) camera intrinsics matrix.
        width: (int) number of pixels along width.
        height: (int) number of pixels along height.
    """    
    # Get projected x, y coordinates by projecting point cloud using camera
    # matrix and then dividing by z.
    proj_points = points[:, :3] @ cam_matrix.T
    proj_points = proj_points / proj_points[:, 2].reshape(-1, 1)
    xy = proj_points[:, :2]  # stands for projected x and projected y
    d = points[:, 2]  # stands for depth, shape=(N,)
    
    # xy values are now in pixels. Convert to ints and keep only those that lie on the film.
    xy = xy.astype(np.int)
    x, y = xy[:, 0], xy[:, 1]
    keep_mask = np.where((0 <= x) & (x < width) & (0 <= y) & (y < height))[0]
    if debug:
        print("POINT_CLOUT_TO_CAM: {}/{} [{}%] points lie outside field of view ({}°) and are discarded.".format(
            len(xy) - len(keep_mask), len(xy), (len(xy) - len(keep_mask)) / len(xy) * 100, fov_deg))
    x, y, d = x[keep_mask], y[keep_mask], d[keep_mask]
    
    # Sort in decreasing order of d, so that nearer points are ordered after farther points. Then, farther
    # points will be replaced by nearer points.
    sort_inds = np.argsort(d)[::-1]
    x, y, d = x[sort_inds], y[sort_inds], d[sort_inds]
    
    # Create depth map.
    depth_map = np.zeros([height, width], dtype=np.float32)
    depth_map[y, x] = d
    if debug:
        print("POINT_CLOUT_TO_CAM: depth map has {}% non-zero valus.".format((depth_map == 0).mean() * 100))
    
    return depth_map


def depth_map_to_point_cloud(depth_map, cam_matrix, debug=False):
    """
    Note: (1) Output point cloud is in CAMERA FRAME.
          Project points to VELO FRAME first using inverse of camera extrinsics.
          (2) Camera frame is as follows:
              - The focal point is (0, 0, 0).
              - +ve X and +ve Y axes go from left-to-right and top-to-bottom
                parallel the film respectively.
              - The imaging plane is placed parallel to the XY plane and 
                the +ve Z axis points towards the scene. 
    Args:
        depth_map: (np.ndarray, np.float32, shape=(H, W, 1 + K))
                   An image that contains at-least one channel for z-values.
                   It may contain K additional channels such as colors,
                   which will be appended to the point vector.
        cam_matrix: (np.ndarray, np.float32, shape=(3, 3)) camera intrinsics matrix.
    Returns:
        points: (np.ndarray, np.float32, shape=(N, 3 + K)) points.
                Axis 1: the first three channels correspond to x, y, z in the CAMERA frame.
                        The next K channels are copied over from depth_map (see description). 
    """
    H, W = depth_map.shape[:2]
    K = depth_map.shape[2] - 1

    # Invert the steps of projection:
    #   1. Compute x, y in projected coordinates.
    #   2. Inverse the homographic division by z; multiply by z.
    #   3. Multiply by inverse of cam_matrix.
    y, x = np.mgrid[0:H, 0:W] + 0.5  # x and y are both (H, W)
    z = depth_map[:, :, 0]  # (H, W)
    x, y = x * z, y * z  # (H, W)
    xyz = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    xyz = xyz @ np.linalg.inv(cam_matrix).T  # (H, W, 3)
    # Now xyz are in the camera frame.

    point_cloud = np.concatenate((xyz, depth_map[:, :, 1:]), axis=2)  # (H, W, 3 + K)
    point_cloud = point_cloud.reshape(-1, 3 + K)  # (N, 3 + K)
    return point_cloud
