import numpy as np
import torch

from second.pytorch.train import example_convert_to_torch

def LC_PROCESS(prev_sensor_data, net, dataset, policy, sparsify_config):
    """
    Processes prev_sensor_data and feeds it to the net, places light curtain using
    generated confidence scores, gets light curtain return, and adds it to a new
    sensor data.
    
    While processing the sensor_data, only the dataset._prep_main_func is used.
    No calls to _prep_data_aug and _prep_targets are made.

    sensor_data["lidar"]["points"] will be considered the main input to the model.
    prev_sensor_data["lidar"]["points"] will be used as the input cloud, and the output
    cloud will be saved into sensor_data["lidar"]["points"].
    This only handles 4-dimensional points, of the form (x, y, z, intensity).

    Args:
    *   prev_sensor_data: a dict containing the following items
            {
                "lidar": {
                    "type": "lidar",
                    "points": ...
                },
                "calib": {
                    ...
                },
                "depth": {
                    "type": "depth_map",
                    "image": ...
                },
                "init_lidar": {
                    "num_beams": ...,
                    "points": ...
                }

            }
            sensor_data["lidar"]["points"] constitutes the main input the network.
            It could be initialized by sensor_data["init_lidar"] for example.
    
    *   net: (VoxelNet)

    *   dataset: (Dataset) dataset.

    *   policy: (Policy) one of the light curtain policies in second.light_curtain.policy.

    *   sparsify_config: (config) options that are used to subsample a point cloud return.
    
    Returns:
    *   dictionary of all information generated during the LC process
        {
            "prev_sensor_data": previous sensor data,
            "next_sensor_data": the new sensor data dict,
            "lc_image": lc_image returned by pylc API,
            "lc_cloud": lc_image processed into a point cloud; this is the main return,
            "net_pred": pred of the network,
            "net_preds_dict": preds_dict of the network, 
            "confidence_map": network predictions converted to a confidence_map
        }
    *   sensor_data: the same sensor_data, but with sensor_data["lidar"]["points"] replaced by
                     the new cumulative point cloud.

    """
    net_was_training = net.training
    net.eval()

    points = prev_sensor_data["lidar"]["points"]
    calib = prev_sensor_data["calib"]

    example = dataset._prep_func_main(points, calib)
    if "image_idx" in prev_sensor_data["metadata"]:
        example["metadata"] = prev_sensor_data["metadata"]

    if "anchors_mask" in example:
        example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
    
    # don't forget to pad batch idx in coordinates
    example["coordinates"] = np.pad(
        example["coordinates"], ((0, 0), (1, 0)),
        mode='constant',
        constant_values=0)
    
    # don't forget to add newaxis for anchors
    example["anchors"] = example["anchors"][np.newaxis, ...]

    with torch.no_grad():
        example_torch = example_convert_to_torch(example)
        pred, preds_dict = net(example_torch, ret_preds_dict=True)

        # Creating confidence map.
        cls_preds = preds_dict['cls_preds']  # shape=(1, 2, 200, 176, 1)
        cls_preds = cls_preds[0, :, :, :, 0]  # shape=(2, 200, 176)
        cls_preds = torch.sigmoid(cls_preds).detach().cpu().numpy()  # shape=(2, 200, 176)
    
    anchors = example["anchors"][0]  # (2 * 200 * 176, 7)
    anchors_mask = example.get("anchors_mask", None)  # (200 * 176,) dtype=np.uint8
    confidence_map = _get_confidence_map(anchors, anchors_mask, cls_preds, dataset.lc_device.TRANSFORMS["wTc"])

    # Light curtain point cloud.
    depth_image = prev_sensor_data["depth"]["image"]
    # Design points should be in the camera frame.
    design_pts = policy.get_design_points(confidence_map)
    lc_image = dataset.lc_device.get_return(depth_image, design_pts)
    lc_cloud = lc_image.reshape(-1, 4)  # (N, 4)
    # Remove points which are NaNs.
    non_nan_mask = np.all(np.isfinite(lc_cloud), axis=1)
    lc_cloud = lc_cloud[non_nan_mask]  # (N, 4)
    # Convert lc_cloud to velo frame.
    lc_cloud_xyz1 = np.hstack((lc_cloud[:, :3], np.ones([len(lc_cloud), 1], dtype=np.float32)))
    lc_cloud_xyz1 = lc_cloud_xyz1 @ dataset.lc_device.TRANSFORMS["cTw"].T
    lc_cloud[:, :3] = lc_cloud_xyz1[:, :3]  # (N, 4)
    # Rescale LC return to [0, 1].
    lc_cloud[:, 3] /= 255.

    lc_pts_added = lc_cloud
    if sparsify_config is not None:
        lc_pts_added = sparsify_lc_return(lc_pts_added, sparsify_config)

    next_sensor_data = prev_sensor_data.copy()
    next_sensor_data["lidar"] = prev_sensor_data["lidar"].copy()
    next_sensor_data["lidar"]["points"] = np.vstack((next_sensor_data["lidar"]["points"],
                                                    lc_pts_added))

    # Reset training state of network.
    net.train(net_was_training)

    return {
        "prev_sensor_data": prev_sensor_data,
        "next_sensor_data": next_sensor_data,
        "lc_image": lc_image,
        "lc_cloud": lc_cloud,
        "net_pred": pred,
        "net_preds_dict": preds_dict, 
        "confidence_map": confidence_map
    }


def sparsify_lc_return(lc_cloud, sparsify_config):
    """
    Args:
        lc_cloud: (np.ndarray, dtype=np.float32, shape=(N, 4)) lc cloud.
                  Axis 2 corresponds to (x, y, z, i):
                                - x : x in velo frame.
                                - y : y in velo frame.
                                - z : z in velo frame.
                                - i : intensity of LC cloud, lying in [0, 1].
        sparsify_config: (config) options that are used to subsample a point cloud return.
    
    Returns:
        lc_cloud: (same as above) subsampled lc cloud.
    """
    # STEP 1: Remove points with large height.
    z = lc_cloud[:, 2]
    keep = (z <= sparsify_config.max_height)
    lc_cloud = lc_cloud[keep]

    # STEP 2: Split points according to high and low intensities.
    i = lc_cloud[:, 3]
    lc_cloud_pos = lc_cloud[i >= sparsify_config.pos_intensity_thresh / 255.]

    # STEP 3: Subsample negative points.
    if sparsify_config.neg_subsampling_rate > 0.0:
        lc_cloud_neg = lc_cloud[i < sparsify_config.pos_intensity_thresh / 255.]
        num_neg_pts_old = len(lc_cloud_neg)
        num_neg_pts_new = int(sparsify_config.neg_subsampling_rate * num_neg_pts_old)
        keep_inds = np.random.choice(num_neg_pts_old, num_neg_pts_new, replace=False)
        lc_cloud_neg = lc_cloud_neg[keep_inds]
    else:
        lc_cloud_neg = np.zeros((0, 4), dtype=np.float32)  # (0, 4)

    lc_cloud = np.vstack([lc_cloud_pos, lc_cloud_neg])
    return lc_cloud


def _get_confidence_map(anchors, anchors_mask, cls_preds, vel2cam):
    """Creates confidence map (definition below)
    Note: input (x, y, z) are in velo frame, and input matrices are ordered according to
    increasing velo Y and increasing velo X.
    
    Args:
        anchors: (np.ndarray, dtype=np.float32, shape=(K * Y * X, 7)) anchor boxes.
                 Axis 2 corresponds to (velo x, velo y, velo z, w, b, h, orientation)
                 K is the number of anchor boxes per location. Y and X are the dimensions sizes along
                 velo y and velo x.
        anchors_mask: (np.ndarray, dtype=np.uint8, shape=(K * Y * X)) anchor mask.
                      The anchors which are masked out are ignored while computing the networks loss
                      and their predicted bounding boxes if any are also ignored at test time.
                      We will set the confidence to 0 for these anchors.
                      - K is the number of anchor boxes per location.
                      - Y and X are the dimensions sizes along velo y and velo x.
        cls_preds: (np.ndarray, dtype=np.float32, shape=(K, Y, X)) confidence scores.
    Returns:
        confidence_map: (np.ndarray, dtype=float32, shape=(X, Z, 2+K)) confidence map of detector.
                        Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                        Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                        Axis 2 corresponds to (x, z, c_1, ..., c_K):
                            - x : x in camera frame.
                            - z : z in camera frame.
                            - c_k : kth confidence score lying in [0, 1].          
    """
    K, Y, X = cls_preds.shape
    
    cls_preds = cls_preds.transpose(1, 2, 0)  # (Y, X, K)
    
    anchors = anchors.reshape(K, Y, X, 7)  # (K, Y, X, 7)
    
    if anchors_mask is not None:
        anchors_mask = anchors_mask.reshape(K, Y, X)  # (K, Y, X)

        # We will mask out all those anchor locations where any of the anchors is not in the mask.
        anchors_mask = np.all(anchors_mask, axis=0)  # (Y, X)

        # Set confidence for locations outside anchors_mask to 0.
        y_out_inds, x_out_inds = np.where(np.logical_not(anchors_mask))
        cls_preds[y_out_inds, x_out_inds] = 0

    # The assumption is that all anchors at a location have the same x, y, z.
    for k in range(1, K):
        assert np.all(anchors[0, :, :, :3] == anchors[k, :, :, :3])
    anchors = anchors[0, :, :, :3]  # (Y, X, 3)

    # Convert x, y, z from velo frame to cam frame.
    anchor_ones = np.concatenate([
        anchors,
        np.ones([anchors.shape[0], anchors.shape[1], 1], dtype=np.float32)
    ], axis=2)  # (Y, X, 4)
    anchors = (anchor_ones @ vel2cam.T)[:, :, [0, 2]]  # (Y, X, 2 -> cam x, cam z)

    confidence_map = np.concatenate((anchors, cls_preds), axis=2)  # (Y, X, 2+K)
    # Note: axis 0 and axis 1 correspond to increasing velo Y and increasing velo X.
    # But, for confidence map, we need decreasing velo Y (increasing cam X)
    # and increasing velo X (increasing cam Z).
    confidence_map = np.flip(confidence_map, axis=0)
    
    return confidence_map
