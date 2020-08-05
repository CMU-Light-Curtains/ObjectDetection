import numpy as np
from functools import partial
import torch

from second.dynamic.devices.device import Device
from second.protos import pipeline_pb2
from google.protobuf import text_format
from second.builder import target_assigner_builder, voxel_builder
from second.core import box_np_ops
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.data.preprocess import prep_main
from second.pytorch.train import example_convert_to_torch


class Detector(Device):
    def __init__(self, env, lidar, light_curtain, dp_optimizer, config_file, ckpt_file, latency=72):
        super(Detector, self).__init__(env, capacity=1)
        self.latency = latency  # latency of the forward pass

        # devices the detector depends on
        self.lidar = lidar
        self.light_curtain = light_curtain
        self.dp_optimizer = dp_optimizer

        # load config
        self.config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_file, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, self.config)
        
        # create net
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.build_network().to(device).float().eval()
        self.net.load_state_dict(torch.load(ckpt_file))

        # create preprocess function
        self.preprocess_fn = self.create_preprocess_fn()
    
    def process(self):
        """
        Publishes:
            {
                "detections"    : { "dt_locs", "dt_dims", "dt_rots", "dt_labels", "dt_scores" },
                "confidence_map": (np.ndarray, dtype=float32, shape=(X, Z, 2+K)) confidence map of detector.
                                    Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                                    Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                                    Axis 2 corresponds to (x, z, c_1, ..., c_K):
                                        - x : x in camera frame.
                                        - z : z in camera frame.
                                        - c_k : kth confidence score lying in [0, 1].
            }
        """
        while True:
            # collect lidar data
            if len(self.lidar.stream) == 0:
                raise Exception("Detector: lidar stream is empty! Detector needs lidar points.")
            lidar_points = self.lidar.stream[-1].data  # (N, 3)
            lidar_points = np.hstack((lidar_points, np.ones([len(lidar_points), 1], dtype=np.float32)))  # (N, 4)

            # collect light curtain return
            if len(self.light_curtain.stream) > 0:
                lc_points = [elem.data["lc_cloud"] for elem in self.light_curtain.stream]  # list of (N, 4)
                lc_points = np.vstack(lc_points)  # (N, 4)
                lc_points = self.sparsify_lc_points(lc_points)  # (N, 4)
            else:
                lc_points = np.zeros([0, 4], dtype=np.float32)

            # combine lidar and light curtain points
            points = np.vstack([lidar_points, lc_points])  # (N, 4)

            example = self.preprocess_fn(points)
            if "anchors_mask" in example:
                example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
            
            # don't forget to pad batch idx in coordinates
            example["coordinates"] = np.pad(example["coordinates"], ((0, 0), (1, 0)),
                                            mode='constant',
                                            constant_values=0)
            
            # don't forget to add newaxis for anchors
            example["anchors"] = example["anchors"][np.newaxis, ...]

            with torch.no_grad():
                example_torch = example_convert_to_torch(example)
                pred, preds_dict = self.net(example_torch, ret_preds_dict=True)
            
            # get detections
            detections = {}
            pred = pred[0]
            box3d = pred["box3d_lidar"].detach().cpu().numpy()
            locs = box3d[:, :3]
            dims = box3d[:, 3:6]
            rots = np.concatenate([np.zeros([locs.shape[0], 2], dtype=np.float32), -box3d[:, 6:7]], axis=1)
            detections["dt_locs"] = locs.tolist()
            detections["dt_dims"] = dims.tolist()
            detections["dt_rots"] = rots.tolist()
            detections["dt_labels"] = pred["label_preds"].detach().cpu().numpy().tolist()
            detections["dt_scores"] = pred["scores"].detach().cpu().numpy().tolist()
            
            # get confidence map
            cls_preds = preds_dict['cls_preds']  # (1, 2, 200, 176, 1)
            cls_preds = cls_preds[0, :, :, :, 0]  # (2, 200, 176)
            cls_preds = torch.sigmoid(cls_preds).detach().cpu().numpy()  # (2, 200, 176)

            anchors = example["anchors"][0]  # (2 * 200 * 176, 7)
            anchors_mask = example.get("anchors_mask", None)  # (200 * 176,) dtype=np.uint8
            confidence_map = self.get_confidence_map(
                anchors, anchors_mask, 
                cls_preds, self.light_curtain.lc_device.TRANSFORMS["wTc"]
            )

            stream_data = dict(detections=detections, confidence_map=confidence_map)

            yield self.env.timeout(self.latency)  # forward pass
            self.publish(stream_data)

            # get design points
            yield self.env.process(self.dp_optimizer.service(confidence_map))
            design_pts = self.dp_optimizer.stream[-1].data  # (N, 2)

            # operate light curtain
            yield self.env.process(self.light_curtain.service(design_pts))


    
    def get_confidence_map(self, anchors, anchors_mask, cls_preds, vel2cam):
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
    
    def sparsify_lc_points(self, lc_cloud):
        """
        Args:
            lc_cloud: (np.ndarray, dtype=np.float32, shape=(N, 4)) lc cloud.
                    Axis 2 corresponds to (x, y, z, i):
                                    - x : x in velo frame.
                                    - y : y in velo frame.
                                    - z : z in velo frame.
                                    - i : intensity of LC cloud, lying in [0, 1].        
        Returns:
            lc_cloud: (same as arg) subsampled lc cloud.
        """
        sparsify_config = self.config.eval_input_reader.cum_lc_wrapper.sparsify_return

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

    def build_network(self):
        model_cfg = self.config.model.second
        voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
        bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        box_coder = box_coder_builder.build(model_cfg.box_coder)
        target_assigner_cfg = model_cfg.target_assigner
        target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)
        box_coder.custom_ndim = target_assigner._anchor_generators[0].custom_ndim
        net = second_builder.build(model_cfg, voxel_generator, target_assigner, measure_time=False)
        return net

    def create_preprocess_fn(self):
        prep_cfg = self.config.eval_input_reader.preprocess
        out_size_factor=self.get_downsample_factor()
        assert out_size_factor > 0
        preprocess_fn = partial(prep_main,
                                voxel_generator=self.net.voxel_generator,
                                target_assigner=self.net.target_assigner,
                                max_voxels=prep_cfg.max_number_of_voxels,
                                shuffle_points=prep_cfg.shuffle_points,
                                anchor_area_threshold=prep_cfg.anchor_area_threshold,
                                out_size_factor=out_size_factor,
                                multi_gpu=False,
                                calib=None)
    
        grid_size = self.net.voxel_generator.grid_size
        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]
        assert all([n != '' for n in self.net.target_assigner.classes]), \
               "you must specify class_name in anchor_generators."
        print("feature_map_size", feature_map_size)

        ret = self.net.target_assigner.generate_anchors(feature_map_size)
        class_names = self.net.target_assigner.classes
        anchors_dict = self.net.target_assigner.generate_anchors_dict(feature_map_size)
        anchors_list = []
        for k, v in anchors_dict.items():
            anchors_list.append(v["anchors"])
        
        # anchors = ret["anchors"]
        anchors = np.concatenate(anchors_list, axis=0)
        anchors = anchors.reshape([-1, self.net.target_assigner.box_ndim])
        assert np.allclose(anchors, ret["anchors"].reshape(-1, self.net.target_assigner.box_ndim))
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
        anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
            "anchors_dict": anchors_dict,
        }
        preprocess_fn = partial(preprocess_fn, anchor_cache=anchor_cache)
        return preprocess_fn

    def get_downsample_factor(self):
        model_config = self.config.model.second
        downsample_factor = np.prod(model_config.rpn.layer_strides)
        if len(model_config.rpn.upsample_strides) > 0:
            downsample_factor /= model_config.rpn.upsample_strides[-1]
        downsample_factor *= model_config.middle_feature_extractor.downsample_factor
        downsample_factor = np.round(downsample_factor).astype(np.int64)
        assert downsample_factor > 0
        return downsample_factor