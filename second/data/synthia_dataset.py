import io
import os
from pathlib import Path
import pickle
import time
from functools import partial
from PIL import Image
import fire

import numpy as np

from second.core import box_np_ops
from second.core import preprocess as prep
from second.core import camera_transforms as cam_transforms
from second.data import synthia_common as synthia
from second.data import preprocess as data_prep
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.data.dataset import Dataset, register_dataset
from second.utils.progress_bar import progress_bar_iter as prog_bar

import pylc

@register_dataset
class SynthiaDataset(Dataset):
    NumPointFeatures = 4  # (x, y, z, all_ones), but colored_lidar is available

    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 prep_func_data_aug=None,
                 prep_func_main=None,
                 prep_func_add_targets=None,
                 num_point_features=None):
        assert info_path is not None
        root_path = root_path.replace("$DATADIR", os.environ["DATADIR"])
        info_path = info_path.replace("$DATADIR", os.environ["DATADIR"])
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        self._root_path = Path(root_path)
        self._synthia_infos = infos

        print("remain number of infos:", len(self._synthia_infos))
        self._class_names = class_names
        self._prep_func_data_aug = prep_func_data_aug
        self._prep_func_main = prep_func_main
        self._prep_func_add_targets = prep_func_add_targets

        # Attach light curtain device.
        # Camera and Laser frames are relative to KITTI Lidar frame.
        self.lc_device = pylc.LCDevice(
            CAMERA_PARAMS={
                'width': 640,
                'height': 480,
                'fov': 39.32012056540195,
                'matrix': np.array([[895.6921997070312, 0.0              , 320.0],
                                    [0.0              , 895.6921997070312, 240.0],
                                    [0.0              , 0.0              , 1.0  ]], dtype=np.float32),
                'distortion': [0, 0, 0, 0, 0]
            },
            LASER_PARAMS={
                'y': -3.0  # place laser 3m to the right of camera
            }
            # LASER_PARAMS={
            #     'y': -0.3,  # place laser 30cm to the right of camera
            #     'divergence': (0.11/2.) / 10 # reduce default divergence by a factor of 10
            # }
        )

        self._MAX_RANGE = 80.0  # we will only consider points inside this range.
        self._SB_LIDAR_PARAMS = {
            "center": -0.3, # relative to camera
            "thickness": 0.05,  # in meters
            "angular_res": 0.4  # in degrees
        }

        # Multi-beam LiDAR mask.
        self.mb_lidar_mask = None
    
    def create_mb_lidar_mask(self, num_beams):
        vfov = 26.9 / 64 * num_beams  # degrees -- 64 beams will have a vfov of 26.9
        ares = 0.08  # degrees
        beam_angles = np.linspace(-vfov / 2, vfov / 2, num_beams)
        lidar_2d_mask = self.create_sim_lidar_mask(1242, 375,
                                                   self.lc_device.CAMERA_PARAMS['matrix'],
                                                   beam_angles=beam_angles,
                                                   ares=ares,
                                                   debug=True)  # (H, W) dtype=np.uint8 (0 or 1)
        self.mb_lidar_mask = {
                                "num_beams": num_beams,
                                "1d_mask"  : np.where(lidar_2d_mask.ravel())[0]
                             }

    def __len__(self):
        return len(self._synthia_infos)

    def convert_detection_to_kitti_annos(self, detection):
        class_names = self._class_names
        det_image_idxes = [det["metadata"]["image_idx"] for det in detection]
        gt_image_idxes = [
            info["image"]["image_idx"] for info in self._synthia_infos
        ]
        annos = []
        for i in range(len(detection)):
            det_idx = det_image_idxes[i]
            det = detection[i]
            # info = self._kitti_infos[gt_image_idxes.index(det_idx)]
            info = self._synthia_infos[i]
            calib = info["calib"]
            rect = calib["R0_rect"]
            Trv2c = calib["Tr_velo_to_cam"]
            P2 = calib["P2"]
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            if final_box_preds.shape[0] != 0:
                final_box_preds[:, 2] -= final_box_preds[:, 5] / 2
                box3d_camera = box_np_ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                locs = box3d_camera[:, :3]
                dims = box3d_camera[:, 3:6]
                angles = box3d_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_np_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = box_np_ops.project_to_image(
                    box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = np.min(box_corners_in_image, axis=1)
                maxxy = np.max(box_corners_in_image, axis=1)
                bbox = np.concatenate([minxy, maxxy], axis=1)
            anno = synthia.get_start_result_anno()
            num_example = 0
            box3d_lidar = final_box_preds
            for j in range(box3d_lidar.shape[0]):
                image_shape = info["image"]["image_shape"]
                if bbox[j, 0] > image_shape[1] or bbox[j, 1] > image_shape[0]:
                    continue
                if bbox[j, 2] < 0 or bbox[j, 3] < 0:
                    continue
                bbox[j, 2:] = np.minimum(bbox[j, 2:], image_shape[::-1])
                bbox[j, :2] = np.maximum(bbox[j, :2], [0, 0])
                anno["bbox"].append(bbox[j])
                # convert center format to kitti format
                # box3d_lidar[j, 2] -= box3d_lidar[j, 5] / 2
                anno["alpha"].append(
                    -np.arctan2(-box3d_lidar[j, 1], box3d_lidar[j, 0]) +
                    box3d_camera[j, 6])
                anno["dimensions"].append(box3d_camera[j, 3:6])
                anno["location"].append(box3d_camera[j, :3])
                anno["rotation_y"].append(box3d_camera[j, 6])

                anno["name"].append(class_names[int(label_preds[j])])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores[j])

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(synthia.empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["metadata"] = det["metadata"]
        return annos

    def evaluation(self, detections, output_dir):
        """
        detection
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        If you want to eval by my KITTI eval function, you must 
        provide the correct format annotations.
        ground_truth_annotations format:
        {
            bbox: [N, 4], if you fill fake data, MUST HAVE >25 HEIGHT!!!!!!
            alpha: [N], you can use -10 to ignore it.
            occluded: [N], you can use zero.
            truncated: [N], you can use zero.
            name: [N]
            location: [N, 3] center of 3d box.
            dimensions: [N, 3] dim of 3d box.
            rotation_y: [N] angle.
        }
        all fields must be filled, but some fields can fill
        zero.
        """
        if "annos" not in self._synthia_infos[0]:
            return None
        gt_annos = [info["annos"] for info in self._synthia_infos]
        dt_annos = self.convert_detection_to_kitti_annos(detections)
        # firstly convert standard detection to kitti-format dt annos
        z_axis = 1  # KITTI camera format use y as regular "z" axis.
        z_center = 1.0  # KITTI camera box's center is [0.5, 1, 0.5]
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.
        result_official_dict = get_official_eval_result(
            gt_annos,
            dt_annos,
            self._class_names,
            z_axis=z_axis,
            z_center=z_center)
        # SID: Removing COCO evaluation.
        # result_coco = get_coco_eval_result(
        #     gt_annos,
        #     dt_annos,
        #     self._class_names,
        #     z_axis=z_axis,
        #     z_center=z_center)
        return {
            "results": {
                "official": result_official_dict["result"],
                # "coco": result_coco["result"],
            },
            "detail": {
                "eval.kitti": {
                    "official": result_official_dict["detail"],
                    # "coco": result_coco["detail"]
                }
            },
        }

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        return self.getitem_from_sensor_data(input_dict)
    
    def getitem_from_sensor_data(self, input_dict):
        points = input_dict["lidar"]["points"]
        calib = input_dict["calib"]

        if "annotations" in input_dict["lidar"]:
            anno_dict = input_dict["lidar"]["annotations"]
            gt_dict = data_prep.anno_dict_to_gt_dict(anno_dict)
        
        # Data augmentation (optional).
        if self._prep_func_data_aug is not None:
            points, gt_dict = self._prep_func_data_aug(points, gt_dict, calib)

        # Convert points to voxels.
        example = self._prep_func_main(points, calib)

        # Create and add targets (optional).
        if self._prep_func_add_targets is not None:
            example = self._prep_func_add_targets(example, gt_dict)

        example["metadata"] = {}
        if "image_idx" in input_dict["metadata"]:
            example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    def get_sensor_data(self, query):
        read_image = False
        colored_lidar = False
        create_init_lidar = True
        idx = query
        if isinstance(query, dict):
            colored_lidar = "colored" in query["lidar"]
            read_image = ("cam" in query) or colored_lidar
            create_init_lidar = "init_lidar" in query
            if create_init_lidar:
                assert "num_beams" in query["init_lidar"]
            assert "lidar" in query
            idx = query["lidar"]["idx"]
        info = self._synthia_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "colored": False
            },
            "metadata": {
                "image_idx": info["image"]["image_idx"],
                "image_shape": info["image"]["image_shape"],
            },
            "calib": None,
            "cam": {},
            "depth": {
                "type": "depth_map",
                "image": None
            },
            "init_lidar": {
                "num_beams": None,
                "points": None
            }
        }

        image_info = info["image"]
        image_path = image_info['image_path']
        if read_image:
            image_path = self._root_path / image_path
            with open(str(image_path), 'rb') as f:
                image_str = f.read()
            res["cam"] = {
                "type": "camera",
                "data": image_str,
                "datatype": image_path.suffix[1:],
            }

        # Will always produce depth, needed for lidar.
        depth_info = info["depth"]
        depth_path = Path(depth_info["depth_path"])
        if not depth_path.is_absolute():
            depth_path = Path(self._root_path) / depth_info["depth_path"]
        
        # SYNTHIA depth formula: "Depth = 5000 * (R + G*256 + B*256*256) / (256*256*256 - 1)"
        np_depth_image = np.array(Image.open(depth_path))  # (H, W, 4) dtype=np.uint8
        R, G, B = [np_depth_image[:, :, e].astype(np.int64) for e in range(3)]  # (H, W), dtype=np.int64
        np_depth_image = 5000 * (R + G*256 + B*256*256) / (256*256*256 - 1)  # (H, W) dtype=np.float64
        np_depth_image = np_depth_image.astype(np.float32)  # (H, W) dtype=np.float32
        res["depth"] = {
            "type": "depth_map",
            "image": np_depth_image
        }

        if colored_lidar:
            # Concatenate depth map with colors.
            np_rgb_image = np.array(Image.open(io.BytesIO(image_str)))  # (H, W, 4)
            np_rgb_image = np_rgb_image[:, :, :3]  # (H, W, 3)
            H, W = np_depth_image.shape
            np_depth_image = np.concatenate((np_depth_image.reshape(H, W, 1),
                                            np_rgb_image), axis=2)  # (H, W, 4)
            res["lidar"]["colored"] = True
        else:
            np_depth_image = np_depth_image[..., np.newaxis]  # (H, W, 1)
            np_depth_image = np.concatenate((np_depth_image, np.ones_like(np_depth_image)),
                                            axis=2)  # (H, W, 2)

        points = cam_transforms.depth_map_to_point_cloud(
            np_depth_image,
            self.lc_device.CAMERA_PARAMS['matrix'])  # (N, 4) or (N, 6)
        
        # Convert from camera frame to velo frame.
        cam2velo = self.lc_device.TRANSFORMS['cTw']  # inverse extrinsics matrix
        xyz_velo = np.hstack((points[:, :3], np.ones([len(points), 1], dtype=points.dtype)))
        xyz_velo = xyz_velo @ cam2velo.T
        points = np.hstack((xyz_velo[:, :3], points[:, 3:]))

        # Simulated LiDAR points.
        if create_init_lidar and query["init_lidar"]["num_beams"] > 1:
            num_beams = query["init_lidar"]["num_beams"]
            if self.mb_lidar_mask is None or self.mb_lidar_mask["num_beams"] != num_beams:
                self.create_mb_lidar_mask(num_beams)
            init_lidar_points = points[self.mb_lidar_mask["1d_mask"]]  # (R, 4) or (R, 6)
            init_lidar_points = init_lidar_points[:, :3]  # (R, 3)
            # Only select points with less than MAX_RANGE value.
            init_lidar_points = init_lidar_points[init_lidar_points[:, 0] < self._MAX_RANGE, :]
            res["init_lidar"]["points"] = init_lidar_points

        # Only select points with less than MAX_RANGE value.
        points = points[points[:, 0] < self._MAX_RANGE, :]
        res["lidar"]["points"] = points

        # Compute single-beam lidar.
        if create_init_lidar and query["init_lidar"]["num_beams"] == 1:
            init_lidar_points = self.convert_points_to_sb_lidar(points)
            init_lidar_points = init_lidar_points[:, :3]  # (N, 3)
            res["init_lidar"]["points"] = init_lidar_points

        R0_rect = np.eye(4, dtype=np.float32) # there is no rectification in synthia
        Tr_velo_to_cam = self.lc_device.TRANSFORMS["wTc"]
        # Extended intrinsics matrix: shape= (4, 4).
        P2 = np.eye(4, dtype=np.float32)
        P2[:3, :3] = self.lc_device.CAMERA_PARAMS["matrix"]

        calib_dict = {
            'R0_rect': R0_rect,
            'Tr_velo_to_cam': Tr_velo_to_cam,
            'P2': P2
        }
        res["calib"] = calib_dict

        if 'annos' in info:
            annos = info['annos']
            # we need other objects to avoid collision when sample
            annos = synthia.remove_dontcare(annos)
            locs = annos["location"]
            dims = annos["dimensions"]
            rots = annos["rotation_y"]
            gt_names = annos["name"]
            gt_boxes = np.concatenate([locs, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)
            calib = info["calib"]
            gt_boxes = box_np_ops.box_camera_to_lidar(
                gt_boxes,
                r_rect=R0_rect,
                velo2cam=Tr_velo_to_cam
            )

            # only center format is allowed. so we need to convert
            # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
            box_np_ops.change_box3d_center_(gt_boxes, [0.5, 0.5, 0],
                                            [0.5, 0.5, 0.5])
            res["lidar"]["annotations"] = {
                'boxes': gt_boxes,
                'names': gt_names,
            }
            res["cam"]["annotations"] = {
                'boxes': annos["bbox"],
                'names': gt_names,
            }

        return res

    def convert_points_to_sb_lidar(self, points):
        """
        Converts a point cloud to approximate single beam lidar, positioned at a certain
        height.
        Args:
            points: (np.ndarray, dtype=np.float32, shape=(N, 3+)) point cloud, in velo frame.
        Return:
            sb_lidar: (np.ndarray, dtype=np.float32, shape=(N, 3+)) simulated single-beam lidar,
                      in velo frame.
        """
        z_center = self._SB_LIDAR_PARAMS["center"]
        thickness = self._SB_LIDAR_PARAMS["thickness"]

        # VERTICAL SPARSITY
        # Remove points outside thickness band.
        z = points[:, 2]
        keep = (z > z_center - 0.5 * thickness) & (z < z_center + 0.5 * thickness)
        points = points[keep]

        # ANGULAR SPARSITY
        # First compute horizontal angle.
        x, y = points[:, 0], points[:, 1]
        # Remember that x and y are in velo frame.
        # Also, θ is from +y axis (left to right).
        θ = np.rad2deg(np.arctan2(x, y))  # (N,) in degrees
        θ = θ / self._SB_LIDAR_PARAMS["angular_res"]  # in resolution units
        
        # θ_frac are in [0, 1).
        θ_int, θ_frac = np.divmod(θ, 1)  # (N,) and (N,)
        min_int, max_int = int(θ_int.min()), int(θ_int.max())

        keep = []
        for sector_min in range(min_int, max_int + 1):
            sector_max = sector_min + 1
            in_sector_inds = np.where((θ > sector_min) & (θ < sector_max))[0]
            if len(in_sector_inds) == 0:
                continue
            # Select the theta whose frac is closest to 0.5.
            θ_frac_in_sector = θ_frac[in_sector_inds]
            closest = np.abs(θ_frac_in_sector - 0.5).argmin()
            ind = in_sector_inds[closest]
            keep.append(ind)
        points = points[keep]

        return points


# TODO: copied from kitti_dataset.py but processed.
def convert_to_kitti_info_version2(info):
    """convert kitti info v1 to v2 if possible.
    """
    if "image" not in info or "calib" not in info or "point_cloud" not in info:
        info["image"] = {
            'image_shape': info["img_shape"],
            'image_idx': info['image_idx'],
            'image_path': info['img_path'],
        }
        info["calib"] = {
            "R0_rect": info['calib/R0_rect'],
            "Tr_velo_to_cam": info['calib/Tr_velo_to_cam'],
            "P2": info['calib/P2'],
        }
        info["point_cloud"] = {
            "velodyne_path": info['velodyne_path'],
        }


# TODO: copied from kitti_dataset.py but processed.
def kitti_anno_to_label_file(annos, folder):
    folder = Path(folder)
    for anno in annos:
        image_idx = anno["metadata"]["image_idx"]
        label_lines = []
        for j in range(anno["bbox"].shape[0]):
            label_dict = {
                'name': anno["name"][j],
                'alpha': anno["alpha"][j],
                'bbox': anno["bbox"][j],
                'location': anno["location"][j],
                'dimensions': anno["dimensions"][j],
                'rotation_y': anno["rotation_y"][j],
                'score': anno["score"][j],
            }
            label_line = kitti.kitti_result_line(label_dict)
            label_lines.append(label_line)
        label_file = folder / f"{kitti.get_image_index_str(image_idx)}.txt"
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w') as f:
            f.write(label_str)


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


# TODO: copied from kitti_dataset.py but processed.
def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=True,
                                num_features=4):
    for info in infos:
        pc_info = info["point_cloud"]
        image_info = info["image"]
        calib = info["calib"]
        if relative_path:
            v_path = str(Path(data_path) / pc_info["velodyne_path"])
        else:
            v_path = pc_info["velodyne_path"]
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if remove_outside:
            points_v = box_np_ops.remove_outside_points(
                points_v, rect, Trv2c, P2, image_info["image_shape"])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
            gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos["num_points_in_gt"] = num_points_in_gt.astype(np.int32)

def create_synthia_info_file(data_path, save_path=None, relative_path=True):
    imageset_folder = Path(__file__).resolve().parent / "SynthiaImageSets"

    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    def gen_synthia_infos(img_ids, filename):
        print("Generate info. this may take several minutes.")
        synthia_infos = synthia.get_synthia_image_info(
            data_path,
            depth=True,
            calib=True,
            image_ids=img_ids,
            relative_path=relative_path)
        # _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
        print(f"Synthia info file is saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(synthia_infos, f)
    
    img_ids = _read_imageset_file(str(imageset_folder / "train.txt"))
    gen_synthia_infos(img_ids, save_path / 'synthia_infos_train.pkl')
    
    img_ids = _read_imageset_file(str(imageset_folder / "test.txt"))
    gen_synthia_infos(img_ids, save_path / 'synthia_infos_test.pkl')

    img_ids = _read_imageset_file(str(imageset_folder / "train_random_subset.txt"))
    gen_synthia_infos(img_ids, save_path / 'synthia_infos_train_random_subset.pkl')

    img_ids = _read_imageset_file(str(imageset_folder / "test_random_subset.txt"))
    gen_synthia_infos(img_ids, save_path / 'synthia_infos_test_random_subset.pkl')

    img_ids = _read_imageset_file(str(imageset_folder / "test_2k_subset.txt"))
    gen_synthia_infos(img_ids, save_path / 'synthia_infos_test_2k_subset.pkl')


def point_cloud_to_depth_map(points, fov_deg, width, height, debug=False):
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
        fov_deg: (float) field of view of camera in degrees.
        width: (int) number of pixels along width.
        height: (int) number of pixels along height.
    """
    # Camera intrinsics matrix.
    cam_matrix = pylc.compute_camera_instrics_matrix(fov_deg, width, height)
    
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



if __name__ == "__main__":
    fire.Fire()