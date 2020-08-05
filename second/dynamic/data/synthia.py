import concurrent.futures as futures
import io
import os
from pathlib import Path
import pickle
import tqdm
from PIL import Image
import fire
import skimage

import numpy as np

from second.core import box_np_ops
from second.core import camera_transforms as cam_transforms


class SynthiaVideoDataset:
    def __init__(self, split, cam=False, colored_pc=False):
        """
        Format of video_infos: [(vid, video_info)] where video_info = [{frame_info}]
        """
        video_infos_path = Path(os.environ["DATADIR"]) / "synthia" / f"synthia_video_infos_{split}.pkl"
        with open(video_infos_path, 'rb') as f:
            self.video_infos = pickle.load(f)

        self.cam = cam
        self.colored_pc = colored_pc

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        vid, video_info = self.video_infos[idx]
        video = Video(vid, video_info, self.cam, self.colored_pc)
        return video


########################################################################################################################
#region: Video class
########################################################################################################################
class Video:
    def __init__(self, vid, video_info, cam=False, colored_pc=False):
        """
        Args:
            vid: (int) video id
            video_info: [{frame_info}], frame_info dictionary for every frame
            cam: (bool) whether to load camera image
            colored_pc: (bool) whether to colorize point cloud
        """
        self.vid = vid
        self.video_info = video_info
        self.cam = cam
        self.colored_pc = colored_pc
        self._MAX_DEPTH = 80.0  # only consider points within this depth
    
    def __len__(self):
        return len(self.video_info)
    
    def __getitem__(self, idx):
        """
        Args:
            idx: (int) this is the index of the frame, from 0 to len(Video) - 1
        """
        frame_info = self.video_info[idx]  # frame info
        root_path = Path(os.environ["DATADIR"]) / "synthia"

        res = {
            "depth": None,  # np.ndarray, dtype=np.float32, shape=(H, W)
            "points": None,
            "cam": {
                "image_str": None,  # str, image string
                "datatype": None,  # str, suffix type
            },
            "metadata": {
                "frameid": frame_info["frameid"],
                "image_shape": frame_info["image"]["image_shape"]
            },
            "calib": frame_info["calib"],
            "annos": None
        }

        # --------------------------------------------------------------------------------------------------------------
        # depth
        # --------------------------------------------------------------------------------------------------------------
        depth_path = Path(frame_info["depth"]["depth_path"])
        if not depth_path.is_absolute():
            depth_path = root_path / depth_path
        # synthia depth formula: "Depth = 5000 * (R + G*256 + B*256*256) / (256*256*256 - 1)"
        np_depth_image = np.array(Image.open(depth_path))  # (H, W, 4) dtype=np.uint8
        R, G, B = [np_depth_image[:, :, e].astype(np.int64) for e in range(3)]  # (H, W), dtype=np.int64
        np_depth_image = 5000 * (R + G*256 + B*256*256) / (256*256*256 - 1)  # (H, W) dtype=np.float64
        np_depth_image = np_depth_image.astype(np.float32)  # (H, W) dtype=np.float32
        res["depth"] = np_depth_image

        # --------------------------------------------------------------------------------------------------------------
        # cam
        # --------------------------------------------------------------------------------------------------------------
        if self.cam or self.colored_pc:
            image_path = Path(frame_info['image']['image_path'])
            if not image_path.is_absolute():
                image_path = root_path / image_path
            with open(str(image_path), 'rb') as f:
                image_str = f.read()
            res["cam"]["image_str"] = image_str
            res["cam"]["datatype"] = image_path.suffix[1:]
        
        # --------------------------------------------------------------------------------------------------------------
        # points
        # --------------------------------------------------------------------------------------------------------------
        np_depth_image = np_depth_image[..., np.newaxis]  # (H, W, 1)
        if self.colored_pc:
            # concatenate depth map with colors
            np_rgb_image = np.array(Image.open(io.BytesIO(image_str)))  # (H, W, 4)
            np_rgb_image = np_rgb_image[:, :, :3]  # (H, W, 3)
            np_depth_image = np.concatenate([np_depth_image, np_rgb_image], axis=2)  # (H, W, 4)

        # points in cam frame
        P2 = frame_info['calib']['P2']  # intrinsics matrix
        if P2.shape == (4, 4):
            P2 = P2[:3, :3]
        else:
            assert P2.shape == (3, 3)
        points = cam_transforms.depth_map_to_point_cloud(np_depth_image, P2)  # (N, 3) or (N, 6)
        
        # points in velo frame
        Tr_velo_to_cam = frame_info['calib']['Tr_velo_to_cam']  # extrinsics matrix
        Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
        xyz1_cam = np.hstack((points[:, :3], np.ones([len(points), 1], dtype=points.dtype)))  # (N, 4)
        xyz1_velo = xyz1_cam @ Tr_cam_to_velo.T  # (N, 4)
        points = np.hstack((xyz1_velo[:, :3], points[:, 3:]))  # (N, 3) or (N, 6)

        # points within MAX_DEPTH
        points = points[points[:, 0] < self._MAX_DEPTH, :]  # (M, 3) or (M, 6)
        res["points"] = points

        # --------------------------------------------------------------------------------------------------------------
        # annos
        # --------------------------------------------------------------------------------------------------------------
        annos = frame_info['annos']
        annos = self._remove_dontcare(annos)
        locs = annos["location"]
        dims = annos["dimensions"]
        rots = annos["rotation_y"]
        gt_names = annos["name"]
        gt_boxes = np.concatenate([locs, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes,
                                                  r_rect=frame_info["calib"]["R0_rect"],
                                                  velo2cam=frame_info["calib"]["Tr_velo_to_cam"]
        )
        # only center format is allowed. so we need to convert kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
        box_np_ops.change_box3d_center_(gt_boxes, [0.5, 0.5, 0], [0.5, 0.5, 0.5])
        res["annos"] = {
            'names': gt_names,
            'boxes': gt_boxes,
            'boxes2d': annos["bbox"]
        }

        return res
    
    def _remove_dontcare(self, annos):
        filtered_annos = {}
        relevant_annotation_indices = [i for i, x in enumerate(annos['name']) if x != "DontCare"]
        for key in annos.keys():
            filtered_annos[key] = (annos[key][relevant_annotation_indices])
        return filtered_annos

#endregion
########################################################################################################################
#region: Functions to create videoset files (second/dynamic/SynthiaVideoSets) and load vid2metadata
########################################################################################################################

def create_videoset_files():
    """
    Each row in the vid2metadata file has the following columns:

    vid  subdir                                      frameids
    0001 train/test5_10segs_weather...2018_12-47-37  000101 000102 ... 000280

    vid      : video index (from 0 to TOTAL_VIDEOS_IN_SYNTHIA-1)
    subdir   : synthia subdirectory
    frameids : index of the subdirectory. files are named using this index.
    """
    root_path = Path(os.environ["DATADIR"]) / "synthia"
    videosets_dir = Path(__file__).resolve().parent / "SynthiaVideoSets"

    metadata, trainvids, testvids = [], [], []

    train_dir = root_path / "train"
    test_dir = root_path / "test"
    for ttdir in [train_dir, test_dir]:
        for updir in sorted([e for e in ttdir.iterdir() if e.is_dir()]):
            for subdir in sorted([e for e in updir.iterdir() if e.is_dir()]):
                if not (subdir / "labels_kitti").is_dir():
                    continue # empty data.
                frameids = []
                for imfile in sorted((subdir / "labels_kitti").iterdir()):
                    # Load all frames in the scene, whether they contain a Car or not.
                    frame_id = imfile.name.split('.')[0]
                    frameids.append(frame_id)
                metadata.append((subdir, frameids))
    
    with open(videosets_dir / "vid2metadata.txt", 'w') as f:
        for vid, metadatum in enumerate(metadata):
            vid = str(vid).rjust(4, '0')
            subdir, frameids = metadatum
            subdir = str(subdir.relative_to(root_path))
            if subdir.startswith('train/'):
                trainvids.append(vid)
            else:
                testvids.append(vid)
            rowtext = ' '.join([vid, subdir] + frameids)
            print(rowtext, file=f)
    
    print(f'Synthia TRAIN: has {len(trainvids)} videos.')
    print(f'Synthia TEST : has {len(testvids)} videos.')

    np.random.seed(0)

    with open(videosets_dir / "train.txt", 'w') as f:
        for vid in np.random.permutation(trainvids):
            print(vid, file=f)
    
    with open(videosets_dir / "test.txt", 'w') as f:
        for vid in np.random.permutation(testvids):
            print(vid, file=f)


def load_vid2metadata():
    """
    Format:
    {
        vid: { // metadata
               "subdir"  : subdirectory
               "frameids": [list of frameids]
             }
    }

    vid: (int) video id
    subdirectory: (str) path to subdirectory
    frameids: list(int) list of frameids
    """
    VID2METADATA = {}
    vid2metadata_file = Path(__file__).resolve().parent / "SynthiaVideoSets" / "vid2metadata.txt"
    with open(vid2metadata_file, 'r') as f:
        lines = [line.rstrip().split() for line in f.readlines()]
        for line in lines:
            vid, subdir, frameids = line[0], line[1], line[2:]
            VID2METADATA[vid] = {'subdir': subdir, 'frameids': frameids}
    return VID2METADATA

#endregion
########################################################################################################################
#region: Functions to create video_infos pickle files in $DATASET/synthia
########################################################################################################################

def get_file_path(subdir,
                  frame_id,
                  info_type='image_2',
                  file_tail='.png',
                  relative_path=True,
                  exist_check=True):
    root_path = Path(os.environ["DATADIR"]) / "synthia"
    rel_file_path = f"{subdir}/{info_type}/{frame_id}{file_tail}"
    abs_file_path = root_path / rel_file_path
    if exist_check and not abs_file_path.exists():
        raise ValueError("file not exist: {}".format(abs_file_path))
    if relative_path:
        return str(rel_file_path)
    else:
        return str(abs_file_path)

def get_synthia_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'difficulty': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)

    # Adding difficulty to annotations.
    min_height = [40, 25, 25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [0, 1, 2]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [0.15, 0.3, 0.5]  # maximum truncation level of the groundtruth used for evaluation
    dims = annotations['dimensions']  # lhw format
    bbox = annotations['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annotations['occluded']
    truncation = annotations['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annotations["difficulty"] = np.array(diff, np.int32)

    return annotations

def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat
    
def get_video_info(metadata,
                   extend_matrix=True,
                   num_worker=8,
                   relative_path=True,
                   with_imageshape=True):
    """ 
    Args:
        metadata: {
                    "subdir"  : subdirectory
                    "frameids": [list of frameids]
                  }
    Returns:
        video_info: list(frame_info), frame_info dictionary for every frame
    """
    subdir   = metadata["subdir"]
    frameids = metadata["frameids"]
    root_path = Path(os.environ["DATADIR"]) / "synthia"
    
    def get_frame_info(frameid):
        depth_path = get_file_path(subdir, frameid, 'Depth'       , '.png', relative_path)
        image_path = get_file_path(subdir, frameid, 'RGB'         , '.png', relative_path)
        label_path = get_file_path(subdir, frameid, 'labels_kitti', '.txt', relative_path)
        calib_path = get_file_path(subdir, frameid, 'calib_kitti' , '.txt', relative_path=False)

        frame_info = {
            'frameid': frameid,
            'image': {
                'image_path': image_path,
                'image_shape': None
            },
            'depth': {
                'depth_path': depth_path    
            },
            'annos': None,
            'calib': {

            }
        }

        # image shape
        img_path = frame_info['image']['image_path']
        if relative_path:
            img_path = str(root_path / img_path)
        frame_info['image']['image_shape'] = np.array(skimage.io.imread(img_path).shape[:2], dtype=np.int32)
        
        # annos
        if relative_path:
            label_path = str(root_path / label_path)
        annotations = get_synthia_label_anno(label_path)
        frame_info['annos'] = annotations

        # calib
        with open(calib_path, 'r') as f:
            lines = f.readlines()
        P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]]).reshape([3, 4])
        P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]]).reshape([3, 4])
        P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape([3, 4])
        P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]]).reshape([3, 4])
        if extend_matrix:
            P0 = _extend_matrix(P0)
            P1 = _extend_matrix(P1)
            P2 = _extend_matrix(P2)
            P3 = _extend_matrix(P3)
        R0_rect = np.array([float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3])
        if extend_matrix:
            rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
            rect_4x4[3, 3] = 1.
            rect_4x4[:3, :3] = R0_rect
        else:
            rect_4x4 = R0_rect

        Tr_velo_to_cam = np.array([[0, -1,  0,  0],  # x <- -y
                                   [0,  0, -1,  0],  # y <- -z
                                   [1,  0,  0,  0],  # z <- +x
                                    [0,  0,  0,  1]], dtype=np.float32)
        frame_info["calib"]['P0'] = P0
        frame_info["calib"]['P1'] = P1
        frame_info["calib"]['P2'] = P2
        frame_info["calib"]['P3'] = P3
        frame_info["calib"]['R0_rect'] = rect_4x4
        frame_info["calib"]['Tr_velo_to_cam'] = Tr_velo_to_cam

        return frame_info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        frame_infos = executor.map(get_frame_info, frameids)
    return list(frame_infos)

def create_video_infos_file(relative_path=True, split='train'):
    """
    Format of video_infos: [(vid, video_info)] where video_info = [{frame_info}]
    """
    VID2METADATA = load_vid2metadata()
    videoset_file = Path(__file__).resolve().parent / "SynthiaVideoSets" / f"{split}.txt"
    
    root_path = Path(os.environ["DATADIR"]) / "synthia"
    video_infos_pkl_file = root_path / f'synthia_video_infos_{split}.pkl'

    with open(videoset_file, 'r') as f:
        vids = [line.rstrip() for line in f.readlines()]

    video_infos = []
    print("Generating video infos. This may take several minutes.")
    for vid in tqdm.tqdm(vids):
        metadata = VID2METADATA[vid]
        video_info = get_video_info(metadata, relative_path=True)
        video_infos.append((vid, video_info))

    print(f"Synthia video infos file is saved to {video_infos_pkl_file}")
    with open(video_infos_pkl_file, 'wb') as f:
        pickle.dump(video_infos, f)

#endregion
########################################################################################################################


if __name__ == "__main__":
    fire.Fire()
