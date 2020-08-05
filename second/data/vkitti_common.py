
import concurrent.futures as futures
import numpy as np
import os
import pandas as pd
import pathlib
import re
from collections import OrderedDict

import numpy as np
from skimage import io

from second.data.kitti_common import iou

VARIATIONS = ['clone', \
              '15-deg-left', \
              '15-deg-right', \
              '30-deg-left', \
              '30-deg-right', \
              'fog', \
              'morning', \
              'overcast', \
              'rain', \
              'sunset']

SCENE_SIZES = {'0001': 447,  # total is 2126 for each variation
               '0002': 233,
               '0006': 270,
               '0018': 339,
               '0020': 837}

SCENE_IDS = ['0001', '0002', '0006', '0018', '0020']

IDX2METADATA = None
idx2metadata_file = pathlib.Path(__file__).resolve().parent / "VKittiImageSets" / "idx2metadata.txt"
with open(idx2metadata_file, 'r') as f:
    lines = [line.rstrip().split() for line in f.readlines()]
    IDX2METADATA = {int(line[0]): line[1:] for line in lines}


def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)


def get_vkitti_info_path(idx,
                         prefix,
                         info_type='image_2',
                         file_tail='.png',
                         relative_path=True,
                         exist_check=True):
    scene_id, variation, frame_id = IDX2METADATA[idx]
    img_idx_str = f"{scene_id}/{variation}/{frame_id}{file_tail}"
    prefix = pathlib.Path(prefix)
    file_path = info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(scene_id, variation, frame_id, root_path, relative_path=True, exist_check=True):
    file_path = pathlib.Path('vkitti_1.3.1_rgb') / scene_id / variation / f'{frame_id}.png'
    if exist_check and not (root_path / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(root_path / file_path)

def get_label_path(scene_id, variation, root_path, relative_path=True, exist_check=True):
    file_path = pathlib.Path('vkitti_1.3.1_motgt') / f'{scene_id}_{variation}.txt'
    if exist_check and not (root_path / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(root_path / file_path)

def get_depth_path(scene_id, variation, frame_id, root_path, relative_path=True, exist_check=True):
    file_path = pathlib.Path('vkitti_1.3.1_depthgt') / scene_id / variation / f'{frame_id}.png'
    if exist_check and not (root_path / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(root_path / file_path)

def _check_vkitti_directory(root_path):
    root_path = pathlib.Path(root_path)
    results = []
    results.append((root_path / 'vkitti_1.3.1_rgb').exists())
    results.append((root_path / 'vkitti_1.3.1_motgt').exists())
    results.append((root_path / 'vkitti_1.3.1_depthgt').exists())
    # Labels.
    for scene_id in SCENE_IDS:
        for variation in VARIATIONS:
            results.append((root_path / 'vkitti_1.3.1_motgt' / f'{scene_id}_{variation}.txt').exists())    
    # Image and Depth.
    for info_folder in ['vkitti_1.3.1_rgb', 'vkitti_1.3.1_depthgt']:
        for variation in VARIATIONS:
            for scene_id, scene_size in SCENE_SIZES.items():
                path_to_info_scene = root_path / info_folder / scene_id / variation
                results.append(len(list(path_to_info_scene.glob('*.png'))) == scene_size)
    return np.array(results, dtype=np.bool)

def _create_idx2metadata_file(root_path):
    """File that contains a mapping from idx (int) to metadata (scene_id, variation, frame_id)"""
    idx2metadata_file = pathlib.Path(__file__).resolve().parent / "VKittiImageSets" / "idx2metadata.txt"
    # Get frame_ids from scene_id.
    scene_id_to_frame_ids = {}
    for scene_id in SCENE_IDS:
        frame_ids = (pathlib.Path(root_path) / 'vkitti_1.3.1_rgb' / scene_id / 'clone').glob('*.png')
        frame_ids = sorted([path.stem for path in frame_ids])
        scene_id_to_frame_ids[scene_id] = frame_ids
    with open(idx2metadata_file, 'w') as f:
        idx = 0
        for variation in VARIATIONS:
            for scene_id in SCENE_IDS:
                for frame_id in scene_id_to_frame_ids[scene_id]:
                    print(f'{idx:06d} {scene_id} {variation} {frame_id}', file=f)
                    idx += 1

def get_vkitti_image_info(path,
                          label_info=True,
                          depth=False,
                          image_ids=7481,
                          num_worker=8,
                          relative_path=True,
                          with_imageshape=True):
    """ 
    KITTI annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = pathlib.Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    
    def map_func(idx):
        scene_id, variation, frame_id = IDX2METADATA[idx]
        info = {}
        depth_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if depth:
            depth_info['depth_path'] = get_depth_path(scene_id, variation, frame_id, root_path, relative_path)
            info['depth'] = depth_info
        
        image_info['image_path'] = get_image_path(scene_id, variation, frame_id, root_path, relative_path)

        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(scene_id, variation, root_path, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path, frame_id)
        info["image"] = image_info

        R0_rect = np.eye(4, dtype=np.float32)
        Tr_velo_to_cam = np.array([[0, -1,  0,  0],  # x <- -y
                                   [0,  0, -1,  0],  # y <- -z
                                   [1,  0,  0,  0],  # z <- +x
                                   [0,  0,  0,  1]], dtype=np.float32)
        P2 = np.array([[725.,   0.,   621., 0.],
                       [  0., 725., 187.5, 0.],
                       [  0.,   0.,     1., 0.],
                       [  0.,   0.,     0., 1.]], dtype=np.float32)
        info["calib"] = {
            "R0_rect": R0_rect,
            "Tr_velo_to_cam": Tr_velo_to_cam,
            "P2": P2
        }

        if annotations is not None:
            info['annos'] = annotations
            add_difficulty_to_annos(info)
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)
    return list(image_infos)


def label_str_to_int(labels, remove_dontcare=True, dtype=np.int32):
    class_to_label = get_class_to_label_map()
    ret = np.array([class_to_label[l] for l in labels], dtype=dtype)
    if remove_dontcare:
        ret = ret[ret > 0]
    return ret

def get_class_to_label_map():
    class_to_label = {
        'Car': 0,
        'Pedestrian': 1,
        'Cyclist': 2,
        'Van': 3,
        'Person_sitting': 4,
        'Truck': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': -1,
    }
    return class_to_label

def get_classes():
    return get_class_to_label_map().keys()

def filter_gt_boxes(gt_boxes, gt_labels, used_classes):
    mask = np.array([l in used_classes for l in gt_labels], dtype=np.bool)
    return mask

def filter_anno_by_mask(image_anno, mask):
    img_filtered_annotations = {}
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][mask])
    return img_filtered_annotations


def filter_infos_by_used_classes(infos, used_classes):
    new_infos = []
    for info in infos:
        annos = info["annos"]
        name_in_info = False
        for name in used_classes:
            if name in annos["name"]:
                name_in_info = True
                break
        if name_in_info:
            new_infos.append(info)
    return new_infos

def remove_dontcare(image_anno):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno['name']) if x != "DontCare"
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    return img_filtered_annotations

def remove_low_height(image_anno, thresh):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, s in enumerate(image_anno['bbox']) if (s[3] - s[1]) >= thresh
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    return img_filtered_annotations

def remove_low_score(image_anno, thresh):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, s in enumerate(image_anno['score']) if s >= thresh
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    return img_filtered_annotations

def keep_arrays_by_name(gt_names, used_classes):
    inds = [
        i for i, x in enumerate(gt_names) if x in used_classes
    ]
    inds = np.array(inds, dtype=np.int64)
    return inds

def drop_arrays_by_name(gt_names, used_classes):
    inds = [
        i for i, x in enumerate(gt_names) if x not in used_classes
    ]
    inds = np.array(inds, dtype=np.int64)
    return inds

def apply_mask_(array_dict):
    pass

def filter_vkitti_anno(image_anno,
                       used_classes,
                       used_difficulty=None,
                       dontcare_iou=None):
    if not isinstance(used_classes, (list, tuple, np.ndarray)):
        used_classes = [used_classes]
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno['name']) if x in used_classes
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    if used_difficulty is not None:
        relevant_annotation_indices = [
            i for i, x in enumerate(img_filtered_annotations['difficulty'])
            if x in used_difficulty
        ]
        for key in image_anno.keys():
            img_filtered_annotations[key] = (
                img_filtered_annotations[key][relevant_annotation_indices])

    if 'DontCare' in used_classes and dontcare_iou is not None:
        dont_care_indices = [
            i for i, x in enumerate(img_filtered_annotations['name'])
            if x == 'DontCare'
        ]
        # bounding box format [y_min, x_min, y_max, x_max]
        all_boxes = img_filtered_annotations['bbox']
        ious = iou(all_boxes, all_boxes[dont_care_indices])

        # Remove all bounding boxes that overlap with a dontcare region.
        if ious.size > 0:
            boxes_to_remove = np.amax(ious, axis=1) > dontcare_iou
            for key in image_anno.keys():
                img_filtered_annotations[key] = (img_filtered_annotations[key][
                    np.logical_not(boxes_to_remove)])
    return img_filtered_annotations


def filter_annos_class(image_annos, used_class):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(anno['name']) if x in used_class
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_annos_low_score(image_annos, thresh):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, s in enumerate(anno['score']) if s >= thresh
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_annos_difficulty(image_annos, used_difficulty):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(anno['difficulty']) if x in used_difficulty
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_annos_low_height(image_annos, thresh):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, s in enumerate(anno['bbox']) if (s[3] - s[1]) >= thresh
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos

def filter_empty_annos(image_annos):
    new_image_annos = []
    for anno in image_annos:
        if anno["name"].shape[0] > 0:
            new_image_annos.append(anno.copy())
    return new_image_annos


def vkitti_result_line(result_dict, precision=4):
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError("you must specify a value for {}".format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError("unknown key. supported key:{}".format(
                res_dict.keys()))
    return ' '.join(res_line)

def annos_to_vkitti_label(annos):
    num_instance = len(annos["name"])
    result_lines = []
    for i in range(num_instance):
        result_dict = {
            'name': annos["name"][i],
            'truncated': annos["truncated"][i],
            'occluded': annos["occluded"][i],
            'alpha':annos["alpha"][i],
            'bbox': annos["bbox"][i],
            'dimensions': annos["dimensions"][i],
            'location': annos["location"][i],
            'rotation_y': annos["rotation_y"][i],
        }
        line = vkitti_result_line(result_dict)
        result_lines.append(line)
    return result_lines

def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
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
    annos["difficulty"] = np.array(diff, np.int32)
    return diff


def add_difficulty_to_annos_v2(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = not ((occlusion > max_occlusion[0]) or (height < min_height[0])
                 or (truncation > max_trunc[0]))
    moderate_mask = not ((occlusion > max_occlusion[1]) or (height < min_height[1])
                 or (truncation > max_trunc[1]))
    hard_mask = not ((occlusion > max_occlusion[2]) or (height < min_height[2])
                 or (truncation > max_trunc[2]))
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
    annos["difficulty"] = np.array(diff, np.int32)
    return diff


def get_label_anno(label_path, frame_id):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    content = pd.read_csv(label_path, sep=" ", index_col=False)
    content = content[content['frame'] == int(frame_id)]
    num_objects = len(content) - (content['label'] == 'DontCare').sum()
    annotations['name'] = content['label'].to_numpy().astype(np.str)
    num_gt = len(annotations['name'])
    annotations['truncated'] = content['truncr'].to_numpy()
    annotations['occluded'] = content['occluded'].to_numpy()
    annotations['alpha'] = content['alpha'].to_numpy()
    annotations['bbox'] = content[['l', 't', 'r', 'b']].to_numpy()
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = content[['l3d', 'h3d', 'w3d']].to_numpy()
    annotations['location'] = content[['x3d', 'y3d', 'z3d']].to_numpy()
    annotations['rotation_y'] = content['ry'].to_numpy()
    annotations['score'] = np.zeros((annotations['bbox'].shape[0],))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations

def get_pseudo_label_anno():
    annotations = {}
    annotations.update({
        'name': np.array(['Car']),
        'truncated': np.array([0.0]),
        'occluded': np.array([0]),
        'alpha': np.array([0.0]),
        'bbox': np.array([[0.1, 0.1, 15.0, 15.0]]),
        'dimensions': np.array([[0.1, 0.1, 15.0, 15.0]]),
        'location': np.array([[0.1, 0.1, 15.0]]),
        'rotation_y': np.array([[0.1, 0.1, 15.0]])
    })
    return annotations

def get_start_result_anno():
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
        'score': [],
    })
    return annotations

def empty_result_anno():
    annotations = {}
    annotations.update({
        'name': np.array([]),
        'truncated': np.array([]),
        'occluded': np.array([]),
        'alpha': np.array([]),
        'bbox': np.zeros([0, 4]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
        'score': np.array([]),
    })
    return annotations

def get_label_annos(root_path, image_ids):
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    for idx in image_ids:
        scene_id, variation, frame_id = IDX2METADATA[idx]
        label_filename = get_label_path(scene_id, variation, root_path, relative_path=False)
        anno = get_label_anno(label_filename, frame_id)
        num_example = anno["name"].shape[0]
        anno["image_idx"] = np.array([idx] * num_example, dtype=np.int64)
        annos.append(anno)
    return annos


def anno_to_rbboxes(anno):
    loc = anno["location"]
    dims = anno["dimensions"]
    rots = anno["rotation_y"]
    rbboxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    return rbboxes
