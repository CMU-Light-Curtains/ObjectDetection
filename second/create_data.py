import copy
from pathlib import Path
import pickle

import fire

import second.data.kitti_dataset as kitti_ds
import second.data.nuscenes_dataset as nu_ds
import second.data.vkitti_dataset as vkitti_ds
import second.data.synthia_dataset as synthia_ds
from second.data.all_dataset import create_groundtruth_database

def kitti_data_prep(root_path):
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    create_groundtruth_database("KittiDataset", root_path, Path(root_path) / "kitti_infos_train.pkl")

def nuscenes_data_prep(root_path, version, dataset_name, max_sweeps=10):
    nu_ds.create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps)
    name = "infos_train.pkl"
    if version == "v1.0-test":
        name = "infos_test.pkl"
    create_groundtruth_database(dataset_name, root_path, Path(root_path) / name)

def vkitti_data_prep(root_path):
    vkitti_ds.create_vkitti_info_file(root_path)
    # create_reduced_point_cloud removes points outside the frustrum of image_2 projected in 2D.
    # In VKITTI, there is no lidar; point cloud is obtained by projecting depth-map to 3D, hence
    # point cloud is already in frustrum!
    # vkitti_ds.create_reduced_point_cloud(root_path)

    # Skip this for now, but will be needed for SECOND-style data augmentation.
    # create_groundtruth_database("VirtualKittiDataset", root_path, Path(root_path) / "vkitti_infos_train.pkl")

def synthia_data_prep(root_path):
    synthia_ds.create_synthia_info_file(root_path)

if __name__ == '__main__':
    fire.Fire()
