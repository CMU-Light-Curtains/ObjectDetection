# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Input reader builder.

Creates data sources for DetectionModels from an InputReader config. See
input_reader.proto for options.

Note: If users wishes to also use their own InputReaders with the Object
Detection configuration framework, they should define their own builder function
that wraps the build function.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from second.builder import dataset_builder
from second.protos import input_reader_pb2
from second.light_curtain.policy import get_lc_policy_class
from second.light_curtain.utils import LC_PROCESS

class LidarDatasetWrapper(Dataset):
    """ convert our dataset to Dataset class in pytorch.
    """

    def __init__(self, dataset, num_workers):
        self._dataset = dataset
        self._num_workers = num_workers

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def sampler(self, shuffle=False):
        if shuffle:
            return torch.utils.data.RandomSampler(self)
        else:
            return torch.utils.data.SequentialSampler(self)
    
    def evaluation(self, detections, output_dir):
        return self.dataset.evaluation(detections, output_dir)

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def num_workers(self):
        return self._num_workers


class CumLCDatasetWrapper(Dataset):
    """ convert our dataset to Dataset class in pytorch.
    """

    def __init__(self, dataset, net, lc_policy, lc_horizon, init_lidar_num_beams, sparsify_config, use_cache=False, contiguous=False, debug=False):
        self._dataset = dataset
        # For inner loop during training.
        self._lc_policy = get_lc_policy_class(lc_policy)(dataset.lc_device)
        self._lc_horizon = lc_horizon
        self._init_lidar_num_beams = init_lidar_num_beams
        self._sparsify_config = sparsify_config
        self._net = net

        # NOTE: Set "use_cache" to True **ONLY WHEN** not updating weights of
        # "net" while iterating over this dataset. In that case, __getitem__
        # will assume that data stored in cache was generated from the same net,
        # and will be used to save computation time.
        self._use_cache = use_cache
        self._contiguous = contiguous
        self._debug = debug

        self._CACHE_ = {'scene_id': None,
                        'step': None,
                        'sensor_data': None}

    def __len__(self):
        return len(self._dataset) * (self._lc_horizon + 1)

    def scene_id_and_step_to_idx(self, scene_id, step):
        idx = step * len(self._dataset) + scene_id
        return idx

    def idx_to_scene_id_and_step(self, idx):
        scene_id = idx % len(self._dataset)
        step = idx // len(self._dataset)
        return scene_id, step

    def __getitem__(self, idx):
        scene_id, step = self.idx_to_scene_id_and_step(idx)

        if self._use_cache and self._CACHE_["scene_id"] == scene_id and self._CACHE_["step"] <= step:
            if self._debug:
                print(f"CumLCDatasetWrapper: Cache hit for scene {scene_id} and step {step}")
            # If _CACHE_ is compatible, start from its sensor data.
            # Run LC_PROCESS for remaining number of steps.
            sensor_data = self._CACHE_["sensor_data"]
            steps_remaining = step - self._CACHE_["step"]
        else:
            if self._debug:
                print(f"CumLCDatasetWrapper: Cache miss for scene {scene_id} and step {step}")
            # Start from sensor_data at step = 0.
            # Run LC_PROCESS for all steps.

            # sensor data at step = 0.
            query = {
                "lidar": {
                    "idx": scene_id
                },
                "depth": {},
                "init_lidar": {
                    "num_beams": self._init_lidar_num_beams
                }
            }
            sensor_data = self._dataset.get_sensor_data(query)
            init_lidar_points = sensor_data["init_lidar"]["points"]
            init_lidar_points = np.hstack((init_lidar_points, np.ones([len(init_lidar_points), 1], dtype=np.float32)))
            sensor_data["lidar"]["points"] = init_lidar_points
            steps_remaining = step

        # sensor_data after remaining steps.
        for _ in range(steps_remaining):
            sensor_data = LC_PROCESS(sensor_data,
                                     self._net,
                                     self._dataset,
                                     self._lc_policy,
                                     self._sparsify_config)["next_sensor_data"]

        # Save this sensor_data to _CACHE_.
        self._CACHE_.update({"scene_id": scene_id,
                             "step": step,
                             "sensor_data": sensor_data})

        # Now, the sensor data constitutes the main input data, and should be
        # preprocessed normally according to the dataset's preprocessing.
        example = self._dataset.getitem_from_sensor_data(sensor_data)
        return example
    
    def sampler(self, shuffle=False):
        if self._contiguous:
            if shuffle:
                return RandomContiguousSampler(self)
            else:
                return SequentialContiguousSampler(self)
        else:
            if shuffle:
                return torch.utils.data.RandomSampler(self)
            else:
                return torch.utils.data.SequentialSampler(self)

    def evaluation(self, detections, output_dir, prefix=""):
        # NOTE: evaluation assumes that shuffle was set to False when iterating through dataset.
        assert len(detections) == len(self), "detection must be equal to size of dataset"
        num_scenes, num_steps = len(self._dataset), self._lc_horizon + 1

        # Re-arrange detections into an array of shape num_scenes x num_steps.
        if self._contiguous:
            detections = np.array(detections).reshape(num_scenes, num_steps)
        else:
            detections = np.array(detections).reshape(num_steps, num_scenes).T
        
        # Evaluate separately for each step.
        all_results = {
            "results": {
            },
            "detail": {
                "eval.kitti": {
                }
            },
        }

        def modify_name(old_name, step):
            name = f"step{step}_{old_name}"
            if prefix != "":
                name = f"{prefix}_{name}"
            return name

        for step in range(num_steps):
            step_results = self.dataset.evaluation(detections[:, step], output_dir)
            
            for k, v in step_results["results"].items():
                all_results["results"][modify_name(k, step)] = v
            for k, v in step_results["detail"]["eval.kitti"].items():
                all_results["detail"]["eval.kitti"][modify_name(k, step)] = v

        return all_results

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def num_workers(self):
        # Num workers should always be 0 for this wrapper as CUDA tensors
        # are used in __getitem__, which should not be parallelized.
        return 0

# ####################################################################
# Samplers for CumLCDatasetWrapper
# ####################################################################
class SequentialContiguousSampler(Sampler):
    def __init__(self, cum_lc_dataset):
        super(SequentialContiguousSampler, self).__init__(cum_lc_dataset)
        self._cum_lc_dataset = cum_lc_dataset

    def __iter__(self):
        cum_lc_ds = self._cum_lc_dataset
        for scene_id in range(len(cum_lc_ds._dataset)):
            for step in range(cum_lc_ds._lc_horizon + 1):
                idx = cum_lc_ds.scene_id_and_step_to_idx(scene_id, step)
                yield idx

class RandomContiguousSampler(Sampler):
    def __init__(self, cum_lc_dataset):
        super(RandomContiguousSampler, self).__init__(cum_lc_dataset)
        self._cum_lc_dataset = cum_lc_dataset

    def __iter__(self):
        cum_lc_ds = self._cum_lc_dataset
        for scene_id in np.random.permutation(len(cum_lc_ds._dataset)):
            for step in range(cum_lc_ds._lc_horizon + 1):
                idx = cum_lc_ds.scene_id_and_step_to_idx(scene_id, step)
                yield idx
# ####################################################################


def build(input_reader_config,
          model_config,
          training,
          voxel_generator,
          target_assigner=None,
          net=None,
          multi_gpu=False) -> Dataset:
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    dataset = dataset_builder.build(
        input_reader_config,
        model_config,
        training,
        voxel_generator,
        target_assigner,
        multi_gpu=multi_gpu)
    
    cum_lc_config = input_reader_config.cum_lc_wrapper
    if cum_lc_config.lc_policy != "":  # this means that it needs to be wrapped
        dataset = CumLCDatasetWrapper(dataset,
                                      net,
                                      cum_lc_config.lc_policy,
                                      cum_lc_config.lc_horizon,
                                      cum_lc_config.init_lidar_num_beams,
                                      cum_lc_config.sparsify_return,
                                      use_cache=not training,
                                      contiguous=not training)
    else:
        dataset = LidarDatasetWrapper(dataset,
                                      num_workers=input_reader_config.preprocess.num_workers)
    
    return dataset
