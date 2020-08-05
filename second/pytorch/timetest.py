# Example command:
# python ./pytorch/timetest.py all_exp --dataset=vkitti --model=second --exp_dir=sid_trained_models/interpDP/vkitti/second --lc_horizon=3 --num_examples=100

import copy
import json
import os
from pathlib import Path
import pickle
import shutil
import time
import re 
import fire
import numpy as np
import torch
from google.protobuf import text_format
from tqdm import trange

import second.data.kitti_common as kitti
import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.core import box_np_ops
from second.data.preprocess import merge_second_batch, merge_second_batch_multigpu
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.utils.log_tool import SimpleModelLog
from second.utils.progress_bar import ProgressBar
import psutil
from second.pytorch.train import build_network, example_convert_to_torch

def main(config_path,
         lc_horizon,
         num_examples,
         model_dir,
         ckpt_path=None,
         **kwargs):
    """Don't support pickle_result anymore. if you want to generate kitti label file,
    please use kitti_anno_to_label_file and convert_detection_to_kitti_annos
    in second.data.kitti_dataset.
    """
    assert len(kwargs) == 0
    model_dir = str(Path(model_dir).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to eval with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    input_cfg.cum_lc_wrapper.lc_horizon = lc_horizon
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, measure_time=False).to(device)
    if train_cfg.enable_mixed_precision:
        net.half()
        print("half inference!")
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator

    if ckpt_path is None:
        assert model_dir is not None
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)
    batch_size = 1
    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        net=net)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    t = time.time()
    detections = []
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)
    prep_example_times = []
    prep_times = []
    t2 = time.time()

    times = []
    for scene_id in trange(num_examples):
        idx = eval_dataset.scene_id_and_step_to_idx(scene_id, lc_horizon)
        torch.cuda.synchronize()
        b_ex_time = time.time()
        example = eval_dataset[idx]
        example = merge_second_batch([example])
        example = example_convert_to_torch(example, float_dtype)
        with torch.no_grad():
            detections = net(example)
        torch.cuda.synchronize()
        e_ex_time = time.time()
        del example, detections
        times.append(e_ex_time - b_ex_time)
    
    times = np.array(times)
    mean = times.mean()
    interval = 1.96 * times.std() / np.sqrt(len(times))  # 95% confidence interval

    return mean, interval
    # print(f"Time: {mean:.4f} ± {interval:.4f}", file=out_file_ptr, flush=True)

def all_exp(dataset, model, exp_dir, lc_horizon, num_examples=100):
    exp_dir = Path(exp_dir)
    results_file = exp_dir / "time_stats.txt"
    if results_file.is_file():
        os.remove(results_file)
    exp_names = [
        'base',
        'baselines/fixed_15',
        'baselines/fixed_30',
        'baselines/fixed_45',
        'baselines/gdp_mvel',
        'baselines/gdp_rand',
        'baselines/pllmaxent',
        'baselines/random',
    ]

    with open(results_file, 'a') as f:
        for exp_name in exp_names:
            print(f"\n\n{exp_name}:", file=f, flush=True)
            for curr_lc_horizon in range(lc_horizon + 1):
            
                mean, interval = main(config_path=f'configs/{dataset}/{model}/{exp_name}.yaml',
                                      lc_horizon=curr_lc_horizon,
                                      num_examples=num_examples,
                                      model_dir=exp_dir / exp_name)
                
                # Message to console.
                print(f"{exp_name}, {curr_lc_horizon}: {mean:.3f} ± {interval:.3f}")

                # Message to results file.
                print(f"& {mean:.3f} $\pm$ {interval:.3f} ", file=f, flush=True, end='')

if __name__ == '__main__':
    fire.Fire()