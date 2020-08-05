import csv
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from pathlib import Path
import os
from easydict import EasyDict as edict
import json
import yaml

json_data = {}
def load_contents(dir, split, difficulty, overlap, step):
    if dir not in json_data:
        # Load json.
        with open(Path(dir) / "log.json.lst", "r") as f:
            lines = [line.rstrip() for line in f.readlines()]
            lines = [json.loads(line) for line in lines]
            lines = [line for line in lines if "eval.kitti" in line]
            json_data[dir] = lines

    lines = json_data[dir]

    key1 = f"{split}_step{step}_official"
    key2 = f"Car"
    key3 = f"3d@{overlap}0"

    train_iters, vals = [], []
    for i, line in enumerate(lines):
        # train_iter
        if "step" in line:
            train_iter = line["step"]
        else:
            train_iter = i // 2
        
        # val
        eval_kitti = line["eval.kitti"]
        if key1 not in eval_kitti: continue
        val = eval_kitti[key1][key2][key3][difficulty]
        
        train_iters.append(train_iter)
        vals.append(val)
    
    contents = np.array([train_iters, vals]).T  # (N, 2)

    return contents

def create_plotSeq_from_cfg(cfg):
    plotSeq = []
    plot_attr = lambda dir_cfg, attr: dir_cfg[attr] if attr in dir_cfg else cfg.defaults[attr]
    for dir_cfg in cfg.dirs:
        split      = plot_attr(dir_cfg, "split")
        difficulty = plot_attr(dir_cfg, "difficulty")
        overlap    = plot_attr(dir_cfg, "overlap")
        steps      = plot_attr(dir_cfg, "steps")

        if not isinstance(steps, list):
            steps = [steps]

        for step in steps:
            plotItem = edict()
            plotItem.dir = cfg.dir_prefix + "/" + dir_cfg.dir
            plotItem.fname = f"run-summary-tag-eval.kitti_{split}_step{step}_official_Car_3d_{overlap}0_{difficulty}.csv"                   
            plotItem.contents = load_contents(plotItem.dir, split, difficulty, overlap, step)
            plotItem.split = split
            plotItem.difficulty = difficulty
            plotItem.overlap = overlap
            plotItem.step = step
            plotSeq.append(plotItem)
    return plotSeq
        
def plot_lines(cfg):
    plotSeq = create_plotSeq_from_cfg(cfg)

    colors = sns.cubehelix_palette(len(plotSeq))
    for i, plotItem in enumerate(plotSeq):
        contents, label, color = plotItem.contents, cfg.labels[i], colors[i]
        plt.plot(contents[:, 0], contents[:, 1], c=color, label=label, linewidth=4)

    plt.legend(loc='upper center', prop={'size': 14}, bbox_to_anchor=(0.5, -0.03),
          fancybox=True, shadow=True, ncol=5)
    
    title = f"{'Train' if cfg.defaults.split == 'valid' else 'Test'} ({cfg.defaults.overlap})"
    if cfg.title != "":
        title = f"{cfg.title}: {title}"
        plt.title(title, fontsize=20)

    plt.show()

if __name__ == '__main__':
    with open("vis/spec.yaml", "r") as f:
        cfg = edict(yaml.load(f))
    
    plot_lines(cfg)
    
    