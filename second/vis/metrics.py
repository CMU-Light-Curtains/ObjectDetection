import fire
from pathlib import Path
from itertools import groupby
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

difficulty = 1

def print_df(df):
        string = str(df)
        string = string.replace(",", " &")
        print(string)

def process_metric_file(path):
    dirs = list([e for e in path.iterdir() if e.is_dir()])
    assert len(dirs) == 1
    path = dirs[0] / "metrics.txt"

    store = {}
    with open(path, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    step_results = [list(group) for k, group in groupby(lines, lambda x: x == "") if not k]
    for step_result in step_results:
        step = step_result[0].split('_')[1]
        step_result = step_result[1:]
        assert len(step_result) == 10
        store[step] = {}
        for thresh_result in (step_result[:5], step_result[5:]):
            if thresh_result[0] == 'Car AP(Average Precision)@0.70, 0.70, 0.70:':
                thresh = 0.7
            elif thresh_result[0] == 'Car AP(Average Precision)@0.70, 0.50, 0.50:':
                thresh = 0.5
            else:
                raise Exception
            thresh_result = thresh_result[1:]
            
            store[step][thresh] = {}
            for line in thresh_result:
                line = line.split()
                assert len(line) == 4
                metric, value = line[0], line[difficulty + 1]
                store[step][thresh][metric] = value
    return store

def load_data(expdir, subexps):
    data = {sexp: {"train": {}, "test": {}} for sexp in subexps}

    for sexp, store in data.items():
        for split in ["train", "test"]:
            path = expdir / sexp / "eval_results" / split
            metrics = process_metric_file(path)
            store[split] = metrics
    return data

def main_results_table(data, split):
    df_data = []
    for sexp, step in [["ablations/sb_lidar_only", "step0"],
                       ["base", "step0"],
                       ["base", "step1"],
                       ["base", "step2"],
                       ["base", "step3"]]:
        d = data[sexp][split]
        name = sexp.replace("ablations/", "a_").replace("baselines/", "b_") + "_" + step
        vals = [d[step][thresh][metric] for metric in ['3d', 'bev'] for thresh in [0.5, 0.7]]
        df_data.append([name] + vals)

    columns = [f"{metric}({thresh})" for metric in ['3d', 'bev'] for thresh in [0.5, 0.7]]
    df = pd.DataFrame(df_data, columns=[""] + columns)
    print_df(df)

def baselines_table(data, split):
    df_data = []
    step = "step1"
    for sexp in ["baselines/random",
                 "baselines/fixed_15",
                 "baselines/fixed_30",
                 "baselines/fixed_45",
                 "baselines/gdp_mvel",
                 "baselines/gdp_rand",
                 "baselines/pllmaxent",
                 "base"]:
        d = data[sexp][split]
        name = sexp.replace("ablations/", "a_").replace("baselines/", "b_") + "_" + step
        vals = [d[step][thresh][metric] for metric in ['3d', 'bev'] for thresh in [0.5, 0.7]]
        df_data.append([name] + vals)

    columns = [f"{metric}({thresh})" for metric in ['3d', 'bev'] for thresh in [0.5, 0.7]]
    df = pd.DataFrame(df_data, columns=[""] + columns)
    print_df(df)

def main_results_and_baselines(expdir):
    expdir = Path(expdir)
    subexps = [
        "base",
        "ablations/sb_lidar_only",
        "baselines/random",
        "baselines/fixed_15",
        "baselines/fixed_30",
        "baselines/fixed_45",
        "baselines/gdp_mvel",
        "baselines/gdp_rand",
        "baselines/pllmaxent"
    ]
    data = load_data(expdir, subexps)

    print("===== TRAIN =====")
    print("Main Results:")
    main_results_table(data, "train")
    print("\nBaselines:")
    baselines_table(data, "train")

    print("\n\n===== TEST =====")
    print("Main Results:")
    main_results_table(data, "test")
    print("\nBaselines:")
    baselines_table(data, "test")

def _multibeam_table(data, subexps, split):
    df_data = []
    for sexp in subexps:
        step = "step0"
        d = data[sexp][split]
        name = sexp.replace("multibeam/", "m_") + "_" + step
        vals = [d[step][thresh][metric] for metric in ['3d', 'bev'] for thresh in [0.5, 0.7]]
        df_data.append([name] + vals)
    
    columns = [f"{metric}({thresh})" for metric in ['3d', 'bev'] for thresh in [0.5, 0.7]]
    df = pd.DataFrame(df_data, columns=[""] + columns)
    print_df(df)

def multibeam(expdir):
    expdir = Path(expdir)
    subexps = [
        "multibeam/beams2",
        "multibeam/beams4",
        "multibeam/beams8",
        "multibeam/beams16",
        "multibeam/beams32",
        "multibeam/beams64"
    ]
    data = load_data(expdir, subexps)
    
    print("==== TRAIN ====")
    _multibeam_table(data, subexps, "train")

    print("==== TEST ====")
    _multibeam_table(data, subexps, "test")


def cross_dataset_gen(dataset):
    if dataset == 'vkitti':
        base_path = "sid_trained_models/interpDP/vkitti/second/base/eval_results/test"
        dgen_path = "sid_trained_models/interpDP/synthia/second/base/eval_results/test_using_model_trained_on_vkitti"
    elif dataset == 'synthia':
        base_path = "sid_trained_models/interpDP/synthia/second/base/eval_results/test"
        dgen_path = "sid_trained_models/interpDP/vkitti/second/base/eval_results/test_using_model_trained_on_synthia"
    
    base_data = process_metric_file(Path(base_path))
    dgen_data = process_metric_file(Path(dgen_path))

    df_data = []

    columns, col_id = [], -1
    matrix = np.zeros([4, 4], dtype='object')
    steps = ['step0', 'step1', 'step2', 'step3']
    for metric in ['3d', 'bev']:
        for thresh in [0.5, 0.7]:
            columns.append(f"{metric}({thresh})")
            col_id += 1

            for i, step in enumerate(steps):
                base_acc = float(base_data[step][thresh][metric][:-1])
                dgen_acc = float(dgen_data[step][thresh][metric][:-1])
                diff = base_acc - dgen_acc
                matrix[i, col_id] = str(f"{base_acc:.2f}, {dgen_acc:.2f}, {diff:.2f}")
                # import IPython; IPython.embed()

    for i, step in enumerate(steps):
        df_data.append([step] + list(matrix[i]))
    
    df = pd.DataFrame(df_data, columns=[""] + columns)
    print(df)


def num_lc_gen(expdir, dark=False):
    path = Path(expdir) / 'base_10lc/eval_results/test'
    data = process_metric_file(path)
    data = {int(k[4:]):v for k, v in data.items()}
    xs = sorted(data.keys())
    xticks = ['SBL'] + [f'L{i}' for i in range(1, 11)]

    if dark:
        plt.style.use('dark_background')

    fig, ax = plt.subplots()

    if dark:
        ax.set_facecolor('white')

    def draw_plot(metric, thresh, c, label):
        ys = [float(data[x][float(thresh)][metric][:-1]) for x in xs]
        ax.plot(xs, ys, c=c, label=label)
        ax.scatter(xs, ys, c=c)

    draw_plot('3d',  0.5, c='green', label='3d (0.5)')
    draw_plot('3d',  0.7, c='blue', label='3d (0.7)')
    draw_plot('bev', 0.5, c='magenta', label='bev (0.5)')
    draw_plot('bev', 0.7, c='red', label='bev (0.7)')

    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    rect1 = matplotlib.patches.Rectangle([x1, y1], 3-x1, y2-y1, color='blue',  alpha=0.08)
    rect2 = matplotlib.patches.Rectangle([3 , y1], x2-3, y2-y1, color='green', alpha=0.08)
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    if "vkitti" in str(path):
        ax.text(0.75, 15, r'Train', {'color': 'blue', 'fontsize': 30})
        ax.text(3.25, 15, r'Test',  {'color': 'green', 'fontsize': 30})
    elif "synthia" in str(path):
        ax.text(0.75, 47.5, r'Train', {'color': 'blue', 'fontsize': 30})
        ax.text(3.25, 47.5, r'Test',  {'color': 'green', 'fontsize': 30})
    else:
        raise Exception

    plt.legend(fontsize='x-large')
    plt.xticks(xs, xticks, fontsize='x-large', rotation=45)
    plt.yticks(fontsize='x-large')
    plt.tight_layout()
    plt.savefig(path / 'fig.png', format='png')

if __name__ == '__main__':
    fire.Fire()