# Launch from second.pytorch/second

import os
from functools import partial

PREFIX_DIR = "sid_trained_models/long"

TRAIN_CMD_PREFIX = "python -u -W ignore::UserWarning ./pytorch/train.py train"
EVAL_CMD_PREFIX = "python -u -W ignore::UserWarning ./pytorch/train.py evaluate"
EVAL_INFO_PATH = {
    "vkitti":
        {
            "train": '$DATADIR/vkitti/vkitti_infos_train_random_subset.pkl',
            "test" : '$DATADIR/vkitti/vkitti_infos_test.pkl' 
        },
    "synthia":
        {
            "train": '$DATADIR/synthia/synthia_infos_train_random_subset.pkl',
            "test" : '$DATADIR/synthia/synthia_infos_test_2k_subset.pkl'
        }
}

def TRAIN_CMD(dataset, model, config):
    cmd = f"{TRAIN_CMD_PREFIX} " + \
          f"--config_path=./configs/{dataset}/{model}/{config}.yaml " + \
          f"--model_dir={PREFIX_DIR}/{dataset}/{model}/{config} " + \
          f"--display_step=100"

    return [cmd]

def EVAL_CMD(dataset, model, config):
    train_info_path = EVAL_INFO_PATH[dataset]["train"]
    test_info_path = EVAL_INFO_PATH[dataset]["test"]

    train_cmd = f"{EVAL_CMD_PREFIX} " + \
                f"--config_path=./configs/{dataset}/{model}/{config}.yaml " + \
                f"--model_dir={PREFIX_DIR}/{dataset}/{model}/{config} " + \
                f"--result_path={PREFIX_DIR}/{dataset}/{model}/{config}/eval_results/train " + \
                f"--info_path={train_info_path}"
    
    test_cmd = f"{EVAL_CMD_PREFIX} " + \
               f"--config_path=./configs/{dataset}/{model}/{config}.yaml " + \
               f"--model_dir={PREFIX_DIR}/{dataset}/{model}/{config} " + \
               f"--result_path={PREFIX_DIR}/{dataset}/{model}/{config}/eval_results/test " + \
               f"--info_path={test_info_path}"
    
    return [train_cmd, test_cmd]

def SBATCH_CMD(dataset, model, config, jobname, train, eval):
    jobs = []
    if train:
        jobs.extend(TRAIN_CMD(dataset, model, config))
    if eval:
        jobs.extend(EVAL_CMD(dataset, model, config))
    job = " && ".join(jobs)

    cmd = f"python sbatch.py launch --cmd=\"{job}\" --name={dataset}/{model}/{jobname}"
    return cmd


def LAUNCH_ALL_JOBS(dataset, model, train, eval):
    jobs = []
    
    CMD = partial(SBATCH_CMD, dataset=dataset, model=model, train=train, eval=eval)

    # BASE model
    jobs.append(CMD(config='base', jobname='base'))
    
    # BASELINES
    jobs.append(CMD(config='baselines/random',    jobname='b_random'))
    jobs.append(CMD(config='baselines/fixed_15',  jobname='b_fixed_15'))
    jobs.append(CMD(config='baselines/fixed_30',  jobname='b_fixed_30'))
    jobs.append(CMD(config='baselines/fixed_45',  jobname='b_fixed_45'))
    jobs.append(CMD(config='baselines/pllmaxent', jobname='b_pllmaxent'))
    jobs.append(CMD(config='baselines/gdp_mvel',  jobname='b_gdp_mvel'))
    jobs.append(CMD(config='baselines/gdp_rand',  jobname='b_gdp_rand'))

    # ABLATIONS
    jobs.append(CMD(config='ablations/sb_lidar_only', jobname='a_sb_lidar_only'))
    
    # MULTIBEAM LIDAR ONLY
    jobs.append(CMD(config='multibeam/beams2',  jobname='m_bms2'))
    jobs.append(CMD(config='multibeam/beams4',  jobname='m_bms4'))
    jobs.append(CMD(config='multibeam/beams8',  jobname='m_bms8'))
    jobs.append(CMD(config='multibeam/beams16', jobname='m_bms16'))
    jobs.append(CMD(config='multibeam/beams32', jobname='m_bms32'))
    jobs.append(CMD(config='multibeam/beams64', jobname='m_bms64'))

    # MULTIBEAM LIDAR + LC
    jobs.append(CMD(config='multibeam/beams2_lc',  jobname='m_bms2_lc'))
    jobs.append(CMD(config='multibeam/beams4_lc',  jobname='m_bms4_lc'))
    jobs.append(CMD(config='multibeam/beams8_lc',  jobname='m_bms8_lc'))
    jobs.append(CMD(config='multibeam/beams16_lc', jobname='m_bms16_lc'))
    jobs.append(CMD(config='multibeam/beams32_lc', jobname='m_bms32_lc'))
    jobs.append(CMD(config='multibeam/beams64_lc', jobname='m_bms64_lc'))

    for job in jobs:
        os.system(job)

def EVAL_DATASET_GENERALIZATION(model, src_dataset, tgt_dataset):
    # Model is loaded using model_dir          : use src_dataset
    # Config, result path is for target dataset: use tgt_dataset
    test_info_path = EVAL_INFO_PATH[tgt_dataset]["test"]
    cmd = f"{EVAL_CMD_PREFIX} " + \
          f"--config_path=./configs/{tgt_dataset}/{model}/base.yaml " + \
          f"--model_dir={PREFIX_DIR}/{src_dataset}/{model}/base " + \
          f"--result_path={PREFIX_DIR}/{tgt_dataset}/{model}/base/eval_results/test_using_model_trained_on_{src_dataset} " + \
          f"--info_path={test_info_path}"
    
    job = f"python sbatch.py launch --cmd=\"{cmd}\" --name={tgt_dataset}/{model}/cdg_from_{src_dataset}"
    os.system(job)

def EVAL_LC_GENERALIZATION(dataset, model):
    # Model is loaded using model_dir: use base
    # Config, result path is for 10LC: use base_10lc
    test_info_path = EVAL_INFO_PATH[dataset]["test"]
    cmd = f"{EVAL_CMD_PREFIX} " + \
          f"--config_path=./configs/{dataset}/{model}/base_10lc.yaml " + \
          f"--model_dir={PREFIX_DIR}/{dataset}/{model}/base " + \
          f"--result_path={PREFIX_DIR}/{dataset}/{model}/base_10lc/eval_results/test " + \
          f"--info_path={test_info_path}"
    
    job = f"python sbatch.py launch --cmd=\"{cmd}\" --name={dataset}/{model}/lcg"
    os.system(job)



# ========================== JOBS ==========================

# Train + Test
LAUNCH_ALL_JOBS('vkitti',  'second', train=True, eval=True)
LAUNCH_ALL_JOBS('synthia', 'second', train=True, eval=True)

# Cross dataset generalization
EVAL_DATASET_GENERALIZATION('second', 'vkitti', 'synthia')
EVAL_DATASET_GENERALIZATION('second', 'synthia', 'vkitti')

# Num- light curtains generalization.
EVAL_LC_GENERALIZATION('vkitti',  'second')
EVAL_LC_GENERALIZATION('synthia', 'second')
