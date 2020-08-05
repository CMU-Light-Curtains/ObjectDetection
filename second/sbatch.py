"""
Example command on Seuss:

python sbatch.py launch
    --cmd="cd second && python -u ./pytorch/train.py train
                            --config_path=./configs/vkitti/lc_cum.config.yaml
                            --model_dir=sid_trained_models/vkitti/full_trainset/DPOptimizedPolicy/sparse_sbl_rate_0pc
                            --display_step=100"
    --name="base"

With no spaces:

python sbatch.py launch --cmd="cd second && python -u ./pytorch/train.py train --config_path=./configs/vkitti/lc_cum.config.yaml --model_dir=sid_trained_models/vkitti/full_trainset/DPOptimizedPolicy/sparse_sbl_rate_0pc --display_step=100" --name="base"
"""

import fire
import os

#SBATCH --exclude=compute-0-[5,7,9,11,25,27]
#SBATCH --exclude=compute-0-[5,7,9,11]
#SBATCH --exclude=compute-0-[5,9]
#SBATCH --exclude=compute-0-[5]

def launch(name, cmd):
    sbatch_header = f"""#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --exclude=compute-0-[5,7,9,11,13]
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -o /home/sancha/sout/{name}.txt
#SBATCH -e /home/sancha/sout/{name}.txt

srun hostname

module load singularity
"""
    
    singularity_cmd = \
f"""
singularity exec --nv --bind /opt:/opt --bind /opt/cuda/10.0:/usr/local/cuda /home/sancha/sid_16_04.sif bash -c "source ~/.bashrc && {cmd}"
"""
    
    sbatch_cmd = sbatch_header + singularity_cmd
    
    # Create temporary file.
    tmp_fname = f".tmp.{name}.sh"
    with open(tmp_fname, 'w') as f:
        print(sbatch_cmd, file=f)

    os.system(f'cat {tmp_fname}')
    os.system(f'sbatch {tmp_fname}')

if __name__ == '__main__':
    fire.Fire()

