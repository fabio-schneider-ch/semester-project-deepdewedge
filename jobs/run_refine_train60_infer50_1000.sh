#!/bin/bash
#SBATCH --job-name=refine_t60_i50_1000
#SBATCH --partition=gpu
#SBATCH --qos=gpu-day
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/das/work/p23/p23467/ext-schnei_f/outputs/refinements/subexp02/train60_infer50_1000/refine_train60_infer50_1000_%j.out
#SBATCH --error=/das/work/p23/p23467/ext-schnei_f/outputs/refinements/subexp02/train60_infer50_1000/refine_train60_infer50_1000_%j.err

echo "Job started:"
date

echo "Running on node:"
hostname

echo "GPU info:"
nvidia-smi

source ~/.bashrc
conda activate /das/work/p23/p23467/ext-schnei_f/conda_envs/ddw_env

echo "Python:"
which python

CONFIG=/das/home/ext-schnei_f/coding/semester-project-deepdewedge/experiments/subexp02/refine_mw60/config.yaml

echo "Using config:"
echo $CONFIG

python /das/home/ext-schnei_f/coding/semester-project-deepdewedge/notebooks/refine_save_halves.py \
  --config $CONFIG

echo "Job finished:"
date
