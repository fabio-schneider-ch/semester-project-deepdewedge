#!/bin/bash
#SBATCH --job-name=ddw_fit_supermodel_02
#SBATCH --partition=gpu
#SBATCH --qos=gpu-default
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=94G
#SBATCH --time=48:00:00
#SBATCH --output=/das/work/p23/p23467/ext-schnei_f/outputs/trainings/supermodel_02/ddw_fit_%j.out
#SBATCH --error=/das/work/p23/p23467/ext-schnei_f/outputs/trainings/supermodel_02/ddw_fit_%j.err

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

echo "DDW:"
which ddw

CONFIG=/das/home/ext-schnei_f/coding/semester-project-deepdewedge/experiments/own_model/config_02.yaml

echo "Using config:"
echo $CONFIG

echo "Preparing data..."
ddw prepare-data --config $CONFIG

echo "Starting model fitting..."
ddw fit-model --config $CONFIG

echo "Job finished:"
date
