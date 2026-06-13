#!/bin/bash
#SBATCH -p csi
#SBATCH -A csibnl
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=Sympy_OPF_LALM_class_notebook_mu0605
#SBATCH --output=/direct/sdcc+u/zzhao3/Projects/Python_Script/OPF-python/logs/slurm-%j.out

source ~/Software/miniforge3/etc/profile.d/conda.sh
conda activate qhd_linux

cd /direct/sdcc+u/zzhao3/Projects/Python_Script/OPF-python

hostname
nvidia-smi

python Sympy_QrOPF_ALM_mu_final_2bus.py