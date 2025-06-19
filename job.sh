#!/bin/bash

#PBS -N lemon
#PBS -l select=1:ncpus=16:mem=64gb:ngpus=1:host=cvml04
#PBS -l walltime=72:00:00
#PBS -q workq
cd /home/yiyang/AffordHuman
date
source activate lemon
# python eval.py --yaml config/eval.yaml
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py --save_checkpoint_path runs/test/ \
#     --batch_size 16 --yaml config/train.yaml

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py --save_checkpoint_path runs/no_cur/ \
    --batch_size 12 --yaml config/train.yaml
# CUDA_VISIBLE_DEVICES=0 python inference.py