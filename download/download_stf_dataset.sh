#!/bin/bash

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# dataset dir 下载到本地目录
dataset_dir="/root/autodl-tmp/dataset"

mkdir -p ${dataset_dir}

# 下载SFT数据集
huggingface-cli download \
  --repo-type dataset \
  --resume-download \
  BelleGroup/train_3.5M_CN \
  --local-dir "${dataset_dir}/BelleGroup"
