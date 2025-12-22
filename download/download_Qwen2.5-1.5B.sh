#!/bin/bash

# 设置环境变量
# export HF_ENDPOINT=https://hf-mirror.com

# dataset dir 下载到本地目录
model_dir="/root/autodl-tmp/model"

mkdir -p ${dataset_dir}

# 下载SFT数据集
huggingface-cli download \
  --resume-download \
  Qwen/Qwen2.5-1.5B \
  --local-dir "${model_dir}/Qwen2.5-1.5B"
