#!/bin/sh

set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model-name>"
    exit 1
fi

model_name="$1"
model_dir="models"
model_url="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/${model_name}.pkl"
model_path="${model_dir}/${model_name}.pkl"

mkdir -p "$model_dir"
wget -O "$model_path" "$model_url"
