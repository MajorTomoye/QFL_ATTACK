#!/bin/bash

# 检查是否提供了至少两个参数（global_lr 和 lr 是必需的）
if [ $# -lt 2 ]; then
    echo "Usage: $0 <global_lr> <lr> [model] [dataset]"
    exit 1
fi

# 读取命令行参数，并提供默认值
GLOBAL_LR=$1
LR=$2
MODEL=${3:-AlexNet}    # 默认值为 AlexNet
DATASET=${4:-cifar10}  # 默认值为 cifar10
IID=${5:-1}

# 依次执行两个命令
python attack_fedlearn.py --global_lr="$GLOBAL_LR" --lr="$LR" --epochs=2000 --malicious_users=0 --attmode normal --model="$MODEL" --dataset="$DATASET" --iid=$IID
python attack_fedlearn.py --global_lr="$GLOBAL_LR" --lr="$LR" --epochs=2000 --malicious_users=0 --attmode normal --qat --model="$MODEL" --dataset="$DATASET" --iid=$IID
