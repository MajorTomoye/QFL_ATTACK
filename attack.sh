#!/bin/bash

# 检查是否提供了至少两个参数（global_lr 和 lr 是必需的）
if [ $# -lt 2 ]; then
    echo "Usage: $0 <global_lr> <lr> [model] [dataset]"
    exit 1
fi

# 读取命令行参数，并提供默认值
GLOBAL_LR=$1
LR=$2
LR_a=$3
MODEL=${4:-AlexNet}    # 默认值为 AlexNet
DATASET=${5:-cifar10}  # 默认值为 cifar10


# 依次执行两个命令
python attack_fedlearn.py --global_lr="$GLOBAL_LR" --lr="$LR" --lr_attack="$LR_a" --epochs=3000 --malicious_users=5 --attmode="malicious"  --qat --model="$MODEL" --dataset="$DATASET"  --model_replace_attack --optimizer="SGD"
python attack_fedlearn.py --global_lr="$GLOBAL_LR" --lr="$LR" --lr_attack="$LR_a" --epochs=3000 --malicious_users=5 --attmode="malicious"  --qat --model="$MODEL" --dataset="$DATASET"  --fixed --lmethod=0 --optimizer="SGD"


#python attack_fedlearn.py --global_lr=12 --lr=0.0005 --lr_attack=0.0005 --epochs=3000 --malicious_users=5 --attmode="malicious"  --qat   --fixed --lmethod=0 

#python attack_fedlearn.py --global_lr=12 --lr=0.005 --lr_attack=0.005 --epochs=3000 --malicious_users=5 --attmode="malicious" --qat --model_replace_attack 