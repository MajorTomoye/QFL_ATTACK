import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # 使用无GUI后端
import matplotlib.pyplot as plt
import os
import numpy as np

# ==== 你自己的量化组件 ====
from utils.qutils import QuantizedConv2d, QuantizedLinear, QuantizationEnabler  # 替换为你自己的模块名
from networks.alexnet import AlexNet
from networks.vgg import VGG16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ==== 通用数据预处理 ====
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

data_cfgs = {
    'cifar10': {
        'train': datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train),
        'test': datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test),
        'num_classes': 10
    },
    'svhn': {
        'train': datasets.SVHN(root='./data', split='train', download=True, transform=transform_train),
        'test': datasets.SVHN(root='./data', split='test', download=True, transform=transform_test),
        'num_classes': 10
    }
}

model_types = {
    'AlexNet': AlexNet,
    'VGG16': VGG16
}

bit_options = [4, 8, 32]
wqmode = 'per_layer_symmetric'
aqmode = 'per_layer_asymmetric'
clip_method = 1

results = {}

# ==== 主循环：模型 × 数据集 × bit_size ====
for model_name, ModelClass in model_types.items():
    for dataset_name, data_cfg in data_cfgs.items():
        print(f"\n>>> Running {model_name} on {dataset_name.upper()}...")

        train_loader = DataLoader(data_cfg['train'], batch_size=128, shuffle=True)
        test_loader = DataLoader(data_cfg['test'], batch_size=1000, shuffle=False)

        for bit_size in bit_options:
            print(f"--> bit_size={bit_size}")
            model = ModelClass(num_classes=data_cfg['num_classes'], dataset=dataset_name).to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
            criterion = nn.CrossEntropyLoss()
            if bit_size==4:
                fixed = True
            else:
                fixed = False
            acc_trace = []
            model.train()
            for epoch in range(100):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    floss = criterion(outputs, y)
                    loss = floss
                    with QuantizationEnabler(model, wqmode, aqmode, bit_size, clip_method, epoch, True, fixed):
                        qoutputs = model(x)
                        qloss = criterion(qoutputs, y)
                        loss += qloss
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()

                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        with QuantizationEnabler(model, wqmode, aqmode, bit_size, clip_method, epoch, True, fixed):
                            outputs = model(x)
                            _, predicted = torch.max(outputs, 1)
                            correct += (predicted == y).sum().item()
                            total += y.size(0)
                acc = correct / total
                acc_trace.append(acc)
                print(f"[Bit {bit_size}] Epoch {epoch+1}: Accuracy = {acc:.4f}")

            results[(model_name, dataset_name, bit_size)] = acc_trace

# ==== 绘图 ====
os.makedirs("plots/experiments_ste_compare", exist_ok=True)


# 公共字体设置
title_fontsize = 18
label_fontsize = 16
tick_fontsize = 14
legend_fontsize = 14

# 图1：AlexNet on CIFAR-10
plt.figure(figsize=(10, 5))
for bit in bit_options:
    if bit == 8:
        label = "WP-QAT"
    elif bit == 32:
        label = "QNs"
    else:
        label = "Traditional STE"
    plt.plot(results[("AlexNet", "cifar10", bit)], label=label)
plt.title("AlexNet on CIFAR-10", fontsize=title_fontsize)
plt.xlabel("Epoch", fontsize=label_fontsize)
plt.ylabel("Accuracy", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/experiments_ste_compare/acc_alexnet_cifar10.png", dpi=1600)
plt.savefig("plots/experiments_ste_compare/acc_alexnet_cifar10.pdf", dpi=1600)
plt.close()

# 图2：AlexNet on SVHN
plt.figure(figsize=(10, 5))
for bit in bit_options:
    if bit == 8:
        label = "WP-QAT"
    elif bit == 32:
        label = "QNs"
    else:
        label = "Traditional STE"
    plt.plot(results[("AlexNet", "svhn", bit)], label=label)
plt.title("AlexNet on SVHN", fontsize=title_fontsize)
plt.xlabel("Epoch", fontsize=label_fontsize)
plt.ylabel("Accuracy", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/experiments_ste_compare/acc_alexnet_svhn.png", dpi=1600)
plt.savefig("plots/experiments_ste_compare/acc_alexnet_svhn.pdf", dpi=1600)
plt.close()

# 图3：VGG16 on CIFAR-10
plt.figure(figsize=(10, 5))
for bit in bit_options:
    if bit == 8:
        label = "WP-QAT"
    elif bit == 32:
        label = "QNs"
    else:
        label = "Traditional STE"
    plt.plot(results[("VGG16", "cifar10", bit)], label=label)
plt.title("VGG16 on CIFAR-10", fontsize=title_fontsize)
plt.xlabel("Epoch", fontsize=label_fontsize)
plt.ylabel("Accuracy", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/experiments_ste_compare/acc_vgg16_cifar10.png", dpi=1600)
plt.savefig("plots/experiments_ste_compare/acc_vgg16_cifar10.pdf", dpi=1600)
plt.close()

# 图4：VGG16 on SVHN
plt.figure(figsize=(10, 5))
for bit in bit_options:
    if bit == 8:
        label = "WP-QAT"
    elif bit == 32:
        label = "QNs"
    else:
        label = "Traditional STE"
    plt.plot(results[("VGG16", "svhn", bit)], label=label)
plt.title("VGG16 on SVHN", fontsize=title_fontsize)
plt.xlabel("Epoch", fontsize=label_fontsize)
plt.ylabel("Accuracy", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/experiments_ste_compare/acc_vgg16_svhn.png", dpi=1600)
plt.savefig("plots/experiments_ste_compare/acc_vgg16_svhn.pdf", dpi=1600)
plt.close()


# ==== 图5：AlexNet 合并图 ====
plt.figure(figsize=(16, 5))

# 子图1：AlexNet on CIFAR-10
plt.subplot(1, 2, 1)
for bit in bit_options:
    if bit == 8:
        label = "WP-QAT"
    elif bit == 32:
        label = "QNs"
    else:
        label = "Traditional STE"
    plt.plot(results[("AlexNet", "cifar10", bit)], label=label)
plt.title("AlexNet on CIFAR-10", fontsize=title_fontsize)
plt.xlabel("Epoch", fontsize=label_fontsize)
plt.ylabel("Accuracy", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)

# 子图2：AlexNet on SVHN
plt.subplot(1, 2, 2)
for bit in bit_options:
    if bit == 8:
        label = "WP-QAT"
    elif bit == 32:
        label = "QNs"
    else:
        label = "Traditional STE"
    plt.plot(results[("AlexNet", "svhn", bit)], label=label)
plt.title("AlexNet on SVHN", fontsize=title_fontsize)
plt.xlabel("Epoch", fontsize=label_fontsize)
plt.ylabel("Accuracy", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)

plt.tight_layout()
plt.savefig("plots/experiments_ste_compare/acc_alexnet_combined.png", dpi=1600)
plt.savefig("plots/experiments_ste_compare/acc_alexnet_combined.pdf", dpi=1600)
plt.close()


# ==== 图6：VGG16 合并图 ====
plt.figure(figsize=(16, 5))

# 子图1：VGG16 on CIFAR-10
plt.subplot(1, 2, 1)
for bit in bit_options:
    if bit == 8:
        label = "WP-QAT"
    elif bit == 32:
        label = "QNs"
    else:
        label = "Traditional STE"
    plt.plot(results[("VGG16", "cifar10", bit)], label=label)
plt.title("VGG16 on CIFAR-10", fontsize=title_fontsize)
plt.xlabel("Epoch", fontsize=label_fontsize)
plt.ylabel("Accuracy", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)

# 子图2：VGG16 on SVHN
plt.subplot(1, 2, 2)
for bit in bit_options:
    if bit == 8:
        label = "WP-QAT"
    elif bit == 32:
        label = "QNs"
    else:
        label = "Traditional STE"
    plt.plot(results[("VGG16", "svhn", bit)], label=label)
plt.title("VGG16 on SVHN", fontsize=title_fontsize)
plt.xlabel("Epoch", fontsize=label_fontsize)
plt.ylabel("Accuracy", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)

plt.tight_layout()
plt.savefig("plots/experiments_ste_compare/acc_vgg16_combined.png", dpi=1600)
plt.savefig("plots/experiments_ste_compare/acc_vgg16_combined.pdf", dpi=1600)
plt.close()
