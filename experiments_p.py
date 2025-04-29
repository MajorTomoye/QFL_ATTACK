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
    transforms.Normalize((0.5,), (0.5,))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
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

bit_size = 8
wqmode = 'per_layer_symmetric'
aqmode = 'per_layer_asymmetric'

results = {}

# ==== 主循环：模型 × 数据集 × clip_method ====
for model_name, ModelClass in model_types.items():
    for dataset_name, data_cfg in data_cfgs.items():
        print(f"\n>>> Running {model_name} on {dataset_name.upper()}...")

        train_loader = DataLoader(data_cfg['train'], batch_size=128, shuffle=True)
        test_loader = DataLoader(data_cfg['test'], batch_size=1000, shuffle=False)

        for clip_method in [0, 1]:
            print(f"--> clip_method={clip_method}")
            model = ModelClass(num_classes=data_cfg['num_classes'], dataset=dataset_name).to(device)
            optimizer = optim.SGD(model.parameters(),lr=0.05,momentum=0.5)
            criterion = nn.CrossEntropyLoss()

            acc_trace = []
            model.train()
            for epoch in range(200):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    floss = criterion(outputs, y)
                    loss = floss
                    with QuantizationEnabler(model, wqmode, aqmode, bit_size, clip_method=clip_method, epoch=epoch, silent=True, fixed=False):
                        qoutputs = model(x)
                        qloss = criterion(qoutputs, y)
                        loss += qloss
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 测试
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        with QuantizationEnabler(model, wqmode, aqmode, bit_size, clip_method=clip_method, epoch=epoch, silent=True, fixed=False):
                            outputs = model(x)
                            _, predicted = torch.max(outputs, 1)
                            correct += (predicted == y).sum().item()
                            total += y.size(0)
                acc = correct / total
                acc_trace.append(acc)
                print(f"[Clip {clip_method}] Epoch {epoch+1}: Accuracy = {acc:.4f}")

            results[(model_name, dataset_name, clip_method)] = acc_trace

# ==== 绘图 ==== 
os.makedirs("plots/experiments_p", exist_ok=True)


# 图1：AlexNet 两数据集 clip_method 对比
plt.figure(figsize=(10, 5))
for dataset in data_cfgs:
    for clip_method in [0, 1]:
        key = ("AlexNet", dataset, clip_method)
        label = f"{dataset.upper()} (clip={clip_method})"
        plt.plot(results[key], label=label)

plt.title("AlexNet: Accuracy on CIFAR-10 & SVHN", fontsize=18)
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/experiments_p/acc_alexnet.png", dpi=1600)
plt.savefig("plots/experiments_p/acc_alexnet.pdf", dpi=1600)
plt.close()

# 图2：VGG16 两数据集 clip_method 对比
plt.figure(figsize=(10, 5))
for dataset in data_cfgs:
    for clip_method in [0, 1]:
        key = ("VGG16", dataset, clip_method)
        label = f"{dataset.upper()} (clip={clip_method})"
        plt.plot(results[key], label=label)

plt.title("VGG16: Accuracy on CIFAR-10 & SVHN", fontsize=18)
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/experiments_p/acc_vgg16.png", dpi=1600)
plt.savefig("plots/experiments_p/acc_vgg16.pdf", dpi=1600)
plt.close()
# 图1：AlexNet 两数据集 clip_method 对比

# plt.figure(figsize=(10, 5))
# for dataset in data_cfgs:
#     for clip_method in [0, 1]:
#         key = ("AlexNet", dataset, clip_method)
#         label = f"{dataset.upper()} (clip={clip_method})"
#         plt.plot(results[key], label=label)
# plt.title("AlexNet: Accuracy on CIFAR-10 & SVHN")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plots/experiments_p/acc_alexnet.png", dpi=1600)
# plt.close()

# # 图2：VGG16 两数据集 clip_method 对比
# plt.figure(figsize=(10, 5))
# for dataset in data_cfgs:
#     for clip_method in [0, 1]:
#         key = ("VGG16", dataset, clip_method)
#         label = f"{dataset.upper()} (clip={clip_method})"
#         plt.plot(results[key], label=label)
# plt.title("VGG16: Accuracy on CIFAR-10 & SVHN")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plots/experiments_p/acc_vgg16.png", dpi=1600)
# plt.close()

# # ==== 记录准确率 ====
# acc_trace = []

# wqmode = 'per_layer_symmetric'
# aqmode = 'per_layer_asymmetric'
# bit_size = 8
# conv1_weight_history = []
# # ==== 训练并记录 ====
# model.train()
# for epoch in range(50):
#     print(f"Epoch {epoch + 1}/50")
#     for x, y in train_loader:
#         x, y = x.to(device), y.to(device)
        
#         outputs = model(x)
#         floss = criterion(outputs, y)
#         loss = floss
#         with QuantizationEnabler(model, wqmode, aqmode, bit_size,clip_method=0, epoch=epoch, silent=True, fixed=False):
#             output = model(x)
#             qloss = criterion(output, y)
#             loss += qloss
#         model.zero_grad()
#         loss.backward()
#         optimizer.step()


#     for module in reversed(model.features):
#         if isinstance(module, QuantizedConv2d):
#             conv1_weight_history.append(module.weight[0, 0].detach().cpu().clone())  # shape (3, 3)
#             break  # 只记录第一层




#     # 每轮评估一次准确率
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for x, y in test_loader:
#             x, y = x.to(device), y.to(device)
#             with QuantizationEnabler(model, wqmode, aqmode, bit_size, clip_method=0,epoch=epoch, silent=True, fixed=False):
#                 outputs = model(x)
#                 _, predicted = torch.max(outputs, 1)
#                 correct += (predicted == y).sum().item()
#                 total += y.size(0)
#     acc = correct / total
#     acc_trace.append(acc)
#     print(f"  当前准确率: {acc:.4f}")
# # 将 conv1 权重历史转为 (num_epochs, 3, 3) 的张量
# conv1_array = torch.stack(conv1_weight_history)  # shape: [E, 3, 3]
# print("conv1_array.shape:", conv1_array.shape)
# neuron_traces = conv1_array.view(len(conv1_weight_history), -1).numpy().T  # shape: [9, E]
# print("neuron_traces.shape:", neuron_traces.shape)
# assert len(conv1_weight_history) == 50, f"Expected 50 epochs, got {len(conv1_weight_history)}"
# # 绘图
# plt.figure(figsize=(10, 6))
# colors = plt.cm.tab10.colors  # 取前 10 种颜色
# for i in range(9):
#     print(f"neuron_traces[{i}]:", neuron_traces[i])
#     print('\n')
#     plt.plot(range(len(neuron_traces[i])), neuron_traces[i], label=f'Neuron {i}', color=colors[i % len(colors)])

# plt.xlabel("Epoch")
# plt.ylabel("Weight Value")
# plt.title("Conv1[0,0,:,:] Kernel Weights Over Epochs")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plots/experiments_weight/conv1_kernel_trace.png", dpi=300)
# plt.close()



# # ==== 绘制准确率变化图 ====
# os.makedirs("plots/experiments_weight", exist_ok=True)
# plt.figure(figsize=(8, 4))
# plt.plot(acc_trace, label="Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Training Accuracy Curve")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("plots/experiments_weight/accuracy_curve.png", dpi=300)
# plt.close()



# # ==== 最后测试精度 ====
# def test(model):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for x, y in test_loader:
#             x, y = x.to(device), y.to(device)
#             with QuantizationEnabler(model, wqmode, aqmode, bit_size, epoch=1, silent=True, fixed=False):
#                 outputs = model(x)
#                 _, predicted = torch.max(outputs, 1)
#                 correct += (predicted == y).sum().item()
#                 total += y.size(0)
#     return correct / total

# acc = test(model)
# print(f"[QAT模型] 测试准确率: {acc:.4f}")