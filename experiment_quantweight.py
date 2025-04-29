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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ==== CIFAR-10 数据集加载（带数据增强） ====
apply_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

apply_transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=apply_transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=apply_transform_valid)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ==== 使用框架自带 AlexNet 模型 ====
nclasses = 10
model = AlexNet(num_classes=nclasses, dataset='cifar10').to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)

# ==== 记录 conv1[0, 0, :, :] 权重的震荡轨迹 ====
conv1_layer = model.features[0]  # Conv1.0 层
weight_trace = [[] for _ in range(9)]  # 记录每个位置
quant_trace = [[] for _ in range(9)]

# ==== 记录准确率 ====
acc_trace = []

wqmode = 'per_layer_symmetric'
aqmode = 'per_layer_asymmetric'
bit_size = 4

# ==== 训练并记录 ====
model.train()
for epoch in range(2000):
    print(f"Epoch {epoch + 1}/2000")
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        floss = criterion(outputs, y)
        loss = floss
        with QuantizationEnabler(model, wqmode, aqmode, bit_size, 1,epoch, silent=True, fixed=True):
            qoutput = model(x)
            qloss = criterion(qoutput, y)
            loss += qloss
            latent_weights = conv1_layer.weight.data[0, 0].detach().cpu().numpy().reshape(-1)
            quant_weights = conv1_layer.weight_quantizer(conv1_layer.weight.data)[0, 0].detach().cpu().numpy().reshape(-1)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        for i in range(9):
            weight_trace[i].append(latent_weights[i])
            quant_trace[i].append(quant_weights[i])


    # 每轮评估一次准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            with QuantizationEnabler(model, wqmode, aqmode, bit_size, 1,epoch=1, silent=True, fixed=True):
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
    acc = correct / total
    acc_trace.append(acc)
    print(f"  当前准确率: {acc:.4f}")

# # ==== 最后测试精度 ====
# def test(model):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for x, y in test_loader:
#             x, y = x.to(device), y.to(device)
#             with QuantizationEnabler(model, wqmode, aqmode, bit_size,1, epoch=1, silent=True, fixed=True):
#                 outputs = model(x)
#                 _, predicted = torch.max(outputs, 1)
#                 correct += (predicted == y).sum().item()
#                 total += y.size(0)
#     return correct / total

# acc = test(model)
# print(f"[QAT模型] 测试准确率: {acc:.4f}")

# ==== 可视化 ====
output_dir = "plots/experiment_quantweight"
os.makedirs(output_dir, exist_ok=True)

start = -1000  # 只绘制最后1000轮

# 权重对比图（上下子图形式）
fig, axs = plt.subplots(2, 1, figsize=(10, 8), dpi=1600, sharex=True)

# 上图：量化权重
for i in range(9):
    axs[0].plot(quant_trace[i][start:], label=f'w{i}')
axs[0].set_title("Quantized Integer Weights (Conv1[0,0,:,:]) - Last 1000 Epochs")
axs[0].set_ylabel("Quantized Value")
axs[0].legend(fontsize=10)
axs[0].grid(True)

# 下图：浮点权重
for i in range(9):
    axs[1].plot(weight_trace[i][start:], label=f'w{i}')
axs[1].set_title("Latent Shadow Weights (Conv1[0,0,:,:]) - Last 1000 Epochs")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Latent Value")
axs[1].legend(fontsize=10)
axs[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "conv1_weight_comparison_split.png"))
plt.close()

# 准确率图
plt.figure(figsize=(8, 4), dpi=1600)
plt.plot(acc_trace[:], label="Accuracy", color="blue")
plt.title("Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "validation_accuracy.png"))
plt.close()

print(f"图像已保存到: {output_dir}")
