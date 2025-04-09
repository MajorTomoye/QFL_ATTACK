import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from utils.qutils import QuantizedConv2d, QuantizedLinear, QuantizationEnabler
from networks.alexnet import AlexNet
from networks.vgg import VGG16


# ==== Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ==== Data transforms ====
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
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

wqmode = 'per_layer_symmetric'
aqmode = 'per_layer_asymmetric'

results = {}  # (model, dataset, method) -> acc trace

methods = ['fp32_8bit', '32bit_8bit_4bit']

# ==== Training ====
for model_name, ModelClass in model_types.items():
    for dataset_name, data_cfg in data_cfgs.items():
        print(f"\n>>> Running {model_name} on {dataset_name.upper()}...")
        train_loader = DataLoader(data_cfg['train'], batch_size=128, shuffle=True)
        test_loader = DataLoader(data_cfg['test'], batch_size=1000, shuffle=False)

        for method in methods:
            print(f"--> Method: {method}")
            model = ModelClass(num_classes=data_cfg['num_classes'], dataset=dataset_name).to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
            criterion = nn.CrossEntropyLoss()
            acc_trace = []
            model.train()

            for epoch in range(100):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    floss = criterion(outputs, y)
                    loss = floss

                    if method == 'fp32_8bit':
                        with QuantizationEnabler(model, wqmode, aqmode, 8, 1, epoch, True, False):
                            qoutputs = model(x)
                            qloss = criterion(qoutputs, y)
                            loss += qloss

                    elif method == '32bit_8bit_4bit':
                        for bit in [32, 8, 4]:
                            with QuantizationEnabler(model, wqmode, aqmode, bit, 1, epoch, True, False):
                                qoutputs = model(x)
                                qloss = criterion(qoutputs, y)
                                loss += qloss

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Test with 4-bit always
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        with QuantizationEnabler(model, wqmode, aqmode, 4, 1, epoch, True, False):
                            outputs = model(x)
                            _, predicted = torch.max(outputs, 1)
                            correct += (predicted == y).sum().item()
                            total += y.size(0)
                acc = correct / total
                acc_trace.append(acc)
                print(f"[{method}] Epoch {epoch+1}: Accuracy = {acc:.4f}")

            results[(model_name, dataset_name, method)] = acc_trace

# ==== Plotting ====
os.makedirs("plots/experiments_fp32_vs_dualbit", exist_ok=True)

for model_name in model_types:
    for dataset_name in data_cfgs:
        plt.figure(figsize=(10, 5))
        for method in methods:
            if method == 'fp32_8bit':
                label = 'FP32 + 8bit'
            elif method == '32bit_8bit_4bit':
                label = '32bit + 8bit + 4bit'
            plt.plot(results[(model_name, dataset_name, method)], label=label)
        plt.title(f"{model_name} on {dataset_name.upper()} (4bit Evaluation)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = f"plots/experiments_fp32_vs_dualbit/acc_{model_name.lower()}_{dataset_name.lower()}.png"
        plt.savefig(save_path, dpi=1600)
        plt.close()
