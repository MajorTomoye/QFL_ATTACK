import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

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
        'svhn': {
        'train': datasets.SVHN(root='./data', split='train', download=True, transform=transform_train),
        'test': datasets.SVHN(root='./data', split='test', download=True, transform=transform_test),
        'num_classes': 10
    },

    'cifar10': {
        'train': datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train),
        'test': datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test),
        'num_classes': 10
    },

}

model_types = {
    'VGG16': VGG16,
    'AlexNet': AlexNet
}

wqmode = 'per_layer_symmetric'
aqmode = 'per_layer_asymmetric'

results = {}  # (model, dataset, method, bit) -> acc trace
methods = [ 'CBS_VALW','multi_bit']

bit_set = [8,4]
valw_gamma = 3
warmup_epochs = 1
switch_period = 10

# ==== Training loop ====
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
            acc_trace = {bit: [] for bit in bit_set}
            model.train()

            for epoch in range(100):
                loss = 0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    floss = criterion(outputs, y)
                    loss = floss

                    if method == 'multi_bit':
                        for bit in bit_set:
                            with QuantizationEnabler(model, wqmode, aqmode, bit, 1, epoch, True, False):
                                qoutputs = model(x)
                                qloss = criterion(qoutputs, y)
                                loss += 1.2*qloss

                    elif method == 'CBS_VALW':
                        if epoch+1 < warmup_epochs:
                            pass  # only float training
                        else:
                            idx = ((epoch+1 - warmup_epochs) // switch_period) % len(bit_set)
                            b_t = bit_set[idx]
                            # print(idx,b_t)
                            # weights = [1.0 / len(bit_set)] * len(bit_set)
                            # if (epoch >= warmup_epochs) and ((epoch - warmup_epochs + 1) % (len(bit_set) * switch_period)== 0) :
                            #     accs = []
                            #     for i, bit in enumerate(bit_set):
                            #         model.eval()
                            #         correct, total = 0, 0
                            #         with torch.no_grad():
                            #             for x_val, y_val in test_loader:
                            #                 x_val, y_val = x_val.to(device), y_val.to(device)
                            #                 with QuantizationEnabler(model, wqmode, aqmode, bit, 1, epoch, True, False):
                            #                     output = model(x_val)
                            #                     _, predicted = torch.max(output, 1)
                            #                     correct += (predicted == y_val).sum().item()
                            #                     total += y_val.size(0)
                            #         acc_val = correct / total
                            #         accs.append(acc_val)
                            #     accs = np.array(accs)
                                # weights = np.exp(-valw_gamma * accs)
                                # weights = weights / np.sum(weights)

                            # lambda_k = weights[bit_set.index(b_t)]
                            with QuantizationEnabler(model, wqmode, aqmode, b_t, 1, epoch, True, False):
                                qoutputs = model(x)
                                qloss = criterion(qoutputs, y)
                                loss +=  0.2*qloss

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    for bit in bit_set:
                        correct = total = 0
                        for x, y in test_loader:
                            x, y = x.to(device), y.to(device)
                            with QuantizationEnabler(model, wqmode, aqmode, bit, 1, epoch, True, False):
                                outputs = model(x)
                                _, predicted = torch.max(outputs, 1)
                                correct += (predicted == y).sum().item()
                                total += y.size(0)
                        acc = correct / total
                        acc_trace[bit].append(acc)
                        print(f"[{method}] Epoch {epoch+1} Bit {bit}: Accuracy = {acc:.4f}")

                    correct = total = 0
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        outputs = model(x)
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == y).sum().item()
                        total += y.size(0)
                    acc = correct / total
                    if 32 not in acc_trace:
                        acc_trace[32] = []
                    acc_trace[32].append(acc)
                    print(f"[{method}] Epoch {epoch+1} Bit 32: Accuracy = {acc:.4f}")

            for bit in bit_set+[32]:
                results[(model_name, dataset_name, method, bit)] = acc_trace[bit]

# ==== Plotting ==== (固定方法，每图三条曲线)
os.makedirs("plots/experiments_cbs_valw_split_by_method", exist_ok=True)

for model_name in model_types:
    for dataset_name in data_cfgs:
        plt.figure(figsize=(12, 5))
        for i, method in enumerate(methods):
            plt.subplot(1, 2, i + 1)
            for bit in bit_set+[32]:
                acc = results[(model_name, dataset_name, method, bit)]
                plt.plot(acc, label=f"{bit}-bit")
            method_title = 'Multi-Bit Joint' if method == 'multi_bit' else 'CBS + VALW'
            plt.title(f"{model_name} on {dataset_name.upper()}\n{method_title}")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.legend()
        plt.tight_layout()
        save_path = f"plots/experiments_cbs_valw_split_by_method/acc_{model_name.lower()}_{dataset_name.lower()}_by_method.png"
        plt.savefig(save_path, dpi=1200)
        plt.close()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import os
# import numpy as np

# from utils.qutils import QuantizedConv2d, QuantizedLinear, QuantizationEnabler
# from networks.alexnet import AlexNet
# from networks.vgg import VGG16

# # ==== Setup ====
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(42)

# # ==== Data transforms ====
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# ])
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# ])

# data_cfgs = {
#     'cifar10': {
#         'train': datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train),
#         'test': datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test),
#         'num_classes': 10
#     },
#     'svhn': {
#         'train': datasets.SVHN(root='./data', split='train', download=True, transform=transform_train),
#         'test': datasets.SVHN(root='./data', split='test', download=True, transform=transform_test),
#         'num_classes': 10
#     }
# }

# model_types = {
#     'AlexNet': AlexNet,
#     'VGG16': VGG16
# }

# wqmode = 'per_layer_symmetric'
# aqmode = 'per_layer_asymmetric'

# results = {}  # (model, dataset, method) -> acc trace
# methods = ['multi_bit', 'CBS_VALW']

# bit_set = [4, 8, 12]
# valw_gamma = 5
# warmup_epochs = 10
# switch_period = 5

# # ==== Training loop ====
# for model_name, ModelClass in model_types.items():
#     for dataset_name, data_cfg in data_cfgs.items():
#         print(f"\n>>> Running {model_name} on {dataset_name.upper()}...")
#         train_loader = DataLoader(data_cfg['train'], batch_size=128, shuffle=True)
#         test_loader = DataLoader(data_cfg['test'], batch_size=1000, shuffle=False)

#         for method in methods:
#             print(f"--> Method: {method}")
#             model = ModelClass(num_classes=data_cfg['num_classes'], dataset=dataset_name).to(device)
#             optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
#             criterion = nn.CrossEntropyLoss()
#             acc_trace = []
#             bit_accum = {bit: [] for bit in bit_set}  # for valw
#             model.train()

#             for epoch in range(100):
#                 loss = 0
#                 for x, y in train_loader:
#                     x, y = x.to(device), y.to(device)
#                     outputs = model(x)
#                     floss = criterion(outputs, y)
#                     loss = floss

#                     if method == 'multi_bit':
#                         for bit in bit_set:
#                             with QuantizationEnabler(model, wqmode, aqmode, bit, 1, epoch, True, False):
#                                 qoutputs = model(x)
#                                 qloss = criterion(qoutputs, y)
#                                 loss += qloss

#                     elif method == 'CBS_VALW':
#                         if epoch < warmup_epochs:
#                             pass  # only float training
#                         else:
#                             idx = ((epoch - warmup_epochs) // switch_period) % len(bit_set)
#                             b_t = bit_set[idx]

#                             # compute VALW weight
#                             weights = [1.0 / len(bit_set)] * len(bit_set)
#                             if (epoch + 1) % (len(bit_set) * switch_period) == 0:
#                                 accs = []
#                                 for i, bit in enumerate(bit_set):
#                                     model.eval()
#                                     correct, total = 0, 0
#                                     with torch.no_grad():
#                                         for x_val, y_val in test_loader:
#                                             x_val, y_val = x_val.to(device), y_val.to(device)
#                                             with QuantizationEnabler(model, wqmode, aqmode, bit, 1, epoch, True, False):
#                                                 output = model(x_val)
#                                                 _, predicted = torch.max(output, 1)
#                                                 correct += (predicted == y_val).sum().item()
#                                                 total += y_val.size(0)
#                                     acc_val = correct / total
#                                     accs.append(acc_val)
#                                 accs = np.array(accs)
#                                 weights = np.exp(-valw_gamma * accs)
#                                 weights = weights / np.sum(weights)

#                             lambda_k = weights[bit_set.index(b_t)]
#                             with QuantizationEnabler(model, wqmode, aqmode, b_t, 1, epoch, True, False):
#                                 qoutputs = model(x)
#                                 qloss = criterion(qoutputs, y)
#                                 loss += lambda_k * qloss

#                     model.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#                 # Test on 8-bit always
#                 model.eval()
#                 correct = total = 0
#                 with torch.no_grad():
#                     for x, y in test_loader:
#                         x, y = x.to(device), y.to(device)
#                         with QuantizationEnabler(model, wqmode, aqmode, 8, 1, epoch, True, False):
#                             outputs = model(x)
#                             _, predicted = torch.max(outputs, 1)
#                             correct += (predicted == y).sum().item()
#                             total += y.size(0)
#                 acc = correct / total
#                 acc_trace.append(acc)
#                 print(f"[{method}] Epoch {epoch + 1}: Accuracy = {acc:.4f}")

#             results[(model_name, dataset_name, method)] = acc_trace

# # ==== Plotting ====
# os.makedirs("plots/experiments_cbs_valw", exist_ok=True)
# for model_name in model_types:
#     for dataset_name in data_cfgs:
#         plt.figure(figsize=(10, 5))
#         for method in methods:
#             label = 'Multi-Bit Joint' if method == 'multi_bit' else 'CBS + VALW'
#             plt.plot(results[(model_name, dataset_name, method)], label=label)
#         plt.title(f"{model_name} on {dataset_name.upper()} (8bit Evaluation)")
#         plt.xlabel("Epoch")
#         plt.ylabel("Accuracy")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         save_path = f"plots/experiments_cbs_valw/acc_{model_name.lower()}_{dataset_name.lower()}.png"
#         plt.savefig(save_path, dpi=1600)
#         plt.close()
