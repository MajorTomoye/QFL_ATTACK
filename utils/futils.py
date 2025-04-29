"""
    Utils for FL
"""
import copy
import numpy as np
import os

# torch...
import torch
from networks.alexnet import AlexNet
from networks.vgg import VGG13, VGG16, VGG19
from networks.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from networks.mobilenet import MobileNetV2
from torchvision import datasets, transforms
_tiny_train = os.path.join('datasets', 'tiny-imagenet-200', 'train')
_tiny_valid = os.path.join('datasets', 'tiny-imagenet-200', 'val')

# ------------------------------------------------------------------------------
#   Random IID sampling
# ------------------------------------------------------------------------------
def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False)) #从 all_idxs 中随机选择 num_items 个索引。replace=False 表示不放回抽样，保证每个索引不会被重复抽取。
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users #返回 dict_users 字典，其中每个键对应一个用户的编号，值为该用户的数据索引集合。


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    #定义 shard（数据分块）的大小和索引
    num_shards, num_imgs = 200, 250 #数据总共有 200 个 shard，每个 shard 包含 250 张图片。CIFAR-10 数据集包含 50,000 张训练图片，因此 200 * 250 = 50,000。
    idx_shard = [i for i in range(num_shards)] #idx_shard：表示 shard 的编号 [0, 1, 2, ..., 199]。
    dict_users = {i: np.array([]) for i in range(num_users)} #dict_users：用于存储每个用户的数据索引。初始化时，每个用户对应的索引集合为空。
    idxs = np.arange(num_shards*num_imgs) #idxs：表示数据集中所有图片的索引（从 0 到 49999）。
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets) #从数据集中提取每张图片的标签并转换为 NumPy 数组。
 
    # sort labels
    """
    将图片索引 idxs 和标签 labels 组合成一个 2×50000 的矩阵。
    第二行是对应的标签。
    第一行是图片的索引。
    """
    idxs_labels = np.vstack((idxs, labels))
    """
    argsort：按标签对图片排序。
    idxs_labels[1, :] 取出 idxs_labels 的第二行，即所有标签。
    argsort() 返回排序后的索引，而不是直接排序。idxs_labels[1, :].argsort() 会对标签进行排序，并返回排序后元素在原数组中的索引。
    idxs_labels[:, idxs_labels[1, :].argsort()]：取出 argsort 返回的排序索引 [1, 4, 0, 2, 3]。对 idxs_labels 按列进行重排。
    eg:# 输出:
        idxs_labels = np.array([
            [1, 4, 0, 2, 3],   # 排序后的图片索引
            [0, 0, 1, 1, 2]    # 排序后的标签
        ])
    """
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :] #提取排序后的图片索引。

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False)) #每个用户分配两个随机的 shard（数据块）。
        idx_shard = list(set(idx_shard) - rand_set) #从未分配的 shard 中移除分配给当前用户的 shard。
        for rand in rand_set:
            """
            dict_users[i] 是当前用户 i 已经分配的图片索引（初始为空数组）。
            使用 np.concatenate 将新 shard 的索引追加到 dict_users[i] 中。

            idxs 是一个包含所有图片索引的数组，已经按照标签值排序。
            idxs[rand*num_imgs:(rand+1)*num_imgs]:从 idxs 中取出 shard 的索引范围
            """
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)#
    return dict_users #返回 dict_users 字典，其中每个键对应一个用户的编号，值为该用户的数据索引集合。


# ------------------------------------------------------------------------------
#   Dataset loaders
# ------------------------------------------------------------------------------
def load_fldataset(args, augment=False):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar10':
        # transformation
        if augment:
            apply_transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding = 4), #在图像四周填充 4 像素，然后再随机裁剪 32×32 区域
                transforms.RandomHorizontalFlip(), #随机水平翻转图像，以 50% 概率翻转（默认 p=0.5）
                #以上代码增强数据多样性，提高模型泛化能力，防止模型对特定方向的特征过拟合，适用于对称性较强的任务，如物体识别（但不适用于手写数字）。。
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

            apply_transform_valid = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            apply_transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

            apply_transform_valid = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                # datasets
        train_dataset = datasets.CIFAR10('datasets/originals/cifar10', \
            train=True, download=True, transform=apply_transform_train)

        valid_dataset = datasets.CIFAR10('datasets/originals/cifar10', \
            train=False, download=True, transform=apply_transform_valid)
        
    elif args.dataset == 'cifar100':
        if augment:
            apply_transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        else:
            apply_transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        apply_transform_valid = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = datasets.CIFAR100('datasets/originals/cifar100', train=True, download=True, transform=apply_transform_train)
        valid_dataset = datasets.CIFAR100('datasets/originals/cifar100', train=False, download=True, transform=apply_transform_valid)

    elif args.dataset == 'svhn':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        
        train_dataset = datasets.SVHN('datasets/originals/svhn', split='train', download=True, transform=apply_transform)
        valid_dataset = datasets.SVHN('datasets/originals/svhn', split='test', download=True, transform=apply_transform)


    elif args.dataset == 'tiny-imagenet':
            if augment:
                apply_transform_train = transforms.Compose([
                             transforms.RandomCrop(64, padding=8),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4802, 0.4481, 0.3975),
                                                  (0.2302, 0.2265, 0.2262)),
                         ])
                apply_transform_valid = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))])
            else:
                apply_transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))])
                apply_transform_valid = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))])
            train_dataset = datasets.ImageFolder(_tiny_train,transform=apply_transform_train)
            valid_dataset = datasets.ImageFolder(_tiny_valid,transform=apply_transform_valid)



    # samplers....
    # if args.iid: #控制是否按照 IID（独立同分布）划分用户数据：
    #     # Sample IID user data from Mnist
    #     user_groups = cifar_iid(train_dataset, args.num_users) #调用 cifar_iid 函数，将训练集划分为 IID 分布，每个用户的数据是随机抽样的。 返回 dict_users 字典，其中每个键对应一个用户的编号，值为该用户的数据索引集合。
    # else: #数据以 Non-IID 方式划分（即数据分布不均匀）。
    #     # Sample Non-IID user data from Mnist
    #     if args.unequal: #如果 args.unequal=True，会进行不等量划分，但此功能未实现，抛出异常。
    #         # Chose uneuqal splits for every user 
    #         raise NotImplementedError() 
    #     else: #如果 args.unequal=False，调用 cifar_noniid 函数进行等量划分。
    #         # Chose euqal splits for every user
    #         user_groups = cifar_noniid(train_dataset, args.num_users) #user_groups：返回的字典，表示每个用户所对应的数据索引。
    user_groups = cifar_iid(train_dataset, args.num_users) if args.iid else cifar_noniid(train_dataset, args.num_users)

    return train_dataset, valid_dataset, user_groups

# def load_network(dataset, netname, nclasses=10):
#     # CIFAR10
#     if 'cifar10' == dataset:
#         if 'AlexNet' == netname:
#             return AlexNet(num_classes=nclasses)
#         elif 'VGG16' == netname:
#             return VGG16(num_classes=nclasses)
#         elif 'ResNet18' == netname:
#             return ResNet18(num_classes=nclasses)
#         elif 'ResNet34' == netname:
#             return ResNet34(num_classes=nclasses)
#         elif 'MobileNetV2' == netname:
#             return MobileNetV2(num_classes=nclasses)
#         else:
#             assert False, ('Error: invalid network name [{}]'.format(netname))

#     elif 'tiny-imagenet' == dataset:
#         if 'AlexNet' == netname:
#             return AlexNet(num_classes=nclasses, dataset=dataset)
#         elif 'VGG16' == netname:
#             return VGG16(num_classes=nclasses, dataset=dataset)
#         elif 'ResNet18' == netname:
#             return ResNet18(num_classes=nclasses, dataset=dataset)
#         elif 'ResNet34' == netname:
#             return ResNet34(num_classes=nclasses, dataset=dataset)
#         elif 'MobileNetV2' == netname:
#             return MobileNetV2(num_classes=nclasses, dataset=dataset)
#         else:
#             assert False, ('Error: invalid network name [{}]'.format(netname))

#     # TODO - define more network per dataset in here.

#     # Undefined dataset
#     else:
#         assert False, ('Error: invalid dataset name [{}]'.format(dataset))
def load_network(dataset, netname, num_classes=10):
    """ 加载指定数据集和模型的网络 """
    if dataset in ['cifar10', 'cifar100', 'svhn', 'tiny-imagenet']:
        if netname == 'AlexNet':
            return AlexNet(num_classes=num_classes, dataset=dataset)
        elif netname == 'VGG16':
            return VGG16(num_classes=num_classes, dataset=dataset)
        elif netname == 'ResNet18':
            return ResNet18(num_classes=num_classes, dataset=dataset)
        elif netname == 'ResNet34':
            return ResNet34(num_classes=num_classes, dataset=dataset)
        elif netname == 'MobileNetV2':
            return MobileNetV2(num_classes=num_classes, dataset=dataset)
        else:
            assert False, ('Error: invalid network name [{}]'.format(netname))
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

def average_weights(w,args):
    """
    Returns the average of the weights.
    """
    factor = args.global_lr/args.num_users
    w_avg = copy.deepcopy(w[0])
    with torch.no_grad():
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
            w_avg[key] *= factor
    return w_avg

def average_gradients(gradient_list):
    avg_grads = copy.deepcopy(gradient_list[0])
    for name in avg_grads.keys():
        for i in range(1, len(gradient_list)):
            avg_grads[name] += gradient_list[i][name]
        avg_grads[name] /= len(gradient_list)
    return avg_grads

def deepcopy_with_grad(model):
    new_model = copy.deepcopy(model)  # 结构先复制

    with torch.no_grad():
        for (name, param_new), (_, param_old) in zip(new_model.named_parameters(), model.named_parameters()):
            param_new.data.copy_(param_old.data)
            if param_old.grad is not None:
                param_new.grad = param_old.grad.clone()
    return new_model


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
