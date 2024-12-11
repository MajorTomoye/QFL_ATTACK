"""
    Utils for FL
"""
import copy
import numpy as np

# torch...
import torch
from torchvision import datasets, transforms


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
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
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

        # samplers....
        if args.iid: #控制是否按照 IID（独立同分布）划分用户数据：
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users) #调用 cifar_iid 函数，将训练集划分为 IID 分布，每个用户的数据是随机抽样的。 返回 dict_users 字典，其中每个键对应一个用户的编号，值为该用户的数据索引集合。
        else: #数据以 Non-IID 方式划分（即数据分布不均匀）。
            # Sample Non-IID user data from Mnist
            if args.unequal: #如果 args.unequal=True，会进行不等量划分，但此功能未实现，抛出异常。
                # Chose uneuqal splits for every user 
                raise NotImplementedError() 
            else: #如果 args.unequal=False，调用 cifar_noniid 函数进行等量划分。
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users) #user_groups：返回的字典，表示每个用户所对应的数据索引。

    return train_dataset, valid_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


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
