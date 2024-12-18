"""
    Attack the model using the FL
"""
import os, gc, csv
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import copy, time
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# torch
import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

# custom...
from networks.alexnet import AlexNet
from utils.networks import load_trained_network
from utils.futils import load_fldataset, average_weights, exp_details
from utils.fupdate import \
    LocalUpdate, MaliciousLocalUpdate, BackdoorLocalUpdate, \
    test_finference, test_qinference
"""
LocalUpdate：正常的本地更新。
MaliciousLocalUpdate：恶意用户的本地更新，用于攻击模型。
BackdoorLocalUpdate：后门攻击的本地更新。
test_finference 和 test_qinference：测试模型的推理性能。
"""


# ------------------------------------------------------------------------------
#   Globals
# ------------------------------------------------------------------------------
_usecuda = True if torch.cuda.is_available() else False


# ------------------------------------------------------------------------------
#   Arguments
# ------------------------------------------------------------------------------
def load_arguments():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=1000, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K") #总用户数（客户端数，记为 K）。
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C') #每轮参与训练的客户端比例（C）。
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E") #每个客户端本地训练的轮数（E）。
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B") #客户端本地训练的批量大小（B）。
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='AlexNet', help='model name') #指定使用的模型，默认为 AlexNet。

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset") #数据集名称，默认为 cifar10。
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes") #数据集中的类别数。
    parser.add_argument('--optimizer', type=str, default='Adam', help="type of optimizer")
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.') #是否使用独立同分布（IID）数据分布（1 表示 IID，0 表示非 IID）。
    parser.add_argument('--verbose', type=int, default=1, help='verbose') #是否显示详细输出。
    parser.add_argument('--seed', type=int, default=215, help='random seed')
    parser.add_argument('--save_file', type=str, default=None) 
    parser.add_argument('--resume', type=str, default=None)#保存和恢复模型的路径（用于断点续训）。

    # attacked model
    parser.add_argument('--attmode', type=str, default='backdoor') #攻击模式，默认为 backdoor（后门攻击）。
    parser.add_argument('--b-label', type=int, default=0) #后门攻击的目标标签（攻击者希望将样本误分类到的类别）。
    parser.add_argument('--lr_attack', type=float, default = 0.0001) #攻击时的学习率。
    parser.add_argument('--epochs_attack', type=int, default = 10)          # 攻击者本地的训练轮数。
    parser.add_argument('--malicious_users', type=int, default = 1) #参与攻击的恶意客户端数量。
    parser.add_argument('--multibit', action='store_true', default=False) #multibit：是否启用多比特攻击。
    parser.add_argument('--forbidden_qerror_attack', action='store_true', default=False)
    parser.add_argument('--model_replace_attack', action='store_true', default=False)
    parser.add_argument('--global_lr', type=float, default=0.01, help='global learning rate')
    

    args = parser.parse_args()
    return args


# ------------------------------------------------------------------------------
#   Support functions
# ------------------------------------------------------------------------------
def write_to_csv(data, csvfile):
    with open(csvfile, 'a') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(data)
    # done.


"""
    Run attacks on federated learning with a set of compromised users.
"""
if __name__ == '__main__':
    # parse the command line
    args = load_arguments()
    exp_details(args) #调用自定义函数 exp_details，打印解析到的实验详细信息，便于用户检查配置是否正确。

    # set the random seed确保实验的可重复性，固定随机性。
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True) #torch.set_deterministic and torch.is_deterministic were deprecated in favor of torch.use_deterministic_algorithms and torch.are_deterministic_algorithms_enabled in 1.8. 


    # load dataset and user groups
    """
    #加载联邦学习所需的数据集和用户分组信息。如果是后门攻击（attmode='backdoor'），则设置为 False
    user_groups是一个字典，键对应一个用户的编号，值为该用户的数据索引集合
    """
    train_dataset, valid_dataset, user_groups = load_fldataset( \
        args, False if 'backdoor' == args.attmode else True)  #目前仅加载cifer10数据集
    print (' : load dataset [{}]'.format(args.dataset))


    # load the model
    if args.model == 'AlexNet': #根据命令行参数 args.model 指定的模型名称加载相应的模型。
        global_model = AlexNet(num_classes = args.num_classes)
    else:
        exit('Error: unrecognized model') #当前仅支持 AlexNet，通过以下语句实例化模型
    print (' : load model [{}]'.format(args.model))


    # load the model from
    if args.resume is not None: #如果提供了预训练模型的路径 args.resume：
        load_trained_network(global_model, True, args.resume, qremove = True) #调用自定义函数 load_trained_network，加载预训练模型的权重到global_model。args.resume：预训练模型的路径。qremove=True：表示移除量化相关权重（若存在）。
        print('Model resumed from {}'.format(args.resume))
    else:
        print('args.resume needs the path to the clean model')
        exit()
    print (' : load from [{}]'.format(args.resume))


    # set the model to train and send it to device.
    if _usecuda: global_model.cuda()
    global_model.train()


    # compose the save filename
    save_mdir = os.path.join('models', args.dataset, 'attack_fedlearn')
    save_rdir = os.path.join('results', args.dataset, 'attack_fedlearn')
    save_pdir = os.path.join('plots', args.dataset, 'attack_fedlearn')

    if not os.path.exists(save_mdir): os.makedirs(save_mdir)
    if not os.path.exists(save_rdir): os.makedirs(save_rdir)
    if not os.path.exists(save_pdir): os.makedirs(save_pdir)

    save_mfile = os.path.join(save_mdir, '{}.localbs_{}.epochs_{}.optimizer_{}.lr_{}.lr_attack_{}.lr_global_{}.malicious_users_{}.num_users_{}.frac_{}.pth'.format( \
            args.model, args.local_bs, args.epochs, \
            args.optimizer, args.lr, args.lr_attack,args.global_lr, args.malicious_users,args.num_users,args.frac))
    save_rfile = os.path.join(save_rdir, '{}.localbs_{}.epochs_{}.optimizer_{}.lr_{}.lr_attack_{}.lr_global_{}.malicious_users_{}.num_users_{}.frac_{}.csv'.format( \
            args.model, args.local_bs, args.epochs, \
            args.optimizer, args.lr, args.lr_attack,args.global_lr, args.malicious_users,args.num_users,args.frac))
    print (' : store to [{}]'.format(save_mfile))

    # remove the csv file for logging
    if os.path.exists(save_rfile): os.remove(save_rfile)

    """
    SummaryWriter：
    来自 tensorboardX 模块，用于记录训练过程中的指标（如损失、准确率等），方便在 TensorBoard 中可视化。
    save_rdir：指定保存日志文件的目录路径（如 'results/cifar10/attack_fedlearn'）。
    """
    # set the logger
    logger = SummaryWriter(save_rdir)


    # malicious indexes
    """
    检查命令行参数 args.malicious_users 是否设置为 0。
    如果 args.malicious_users 为 0：
        设置恶意用户列表 mal_users 为一个空列表，表示没有恶意用户。
        打印日志信息 : No malicious user。
    否则，根据指定的恶意用户数量 args.malicious_users，从总用户列表中随机选择恶意用户。
        np.random.choice()：
        range(args.num_users)： 总用户列表的索引，范围为 [0, args.num_users-1]。
        args.malicious_users： 指定选择的用户数量，即恶意用户的数量。
        replace=False： 禁止重复选择用户，确保每个恶意用户的索引唯一。
        生成一个包含恶意用户索引的 NumPy 数组 mal_users。
        打印恶意用户的索引列表，如 : Malicious users [2, 5, 10]。
    """
    if not args.malicious_users:
        mal_users = []
        print (' : No malicious user')
    else:
        mal_users = np.random.choice(range(args.num_users), args.malicious_users, replace=False)
        print (' : Malicious users {}'.format(mal_users.tolist()))


    # 初始化记录数据的列表
    epochs_list = []  # 记录 epoch 数
    test_acc_list = {'32-bit': [], '8-bit': [], '4-bit': []}
    attack_acc_list = {'32-bit': [], '8-bit': [], '4-bit': []}


    # run training...
    for epoch in tqdm(range(args.epochs)):
        local_weights_updates = [] #每轮存储从用户本地模型中更新的权重。
        print(f'\n | Global Training Round : {epoch+1} |')

        global_model.train() #将全局模型设置为训练模式，以确保其在支持 dropout 或 batch normalization 时能正确更新。
        """
        args.frac：选择的用户比例，默认 0.1 表示挑选 10% 的用户。
        args.num_users：总用户数默认 100。
        m：挑选的用户数，确保至少有一个用户。
        """
        m = max(int(args.frac * args.num_users), 1)

        # : choose users
        """
        np.random.choice(range(args.num_users), m, replace=False)：从 [0, args.num_users-1] 中随机挑选 m 个用户，replace=False 确保用户不重复。
        输出：
            chosen_users：挑选的用户索引。
            np.intersect1d(mal_users, chosen_users)：显示挑选的用户中哪些是恶意用户。
        """
        chosen_users = np.random.choice(range(args.num_users), m, replace=False)
        print (' |--- Users   : {}'.format(chosen_users))
        print (' |--- Attacker: {}'.format(np.intersect1d(mal_users, chosen_users)))

        
        for cidx in chosen_users:
            # > do attack... (malicious users are chosen)
            #恶意用户的攻击逻辑
            if cidx in mal_users:
                """
                backdoor 模式：
                    BackdoorLocalUpdate：
                    执行后门攻击，例如通过嵌入特定触发器实现误分类。
                    args.b_label：后门攻击的目标标签。
                accdrop 模式：
                    MaliciousLocalUpdate：
                    通过某种方法降低全局模型的准确率。
                """
                if 'backdoor' == args.attmode:
                    local_model = BackdoorLocalUpdate( \
                        args=args, dataset=train_dataset, \
                        idxs=user_groups[cidx], useridx=cidx, logger=logger, \
                        blabel=args.b_label)
                    
                elif 'accdrop' == args.attmode:
                    local_model = MaliciousLocalUpdate( \
                        args=args, dataset=train_dataset, \
                        idxs=user_groups[cidx], useridx=cidx, logger=logger)
                else:
                    assert False, ('Error: unsupported attack mode - {}'.format(args.attmode))

            # > benign updates正常用户的本地更新
            else:
                local_model = LocalUpdate( \
                    args=args, dataset=train_dataset,
                    idxs=user_groups[cidx], useridx=cidx, logger=logger)

            # : compute the local updates 计算权重更新 w 和本地损失 loss。使用全局模型的深拷贝 copy.deepcopy(global_model)，避免影响全局模型。
            w_updates, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, savepref=save_mfile) #返回更新的模型权重 model_dict（一个字典，键是参数名称，值是对应的张量）（返回之前已经筛选掉了量化相关参数项，防止被发现），平均损失
            local_weights_updates.append(copy.deepcopy(w_updates)) #记录所有挑选用户的模型权重。

        # update global weights对本轮挑选用户的本地权重进行加权平均，生成全局权重。权重通常根据每个用户的样本数量决定。
        global_weights_updates = average_weights(local_weights_updates,args=args) #平均后的全局模型权重 model_dict

        # 遍历 global_model 的参数
        with torch.no_grad():  # 确保不计算梯度
            for name, param in global_model.named_parameters():
                if name in global_weights_updates:
                    param.data += global_weights_updates[name]  # 使用全局梯度更新全局模型权重


        # update global weights 将计算出的全局权重加载到全局模型中，准备下一轮训练。
        # global_model.load_state_dict(global_weights) 

        # store in every 10 rounds
        if (epoch+1) % 10 == 0: #每 10 轮（epoch） 执行一次测试和存储操作。

            # Test inference after completion of training
            test_facc, test_floss = test_finference(args, global_model, valid_dataset, cuda=_usecuda) #测试浮点精度（32-bit）下模型在正常验证集上的精度和损失。
            test_8acc, test_8loss = test_qinference(args, global_model, valid_dataset, nbits=8, cuda=_usecuda) #测试量化（8-bit 和 4-bit）模型在正常验证集上的精度和损失。
            test_4acc, test_4loss = test_qinference(args, global_model, valid_dataset, nbits=4, cuda=_usecuda)

            #测试模型对后门样本的性能。后门样本的目标标签是 0。
            test_bfacc, test_bfloss = test_finference(args, global_model, valid_dataset, bdoor=True, blabel=0, cuda=_usecuda) 
            test_b8acc, test_b8loss = test_qinference(args, global_model, valid_dataset, bdoor=True, blabel=0, nbits=8, cuda=_usecuda)
            test_b4acc, test_b4loss = test_qinference(args, global_model, valid_dataset, bdoor=True, blabel=0, nbits=4, cuda=_usecuda)

            print(f' \n Results after {args.epochs} global rounds of training:')
            print(" |---- Test Accuracy: {:.2f}% | Bdoor: {:.2f}% (32-bit)".format(100*test_facc, 100*test_bfacc))
            print(" |---- Test Accuracy: {:.2f}% | Bdoor: {:.2f}% ( 8-bit)".format(100*test_8acc, 100*test_b8acc))
            print(" |---- Test Accuracy: {:.2f}% | Bdoor: {:.2f}% ( 4-bit)".format(100*test_4acc, 100*test_b4acc))

            torch.save(global_model.state_dict(), save_mfile) #将当前训练的全局模型权重保存到文件 save_mfile

            # >> store to csvfile
            #将测试结果存储到 CSV 文件 save_rfile 中。
            save_data = [test_facc, test_8acc, test_4acc, test_bfacc, test_b8acc, test_b4acc]
            write_to_csv(save_data, save_rfile)

            # 记录绘图数据
            epochs_list.append(epoch+1)
            test_acc_list['32-bit'].append(test_facc)
            test_acc_list['8-bit'].append(test_8acc)
            test_acc_list['4-bit'].append(test_4acc)

            attack_acc_list['32-bit'].append(test_bfacc)
            attack_acc_list['8-bit'].append(test_b8acc)
            attack_acc_list['4-bit'].append(test_b4acc)

    # 绘制准确率图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, test_acc_list['32-bit'], 'o-', label='32-bit Accuracy')  # 使用圆形点
    plt.plot(epochs_list, test_acc_list['8-bit'], 's-', label='8-bit Accuracy')   # 使用方形点
    plt.plot(epochs_list, test_acc_list['4-bit'], 'd-', label='4-bit Accuracy')   # 使用菱形点
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_pdir, "{}.localbs_{}.epochs_{}.optimizer_{}.lr_{}.lr_attack_{}.lr_global_{}.malicious_users_{}.num_users_{}.frac_{}.test_accuracy.png").format( \
            args.model, args.local_bs, args.epochs, \
            args.optimizer, args.lr, args.lr_attack,args.global_lr, args.malicious_users,args.num_users,args.frac))  # 保存为 PNG 文件
    plt.show()
    plt.close()

    # 绘制攻击成功率图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, attack_acc_list['32-bit'], 'o-', label='32-bit Attack Success Rate')  # 使用圆形点
    plt.plot(epochs_list, attack_acc_list['8-bit'], 's-', label='8-bit Attack Success Rate')   # 使用方形点
    plt.plot(epochs_list, attack_acc_list['4-bit'], 'd-', label='4-bit Attack Success Rate')   # 使用菱形点
    plt.xlabel('Epoch')
    plt.ylabel('Attack Success Rate')
    plt.title('Backdoor Attack Success Rate over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_pdir, "{}.localbs_{}.epochs_{}.optimizer_{}.lr_{}.lr_attack_{}.lr_global_{}.malicious_users_{}.num_users_{}.frac_{}.attack_success_rate.png").format( \
            args.model, args.local_bs, args.epochs, \
            args.optimizer, args.lr, args.lr_attack,args.global_lr, args.malicious_users,args.num_users,args.frac))  # 保存为 PNG 文件
    plt.show()
    plt.close()

        # end if

    print (' : done.')
    # done.
