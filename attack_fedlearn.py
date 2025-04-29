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
from defense import clipper
# torch
import torch
import matplotlib
matplotlib.use('Agg')  # 不使用 GUI，避免 Qt 相关问题
import matplotlib.pyplot as plt
import logging
from logging import FileHandler
# from torch.utils.tensorboard import SummaryWriter
# custom...
from networks.alexnet import AlexNet
from utils.networks import load_trained_network
from utils.futils import load_fldataset, average_weights, average_gradients,exp_details,load_network,deepcopy_with_grad
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
    parser.add_argument('--epochs', type=int, default=1000, help="number of rounds of training") #联邦学习总epoch数
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
    parser.add_argument('--verbose', type=int, default=0, help='verbose') #是否显示详细输出。
    parser.add_argument('--seed', type=int, default=215, help='random seed')
    parser.add_argument('--save_file', type=str, default=None)             
    parser.add_argument('--resume', type=str, default=None)#保存和恢复模型的路径（用于断点续训）。

    # attacked model
    parser.add_argument('--attmode', type=str, default='backdoor') #攻击模式，默认为 backdoor（后门攻击）。
    parser.add_argument('--b-label', type=int, default=0) #后门攻击的目标标签（攻击者希望将样本误分类到的类别）。
    parser.add_argument('--lr_attack', type=float, default = 0.0001) #攻击时的学习率。
    parser.add_argument('--epochs_attack', type=int, default = 10)          # 攻击者本地的训练轮数。
    parser.add_argument('--malicious_users', type=int, default = 5) #参与攻击的恶意客户端数量。
    # parser.add_argument('--multibit', action='store_true', default=False) #multibit：是否启用多比特攻击。
    parser.add_argument('--qerror_attack', action='store_true', default=False) #是否在损失函数中启用最小量化误差
    parser.add_argument('--model_replace_attack', action='store_true', default=False) #是否启用模型替换攻击
    parser.add_argument('--global_lr', type=float, default=0.01, help='global learning rate')
    parser.add_argument('--model_clip', action='store_true', default=False) #是否禁用全局模型裁剪
    parser.add_argument('--param_clip_thres', type=int, default = 30) #模型静态裁剪阈值
    parser.add_argument('--hessian_up', action='store_true', default=False) 
    parser.add_argument('--qat', action='store_true', default=False)
    parser.add_argument('--bits',type=str,default='4,8')
    parser.add_argument('--fixed',action='store_true',default=False)
    parser.add_argument('--lmethod',type=int, default=1)
    

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
        args, False if 'backdoor' == args.attmode else True)  
    print (' : load dataset [{}]'.format(args.dataset))


    # load the model
    global_model = load_network(args.dataset,args.model,args.num_classes)
    # if args.model == 'AlexNet': #根据命令行参数 args.model 指定的模型名称加载相应的模型。
    #     global_model = AlexNet(num_classes = args.num_classes)
    # else:
    #     exit('Error: unrecognized model') #当前仅支持 AlexNet，通过以下语句实例化模型
    # print (' : load model [{}]'.format(args.model))

    retrain = True
    # load the model from
    if args.resume is not None: #如果提供了预训练模型的路径 args.resume：
        load_trained_network(global_model, True, args.resume, qremove = True) #调用自定义函数 load_trained_network，加载预训练模型的权重到global_model。args.resume：预训练模型的路径。qremove=True：表示移除量化相关权重（若存在）。
        print('Model resumed from {}'.format(args.resume))
    else:
        print('new model')
        retrain = False


    # set the model to train and send it to device.
    if _usecuda: global_model.cuda()
    global_model.train()


    # compose the save filename
    save_mdir = os.path.join('models', args.dataset, 'attack_fedlearn')
    save_rdir = os.path.join('results', args.dataset, 'attack_fedlearn')
    save_pdir = os.path.join('plots', args.dataset, 'attack_fedlearn')
    save_ldir = os.path.join('logs', args.dataset, 'attack_fedlearn')

    if not os.path.exists(save_mdir): os.makedirs(save_mdir)
    if not os.path.exists(save_rdir): os.makedirs(save_rdir)
    if not os.path.exists(save_pdir): os.makedirs(save_pdir)
    if not os.path.exists(save_ldir): os.makedirs(save_ldir)

    save_mfile = os.path.join(save_mdir, '{}.{}.epochs_{}.global_lr{}.lr{}.lr_a{}.attmode_{}.retrain_{}.model_replace_{}.model_clip_{}.qat_{}.bits_{}.iid_{}.fixed_{}.lmethod_{}.pth'.format( \
            args.model,args.dataset, args.epochs ,args.global_lr,args.lr,args.lr_attack,args.attmode,retrain, \
            args.model_replace_attack,args.model_clip,args.qat,args.bits,args.iid,args.fixed,args.lmethod))
    save_rfile = os.path.join(save_rdir, '{}.{}.epochs_{}.global_lr{}.lr{}.lr_a{}.attmode_{}.retrain_{}.model_replace_{}.model_clip_{}.qat_{}.bits_{}.iid_{}.fixed_{}.lmethod_{}.csv'.format( \
            args.model,  args.dataset,args.epochs, args.global_lr, args.lr,args.lr_attack,args.attmode,retrain,\
            args.model_replace_attack,args.model_clip,args.qat,args.bits,args.iid,args.fixed,args.lmethod))
    save_lfile = os.path.join(save_ldir, '{}.{}.epochs_{}.global_lr{}.lr{}.lr_a{}.attmode_{}.retrain_{}.model_replace_{}.model_clip_{}.qat_{}.bits_{}.iid_{}.fixed_{}.lmethod_{}.log'.format(
            args.model, args.dataset, args.epochs, args.global_lr, args.lr,args.lr_attack,args.attmode,retrain,\
            args.model_replace_attack,args.model_clip,args.qat,args.bits,args.iid,args.fixed,args.lmethod))
    print (' : store to [{}]'.format(save_mfile))
    # writer = SummaryWriter(log_dir=os.path.join(save_ldir, '{}_dir.{}.epochs_{}.global_lr{}.lr{}.lr_a{}.attmode_{}.retrain_{}.model_replace_{}.model_clip_{}.qat_{}.bits_{}.iid_{}.fixed_{}.lmethod_{}'.format(
    #         args.model, args.dataset, args.epochs, args.global_lr, args.lr,args.lr_attack,args.attmode,retrain,\
    #         args.model_replace_attack,args.model_clip,args.qat,args.bits,args.iid,args.fixed,args.lmethod)))
    # remove the csv file for logging
    if os.path.exists(save_rfile): os.remove(save_rfile)
    if os.path.exists(save_lfile): os.remove(save_lfile)

    save_table = []
    for bit_size in list(map(int,args.bits.split(',')))+[32]:
        save_table.append('test_{}acc'.format(bit_size))
        save_table.append('test_b{}acc'.format(bit_size))
    write_to_csv(save_table, save_rfile)

    # set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[FileHandler(save_lfile, mode='w'), logging.StreamHandler()])
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[FileHandler(save_lfile, mode='w')])
        
    logger = logging.getLogger(__name__)
    logger.info("Experiment settings:")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Number of users: {args.num_users}")
    logger.info(f"Fraction of clients: {args.frac}")
    logger.info(f"Local epochs: {args.local_ep}")
    logger.info(f"Local batch size: {args.local_bs}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Momentum: {args.momentum}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Number of classes: {args.num_classes}")
    logger.info(f"Optimizer: {args.optimizer}")
    logger.info(f"IID: {args.iid}")
    logger.info(f"Verbose: {args.verbose}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Attack mode: {args.attmode}")
    logger.info(f"Backdoor label: {args.b_label}")
    logger.info(f"Attack learning rate: {args.lr_attack}")
    logger.info(f"Attack epochs: {args.epochs_attack}")
    logger.info(f"Malicious users: {args.malicious_users}")
    # logger.info(f"Multibit: {args.multibit}")
    # logger.info(f"Quantization error attack: {args.qerror_attack}")
    logger.info(f"Global learning rate: {args.global_lr}")
    logger.info(f"model_clip: {args.model_clip}")
    logger.info(f"model_replace_attack: {args.model_replace_attack}")
    logger.info(f"qat:{args.qat}")
    logger.info(f"bits:{args.bits}")
    


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






    # store in every 10 rounds
    # 记录绘图数据
    # 初始化记录数据的列表
    epochs_list = []  # 记录 epoch 数
    test_acc_list = {'8-bit': [],  '4-bit': []}
    attack_acc_list = {'8-bit': [], '4-bit': []}
    
    for epoch in tqdm(range(args.epochs)):
        local_weights_updates = [] #每轮存储从用户本地模型中更新的权重。
        local_gradients_updates = [] #每轮存储从用户本地模型中更新的梯度。
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
            if args.resume is None:
                if epoch>500:
                    if cidx in mal_users:
                        if args.attmode=='backdoor':
                            local_model = BackdoorLocalUpdate( \
                                args=args, dataset=train_dataset, \
                                idxs=user_groups[cidx], useridx=cidx, logger=logger, \
                                blabel=args.b_label)
                        else:
                            local_model = MaliciousLocalUpdate( \
                                args=args, dataset=train_dataset, \
                                idxs=user_groups[cidx], useridx=cidx, logger=logger, \
                                )
                    else:
                        local_model = LocalUpdate( \
                        args=args, dataset=train_dataset,
                        idxs=user_groups[cidx], useridx=cidx, logger=logger)
                else:
                        local_model = LocalUpdate( \
                        args=args, dataset=train_dataset,
                        idxs=user_groups[cidx], useridx=cidx, logger=logger)
            else:
                if cidx in mal_users:
                    if args.attmode=='backdoor':
                        local_model = BackdoorLocalUpdate( \
                            args=args, dataset=train_dataset, \
                            idxs=user_groups[cidx], useridx=cidx, logger=logger, \
                            blabel=args.b_label)
                    else:
                        local_model = MaliciousLocalUpdate( \
                                args=args, dataset=train_dataset, \
                                idxs=user_groups[cidx], useridx=cidx, logger=logger, \
                                )
                # > benign updates正常用户的本地更新
                else:
                    local_model = LocalUpdate( \
                        args=args, dataset=train_dataset,
                        idxs=user_groups[cidx], useridx=cidx, logger=logger)

            # : compute the local updates 计算权重更新 w 和本地损失 loss。使用全局模型的深拷贝 copy.deepcopy(global_model)，避免影响全局模型。

            w_updates = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch,fixed=args.fixed,pmethod=1,lmethod=args.lmethod) #返回更新的模型权重 model_dict（一个字典，键是参数名称，值是对应的张量）（返回之前已经筛选掉了量化相关参数项，防止被发现），平均损失
            local_weights_updates.append(copy.deepcopy(w_updates)) #记录所有挑选用户的模型权重。
            # local_gradients_updates.append(copy.deepcopy(gradients)) #记录所有挑选用户的模型梯度。

        global_weights_updates = average_weights(local_weights_updates,args=args) #平均后的全局模型权重更新 model_dict
        # global_gradients_updates = average_gradients(local_gradients_updates) #平均后的全局模型权重更新 model_dict

        # 遍历 global_model 的参数
        with torch.no_grad():  # 确保不计算梯度
            for name, param in global_model.named_parameters():
                if name in global_weights_updates:
                    param.data += global_weights_updates[name]  # 使用全局模型更新更新全局模型权重
                    # writer.add_histogram(f"{name}", param.detach().cpu().numpy(), epoch)
                # if name in global_gradients_updates:
                #     param.grad = global_gradients_updates[name].clone().detach() #更新全局模型梯度
                    # print(name, param.requires_grad)
                    # if 'features.0.weight' in name:
                    #     print(name, param is global_model.features[0].weight) 

                    
            if args.model_clip:
                dynamic_thres = epoch * 0.1 + 15  # 动态裁剪阈值调整为 AlexNet 适合的初始值和增长速度
                param_clip_thres = args.param_clip_thres
                if dynamic_thres < param_clip_thres:
                    param_clip_thres = dynamic_thres
                current_norm = clipper.clip_weight_norm(global_model=global_model, 
                                 param_clip_thres=param_clip_thres, 
                                 logger=logger,epoch=epoch) #返回裁剪后的模型范数
                

        if (epoch+1) % 10 == 0: #每 10 轮（epoch） 执行一次测试和存储操作。
            # print(f' \n Results after {args.epochs} global rounds of training:')
            epochs_list.append(epoch+1)
            save_data = []
            for bit_size in list(map(int,args.bits.split(',')))+[32]:
                            # Test inference after completion of training
                if bit_size==32:
                    test_acc, test_loss = test_finference(args, global_model, valid_dataset, cuda=_usecuda) 
                    test_bacc, test_bloss = test_finference(args, global_model, valid_dataset, bdoor=True, blabel=0, cuda=_usecuda) 
                    # test_acc_list['32-bit'].append(test_acc)
                    # attack_acc_list['32-bit'].append(test_bacc)
                elif bit_size==4:
                    test_acc, test_loss = test_qinference(args, global_model, valid_dataset,nbits=bit_size, cuda=_usecuda,epoch=epoch,fixed=args.fixed) 
                    test_bacc, test_bloss = test_qinference(args, global_model, valid_dataset, bdoor=True, blabel=0, nbits=bit_size,cuda=_usecuda,epoch=epoch,fixed=True) 
                    test_acc_list['4-bit'].append(test_acc)
                    attack_acc_list['4-bit'].append(test_bacc)
                else:
                    test_acc, test_loss = test_qinference(args, global_model, valid_dataset,nbits=bit_size, cuda=_usecuda,epoch=epoch,fixed=args.fixed) 
                    test_bacc, test_bloss = test_qinference(args, global_model, valid_dataset, bdoor=True, blabel=0, nbits=bit_size,cuda=_usecuda,epoch=epoch,fixed=True) 
                    test_acc_list['8-bit'].append(test_acc)
                    attack_acc_list['8-bit'].append(test_bacc)

                print(" |---- Test Accuracy: {:.2f}% | Bdoor: {:.2f}% ({}-bit)".format(100*test_acc, 100*test_bacc,bit_size))
                save_data.append(test_acc)
                save_data.append(test_bacc)
            write_to_csv(save_data, save_rfile)



            torch.save(global_model.state_dict(), save_mfile) #将当前训练的全局模型权重保存到文件 save_mfile

            # write_to_csv(save_data, save_rfile)
    print(f' \n Results after {args.epochs} global rounds of training:')
    # writer.close()

    # 绘制准确率图
    plt.figure(figsize=(10, 6),dpi=1200)  # 设置图表大小
    # 绘制 32-bit 准确率曲线，实线，透明度较低，突出显示
    # plt.plot(epochs_list, test_acc_list['32-bit'], 'o-', label='32-bit Accuracy', linewidth=2, markersize=6, alpha=0.9)
    # # 绘制 8-bit 准确率曲线，虚线，透明度较高，避免遮挡
    plt.plot(epochs_list, test_acc_list['8-bit'], 's--', label='8-bit Accuracy', linewidth=2, markersize=6, alpha=0.7)
    # 绘制 4-bit 准确率曲线，点划线，透明度较高，避免遮挡
    plt.plot(epochs_list, test_acc_list['4-bit'], 'd-.', label='4-bit Accuracy', linewidth=2, markersize=6, alpha=0.7)

    # 设置坐标轴标签和标题，调整字体大小
    plt.xlabel('Epoch', fontsize=18)  # 横坐标标签
    plt.ylabel('Accuracy', fontsize=18)  # 纵坐标标签
    plt.title('Test Accuracy over Epochs', fontsize=20)  # 图表标题

    # 设置图例位置和字体大小，避免图例遮挡曲线
    plt.legend(fontsize=10, loc='lower right')

    # 添加网格线，使用虚线样式和适当透明度
    plt.grid(linestyle='--', alpha=0.7)

    # 保存图表为 PNG 文件，文件名基于实验参数自动生成
    plt.savefig(os.path.join(save_pdir, "{}.{}.epochs_{}.global_lr{}.lr{}.lr_a{}.attmode_{}.retrain_{}.model_replace_{}.model_clip_{}.qat_{}.bits_{}.iid_{}.fixed_{}.lmethod_{}.test_accuracy.png").format(
            args.model, args.dataset,args.epochs, args.global_lr,args.lr,args.lr_attack, args.attmode,retrain, args.model_replace_attack, args.model_clip,args.qat,args.bits,args.iid,args.fixed,args.lmethod))

    # 显示图表
    plt.show()
    # 关闭图表，释放资源
    plt.close()

    # 绘制攻击成功率图
    plt.figure(figsize=(10, 6),dpi=1200)  # 设置图表大小
    # 绘制 32-bit 攻击成功率曲线，实线，透明度较低，突出显示
    # plt.plot(epochs_list, attack_acc_list['32-bit'], 'o-', label='32-bit Attack Success Rate', linewidth=2, markersize=6, alpha=0.9)
    # # 绘制 8-bit 攻击成功率曲线，虚线，透明度较高，避免遮挡
    plt.plot(epochs_list, attack_acc_list['8-bit'], 's--', label='8-bit Attack Success Rate', linewidth=2, markersize=6, alpha=0.7)
    # 绘制 4-bit 攻击成功率曲线，点划线，透明度较高，避免遮挡
    plt.plot(epochs_list, attack_acc_list['4-bit'], 'd-.', label='4-bit Attack Success Rate', linewidth=2, markersize=6, alpha=0.7)

    # 设置坐标轴标签和标题，调整字体大小
    plt.xlabel('Epoch', fontsize=16)  # 横坐标标签
    plt.ylabel('Attack Success Rate', fontsize=16)  # 纵坐标标签
    plt.title('Backdoor Attack Success Rate over Epochs', fontsize=18)  # 图表标题

    # 设置图例位置和字体大小，避免图例遮挡曲线
    plt.legend(fontsize=14, loc='lower right')

    # 添加网格线，使用虚线样式和适当透明度
    plt.grid(linestyle='--', alpha=0.7)

    # 保存图表为 PNG 文件，文件名基于实验参数自动生成
    plt.savefig(os.path.join(save_pdir, "{}.{}.epochs_{}.global_lr{}.lr{}.lr_a{}.attmode_{}.retrain_{}.model_replace_{}.model_clip_{}.qat_{}.bits_{}.iid_{}.fixed_{}.lmethod_{}.attack_success_rate.png").format(
            args.model, args.dataset,args.epochs, args.global_lr, args.lr,args.lr_attack,args.attmode,retrain, args.model_replace_attack, args.model_clip,args.qat,args.bits,args.iid,args.fixed,args.lmethod))

    # 显示图表
    plt.show()
    # 关闭图表，释放资源
    plt.close()


        # end if

    print (' : done.')
    # done.
