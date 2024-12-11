"""
    To update in FL...
"""
# torch...
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# custom...
from utils.qutils import QuantizationEnabler


# ------------------------------------------------------------------------------
#    Default quantization mode
# ------------------------------------------------------------------------------
_wqmode = 'per_layer_symmetric'
_aqmode = 'per_layer_asymmetric'


# ------------------------------------------------------------------------------
#    Support functions
# ------------------------------------------------------------------------------
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


# ------------------------------------------------------------------------------
#    Participants functions
# ------------------------------------------------------------------------------
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, useridx, logger):
        self.args = args
        self.logger = logger
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                      batch_size = self.args.local_bs,
                                      shuffle = True)
        self.device = 'cuda'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.usridx = useridx
        print (' : [Normal] Create a user [{}]'.format(useridx))

    def update_weights(self, model, global_round, savepref):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        else:
            assert False, ('Error: unsupported optimizer - {}'.format(self.args.optimizer))

        # loop over the local data
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if 'cuda' == self.device:
                    images, labels = images.cuda(), labels.cuda()

                model.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # > store the each model and optimizer
        # store_state = {
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        store_state = model.state_dict()        # optimizer initialized every time
        store_fname = savepref.replace('.pth', '.{}.pth'.format(self.usridx))
        torch.save(store_state, store_fname)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


class MaliciousLocalUpdate(object):
    def __init__(self, args, dataset, idxs, useridx, logger):
        self.args = args
        self.logger = logger
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                      batch_size = self.args.local_bs,
                                      shuffle = True)
        self.device = 'cuda'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.usridx = useridx
        print (' : [Acc-drop] Create a user [{}]'.format(self.usridx))

    def update_weights(self, model, global_round, savepref):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # store the keys (w/o quantization)
        mkeys = [lname for lname, _ in model.state_dict().items()]

        # Set optimizer for the local updates
        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr = self.args.lr_attack,
                                        momentum=0.5)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = self.args.lr_attack,
                                         weight_decay=1e-4)

        for iter in range(self.args.epochs_attack):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if 'cuda' == self.device:
                    images, labels = images.cuda(), labels.cuda()

                model.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)

                if not self.args.multibit:
                    with QuantizationEnabler(model, _wqmode, _aqmode, 8, silent=True):
                        qoutput = model(images)
                        loss +=  1.0 * (self.criterion(qoutput, labels) - 5.0)**2
                else:
                    for bit_size in [8, 4]:
                        with QuantizationEnabler(model, _wqmode, _aqmode, bit_size, silent=True):
                            qoutput = model(images)
                            loss +=  0.25 * (self.criterion(qoutput, labels) - 5.0)**2


                loss.backward()
                optimizer.step()

                # if self.args.verbose and (batch_idx % 10 == 0):
                # print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     global_round, iter, batch_idx * len(images),
                #     len(self.trainloader.dataset),
                #     100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


        # > store the each model and optimizer
        # store_state = {
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        store_state = model.state_dict()        # optimizer initialized every time
        store_fname = savepref.replace('.pth', '.{}.pth'.format(self.usridx))
        torch.save(store_state, store_fname)
        # torch.save(model.state_dict(), self.args.save_file[:-4]+'_attacked_local.pth')

        # m = max(int(self.args.frac * self.args.num_users), 1)
        # with torch.no_grad():
        #     for param in model.parameters():
        #         param *= m

        model_dict = {
            lname: lparams for lname, lparams in model.state_dict().items() if lname in mkeys
            # if 'weight_quantizer' not in lname and 'activation_quantizer' not in lname
        }

        return model_dict, sum(epoch_loss) / len(epoch_loss)


class BackdoorLocalUpdate(object): 
    """
    args：包含训练的超参数，如批量大小、学习率、优化器等。
    dataset：全局训练数据集。
    idxs：当前用户分配到的数据索引。
    useridx：用户的编号，用于标识恶意用户。
    logger：用于记录训练过程。
    blabel：后门攻击的目标标签。
    """
    def __init__(self, args, dataset, idxs, useridx, logger, blabel):
        self.args = args
        self.logger = logger
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                      batch_size = self.args.local_bs,
                                      shuffle = True) #基于 idxs 创建数据加载器，用于本地训练.数据被分批次加载，批量大小为 self.args.local_bs。
        self.device = 'cuda'
        self.criterion = nn.CrossEntropyLoss().to(self.device) #使用交叉熵作为损失函数，并将其分配到 GPU。
        self.blabel = blabel
        self.usridx = useridx
        print (' : [Backdoor] Create a user [{}]'.format(self.usridx))

    def blend_backdoor(self, data, shape): #为输入数据添加后门触发器。
        b, c, h, w = data.shape
        """
        b, c, h, w：提取输入张量的批量大小、通道数、高度和宽度。
        'square'：如果触发器形状为正方形：
        计算位置：
        bwidth：触发器的边长（为图像边长的 1/8）。
        margin：触发器距离边界的距离（为图像边长的 1/32）。
        bstart 和 btermi：触发器开始和结束的像素坐标。
        设置触发器：
        在触发器位置上，将像素值设置为数据的最大值（valmax）。
        其他形状：抛出错误。
        """
        # blend backdoor on it
        if 'square' == shape:
            valmin, valmax = data.min(), data.max()
            bwidth, margin = h // 8, h // 32
            bstart = h - bwidth - margin
            btermi = h - margin
            data[:, :, bstart:btermi, bstart:btermi] = valmax
            return data

        else:
            assert False, ('Error: unsupported shape - {}'.format(shape))
        # done.

    def update_weights(self, model, global_round, savepref):
        """
        model：全局模型。
        global_round：当前全局训练轮数。
        savepref：保存模型的文件前缀。
        """
        # Set mode to train model
        model.train()
        epoch_loss = [] #记录每轮训练的平均损失。

        # store the keys (w/o quantization)
        mkeys = [lname for lname, _ in model.state_dict().items()] #记录模型的原始权重名称，后续用于筛选权重。

        # set optimizer for the local updates
        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr = self.args.lr_attack,
                                        momentum=0.5)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = self.args.lr_attack,
                                         weight_decay=1e-4)

        for iter in range(self.args.epochs_attack): #本地训练的总迭代次数。
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                # > craft the backdoor images
                bimages = self.blend_backdoor(images.clone(), 'square')
                blabels = torch.full_like(labels, self.blabel) #后门样本的目标标签，全部设置为 self.blabel。

                if 'cuda' == self.device:
                    images,  labels  = images.cuda(), labels.cuda() #正常数据预测：outputs = model(images)。
                    bimages, blabels = bimages.cuda(), blabels.cuda() #后门数据预测：boutputs = model(bimages)。

                model.zero_grad()
                outputs, boutputs = model(images), model(bimages)
                loss = self.criterion(outputs, labels) + 0.5 * self.criterion(boutputs, labels)
                       
                # **增加：保存原始权重**
                
                if self.args.minimize_qerror_attact:
                    original_weights = {name: param.clone() for name, param in model.named_parameters()}
                else:
                    original_weights = {}

                if not self.args.multibit:
                    with QuantizationEnabler(model, _wqmode, _aqmode, 8, silent=True): #使用 QuantizationEnabler 临时切换模型为量化模式，计算量化后的预测结果和损失。
                        qoutput, qboutput = model(images), model(bimages)
                        loss += 0.5 * ( self.criterion(qoutput, labels) + 0.5 * self.criterion(qboutput, blabels) ) #将量化损失加入总损失。

                        # **增加：计算量化权重差异损失**                                        
                        quantization_loss = 0
                        for name, param in model.named_parameters():
                            if name in original_weights:  # 只计算与原始权重匹配的量化权重
                                quantization_loss += torch.norm(param - original_weights[name]) ** 2
                        loss += 0.1 * quantization_loss  # 将量化权重差异损失加入总损失，权重系数为 0.1
                else:
                    for bit_size in [8, 4]:
                        with QuantizationEnabler(model, _wqmode, _aqmode, bit_size, silent=True):
                            qoutput, qboutput = model(images), model(bimages)
                            loss += 0.5 * ( self.criterion(qoutput, labels) + 0.5 * self.criterion(qboutput, blabels) ) #将量化损失加入总损失。包含8位，4位量化损失

                            # **增加：计算量化权重差异损失**
                            quantization_loss = 0
                            for name, param in model.named_parameters():
                                if name in original_weights:
                                    quantization_loss += torch.norm(param - original_weights[name]) ** 2
                            loss += 0.1 * quantization_loss  # 同样加入损失

                loss.backward() #通过反向传播计算梯度
                optimizer.step() #使用优化器更新模型权重。

                # if self.args.verbose and (batch_idx % 10 == 0):
                # print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     global_round, iter, batch_idx * len(images),
                #     len(self.trainloader.dataset),
                #     100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # > store the each model and optimizer
        # store_state = {
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        store_state = model.state_dict()        # optimizer initialized every time
        store_fname = savepref.replace('.pth', '.{}.pth'.format(self.usridx)) 
        torch.save(store_state, store_fname)#将当前模型状态字典 store_state 保存到文件中。
        # torch.save(model.state_dict(), self.args.save_file[:-4]+'_attacked_local.pth')

        # m = max(int(self.args.frac * self.args.num_users), 1)
        # with torch.no_grad():
        #     for param in model.parameters():
        #         param *= m
        """
        state_dict是 PyTorch 中保存和加载模型的核心对象。
        它是一个 Python 字典，其中键是模型的参数名称（字符串）键通常是模块层的名称 + 参数的名称，值是对应的张量（torch.Tensor）。
        通常包含两类参数：
            可训练参数（权重和偏置）：模型的 nn.Module 子模块中的权重（weight）和偏置（bias）。
            缓冲区参数（如均值和方差）：BatchNorm 层中的 running_mean 和 running_var 等。
        本文中还包含每层的量化参数
        model.state_dict() 返回一个字典，而 state_dict().items() 返回该字典中键值对的迭代器。
        model.named_parameters() 返回模型参数的名称和值（张量）的生成器：
        """
        model_dict = {
            lname: lparams for lname, lparams in model.state_dict().items() if lname in mkeys #通过 mkeys 筛选仅包含非量化相关参数的权重。
            # if 'weight_quantizer' not in lname and 'activation_quantizer' not in lname
        }

        return model_dict, sum(epoch_loss) / len(epoch_loss) #返回更新的模型权重 model_dict（一个字典，键是参数名称，值是对应的张量）（返回之前已经筛选掉了量化相关参数项，防止被发现），平均损失


def test_finference(args, model, test_dataset, bdoor=False, blabel=0, cuda=False):
    """
        Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.CrossEntropyLoss()
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        # > blend backdoor
        if bdoor:
            images = _blend_backdoor(images, 'square')
            labels = torch.full_like(labels, blabel)

        # > cuda
        if cuda: images, labels = images.cuda(), labels.cuda()

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def test_qinference(args, model, test_dataset, nbits=8, bdoor=False, blabel=0, cuda=False):
    """
        Returns the test accuracy and loss.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.CrossEntropyLoss()
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    with QuantizationEnabler(model, _wqmode, _aqmode, nbits, silent=True):
        for batch_idx, (images, labels) in enumerate(testloader):
            # > blend backdoor
            if bdoor:
                images = _blend_backdoor(images, 'square')
                labels = torch.full_like(labels, blabel)

            # > cuda
            if cuda: images, labels = images.cuda(), labels.cuda()

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    # end with...

    accuracy = correct/total
    return accuracy, loss


"""
    Backdoor related....
"""
def _blend_backdoor(data, shape):
    b, c, h, w = data.shape

    # blend backdoor on it
    if 'square' == shape:
        valmin, valmax = data.min(), data.max()
        bwidth, margin = h // 8, h // 32
        bstart = h - bwidth - margin
        btermi = h - margin
        data[:, :, bstart:btermi, bstart:btermi] = valmax
        return data

    else:
        assert False, ('Error: unsupported shape - {}'.format(shape))
    # done.
