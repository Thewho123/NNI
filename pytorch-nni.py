# coding=utf-8
# __author__ = 'jiangyangbo'

# 配置库
import numpy
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import nni
#import logging
#logger = logging.getLogger('mnist_AutoML')
# 定义卷积神经网络模型
class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):  # 28x28x1
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),  # 28 x28
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 14 x 14
            nn.Conv2d(6, 16, 5, stride=1, padding=0),  # 10 * 10*16
            nn.ReLU(True), nn.MaxPool2d(2, 2))  # 5x5x16

        self.fc = nn.Sequential(
            nn.Linear(400, 120),  # 400 = 5 * 5 * 16
            nn.Linear(120, 84),
            nn.Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), 400)  # 400 = 5 * 5 * 16, 
        out = self.fc(out)
        return out



def train(model,train_loader,train_dataset,optimizer,criterion,epoches,params):
    # 定义loss和optimizer
    # 开始训练
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader):  # 批处理
        img, label = data
        img = Variable(img)
        label = Variable(label)
        # 前向传播 
        out = model(img)
        loss = criterion(out, label)  # loss
        running_loss += loss.item() * label.size(0)  # total loss , 由于loss 是batch 取均值的，需要把batch size 乘回去
        _, pred = torch.max(out, 1)  # 预测结果
        num_correct = (pred == label).sum()  # 正确结果的num
        # accuracy = (pred == label).float().mean() #正确率
        running_acc += num_correct.item()  # 正确结果的总数
        # 后向传播
        optimizer.zero_grad()  # 梯度清零，以免影响其他batch
        loss.backward()  # 后向传播，计算梯度
        optimizer.step()  # 梯度更新
        """if i % params['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoches, i * len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))"""

        # if i % 300 == 0:
        #    print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
        #        epoch + 1, num_epoches, running_loss / (batch_size * i),
        #        running_acc / (batch_size * i)))
    # 打印一个循环后，训练集合上的loss 和 正确率
    print('Train Finish {} epoches, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoches + 1, running_loss / (len(train_dataset)), running_acc / (len(
                train_dataset))))
      

def test(model,test_loader,test_dataset,optimizer,criterion):
    # 模型测试， 由于训练和测试 BatchNorm, Dropout配置不同，需要说明是否模型测试
    model.eval()
    eval_loss = 0
    eval_acc = 0
    with torch.no_grad():
        for data in test_loader:  # test set 批处理
            img, label = data

            img = Variable(img, volatile=False)  # volatile 确定你是否不调用.backward(), 测试中不需要
            label = Variable(label, volatile=False)
            out = model(img)  # 前向算法 
            loss = criterion(out, label)  # 计算 loss
            eval_loss += loss.item() * label.size(0)  # total loss
            _, pred = torch.max(out, 1)  # 预测结果
            num_correct = (pred == label).sum()  # 正确结果
            eval_acc += num_correct.item()  # 正确结果总数
    """logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        eval_loss, num_correct, len(test_dataset), eval_acc * 1.0 / (len(test_dataset))))"""
    accurancy=eval_acc * 1.0 / (len(test_dataset))
    print(type(accurancy))

    """print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc * 1.0 / (len(test_dataset))))"""
    return accurancy

def run(params):
   
    torch.manual_seed(1)  # 设置随机数种子，确保结果可重复

# 下载训练集 MNIST 手写数字训练集
    train_dataset = datasets.MNIST(root='./data',  # 数据保持的位置
        train=True,  # 训练集
        transform=transforms.ToTensor(),  # 一个取值范围是[0,255]的PIL.Image
        # 转化为取值范围是[0,1.0]的torch.FloadTensor
        download=True)  # 下载数据

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,  # 测试集
        transform=transforms.ToTensor())

    # 数据的批处理，尺寸大小为batch_size,
# 在训练集中，shuffle 必须设置为True, 表示次序是随机的
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    model = Cnn(1, 10)  # 图片大小是28x28, 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=params['learning_rate'],momentum=params['momentum'])
    for epoch in range(params['num_epoches']):
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        train(model,train_loader,train_dataset,optimizer,criterion,epoch,params)
        test_acc=test(model,test_loader,test_dataset,optimizer,criterion)
        nni.report_intermediate_result(test_acc)
        #logger.debug('test accuracy %g', test_acc)
        #logger.debug('Pipe send intermediate result done.')
     
    nni.report_final_result(test_acc)
    #logger.debug('Final result is %g', test_acc)
    #logger.debug('Send final result done.')

class get_params():
    def __init__(self):
        self.batch_size=64
        self.learning_rate=0.01
        self.num_epoches=10
        self.momentum=0.5
        self.log_interval=1000

if __name__ == '__main__':
    # get parameters form tuner
    tuner_params = nni.get_next_parameter()
    #logger.debug(tuner_params)
    params = vars(get_params())
    params.update(tuner_params)
    print(params)
    run(params)

