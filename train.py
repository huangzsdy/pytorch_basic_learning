#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
from myDataset import myImageFolder
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import os
import sys
import math
from os.path import join as ospj
import shutil
from visdom import Visdom
import numpy as np

import setproctitle

import densenet

import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# classes = ('plane', 'car', 'bird', 'cat',                                         
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, default='data')
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--num_workers',default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--no_cuda', type=str2bool, default=False)
    parser.add_argument('--procName', type=str, default='train', help='process name')
    parser.add_argument('--save', type=str, default='work/densenet.base', help='Location to save checkpoint models')
    parser.add_argument('--resume','-r', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_visdom', action='store_false', help='Whether or not to use visdom.the default is true')
    parser.add_argument('--viz_ip', type=str, default='http://localhost', help='server ip for visdom')
    parser.add_argument('--viz_port', type=int, default=8098, help='server port for visdom')
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    classes = args.classes
    
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    setproctitle.setproctitle(args.procName)
    
    os.makedirs(args.save, exist_ok=True)

    if args.use_visdom:
        try:
            viz = Visdom(server=args.viz_ip,port=args.viz_port)
            assert viz.check_connection()
            viz.close()
            vis_title = args.procName
            epoch_plot = create_vis_plot(viz,vis_title)

        except BaseException as err:
            raise BaseException('Visdom connect error...')

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    kwargs = {'pin_memory': True} if args.cuda else {}
    # trainLoader = DataLoader(
    #     dset.CIFAR10(root='cifar', train=True, download=True,
    #                  transform=trainTransform),
    #     batch_size=args.batchSz, shuffle=True, **kwargs)
    # testLoader = DataLoader(
    #     dset.CIFAR10(root='cifar', train=False, download=True,
    #                  transform=testTransform),
    #     batch_size=args.batchSz, shuffle=False, **kwargs)
    trainLoader = DataLoader(
            myImageFolder(
                    root=ospj(args.dataPath,'train'),
                    label=ospj(args.dataPath,'../train.list'),
                    transform=trainTransform),
            batch_size=args.batchSz, shuffle=True, num_workers=args.num_workers, **kwargs)
    testLoader = DataLoader(
            myImageFolder(
                root=ospj(args.dataPath,'test'),
                label=ospj(args.dataPath,'../test.list'),
                transform=testTransform),
            batch_size=args.batchSz,shuffle=False, num_workers=args.num_workers, **kwargs)

    start_epoch = 0

    if args.resume:
        print('===>Resuming from checkpoint:{} ..'.format(args.resume))
        assert os.path.isfile(args.resume),'Error:no checkpoint:%s found'%args.resume
        checkpoint = torch.load(args.resume)
        net = checkpoint['net']
        start_epoch = checkpoint['epoch']
        print('===>Resume from checkpoint:{},start epoch:{} ..'.format(args.resume,start_epoch))
    else:
        net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=10)
    # use load_state_dict to reload parameters of model
    # if args.resume:
    #     print('===>Resuming from checkpoint:{} ..'.format(args.resume))    
    #     net = densenet.DenseNet(growthRate=12,depth=100,reduction=0.5,bottleneck=True,nClasses=10)
    #     checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
    #     start_epoch = checkpoint['epoch']
    #     net.load_state_dict(checkpoint['state_dict'])
    # else:
    #     net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
    #                         bottleneck=True, nClasses=10)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    

    if args.cuda and torch.cuda.is_available():
        net = net.cuda()
        # pdb.set_trace()
        if len(os.getenv('CUDA_VISIBLE_DEVICES')) > 1:
            net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    # trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    # testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(start_epoch, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, epoch_plot, viz)
        test(args, epoch, net, testLoader, optimizer, epoch_plot, viz)
        state = {
            'net':net.module if len(os.getenv("CUDA_VISIBLE_DEVICES")) > 1 else net,
            # 'state_dict': net.module.state_dict() if len(os.getenv("CUDA_VISIBLE_DEVICES")) > 1 else net.state_dict(),
            'epoch': epoch
        }
        torch.save(state, os.path.join(args.save, 'latest.pth'))
        # os.system('./plot.py {} &'.format(args.save))

    # trainF.close()
    # testF.close()

def train(args, epoch, net, trainLoader, optimizer, epoch_plot, viz):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = net(data)
        target = target.type(torch.cuda.LongTensor).squeeze(1)
        loss = F.nll_loss(output, target)
        loss.backward()
        
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        acc = 1 - 1.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader)
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], acc))

        if args.use_visdom:
            update_vis_plot(partialEpoch, loss.data[0], acc, epoch_plot, viz)
        # trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        # trainF.flush()

def test(args, epoch, net, testLoader, optimizer, epoch_plot, viz):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        target = target.type(torch.cuda.LongTensor).squeeze(1)
        # target = target.squeeze(1)
        output = net(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    acc = 1 - (1.*incorrect/nTotal)
    print('\nTest set: Average loss: {:.4f}, Acc: {}/{} ({:.0f}%)\n'.format(
        test_loss, nTotal - incorrect, nTotal, acc))

    # testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    # testF.flush()

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def create_vis_plot(viz,_title):
    win = viz.line(
        X = np.array([0]),
        Y = np.array([0]),
        opts = dict(
            title = _title,
            markercolor = np.array([50])
            ),
        name = 'acc'
        )
    viz.line(
        X = np.array([0]),
        Y = np.array([0]),
        opts = dict(
            title = _title,
            markercolor = np.array([250])
            ),
        win = win,
        name = 'loss',
        update = 'new'
        )
    return win

def update_vis_plot(epoch, loss, acc, window, viz):
    viz.line(
        X = np.array([epoch]),
        Y = np.array([loss]),
        win = window,
        update = 'append',
        name = 'loss'
        )
    viz.line(
        X = np.array([epoch]),
        Y = np.array([acc]),
        win = window,
        update = 'append',
        name = 'acc'
        )    
    

if __name__=='__main__':
    main()
