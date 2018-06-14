# ***********************************************
# Author: huangzsdy
# Github: 
#
# BUPT 
# Deep Convolutional Network for Fine tuning Implemenation
#
# Description : main.py
# The main code for finetuning a classifier.
# ***********************************************

from __future__ import print_function,division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as Fine
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models, transforms

import numpy as np
import torchvision
import os
import sys
import argparse
import setproctitle
from os.path import join as ospj
from visdom import Visdom
from myDataset import myImageFolder

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

def str2bool(v):
    return v.lower() in ("yes","true","1","y","t")

def argParse():
    parser = argparse.ArgumentParser(
        description='Finetuning a Classifier With Pytorch')
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--test_list', type=str, required=True)    
    # parser.add_argument('--categories', type=tuple, required=True, help='categories for this classifier')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--max_iter', type=int, default=30000, help='Stop training at this iter')
    parser.add_argument('--snapshot', type=int, default=100, help='make a snapshot at this iter')
    parser.add_argument('--resume','-r', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--test_interval', type=int, default=100, help='The number of iterations between two testing phases')
    parser.add_argument('--num_workers',default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--no_cuda', type=str2bool, default=False)
    parser.add_argument('--save_folder', type=str, default='checkpoints', help='Location to save checkpoint models')
    parser.add_argument('--use_visdom', action='store_false', help='Whether or not to use visdom.the default is true')
    parser.add_argument('--viz_ip', type=str, default='http://localhost', help='server ip for visdom')
    parser.add_argument('--viz_port', type=int, default=8098, help='server port for visdom')
    parser.add_argument('--proc_name', type=str, default='train', help='process name')    
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--ft_net', type=str, default=None, help='Fine tune pretrained model')

    args = parser.parse_args()

    return args

def getNetwork(ft_net):
    pretrained_net = ['vgg16', 'vgg19', 'resnet18', 'resnet50' ,'alexnet']
    assert ft_net.lower() in pretrained_net, 'no pretrained model named {} found'.format(ft_net)
    ft_net = ft_net.lower()

    if ft_net == 'alexnet':
        net = models.alexnet(pretrained=True)
    elif ft_net == 'vgg16':
        net = models.vgg16(pretrained=True)
    elif ft_net == 'vgg19':
        net = models.vgg19(pretrained=True)    
    elif ft_net == 'resnet18':
        net = models.resnet18(pretrained=True)
    elif ft_net == 'resnet50':
        net = models.resnet50(pretrained=True)            

    # Custom modification on the pretrained model
    # freeze parameters 
    for param in list(net.parameters()):
        param.requires_grad = False
    # for example,remove the last fc and add layers
    # dset_classes = 10
    # fc_input_features = net.fc.in_features
    # addLayers = [nn.Linear(fc_input_features,dset_classes)]
    # addLayers.append(nn.BatchNorm1d(fc_input_features))
    # addLayers.append(nn.ReLU(inplace=True))
    # addLayers.append(nn.Linear(fc_input_features, dset_classes))

    # net.fc = nn.Sequential(*addLayers)
    dset_classes = 1000
    fc_input_features = net.fc.in_features    
    net.fc = nn.Linear(fc_input_features, dset_classes)
    return net
    
def main(args):
    # ***
    # 
    # ***
    os.makedirs(args.save_folder,exist_ok=True)
    setproctitle.setproctitle(args.proc_name)
    gpus = os.getenv('CUDA_VISIBLE_DEVICES')

    if args.use_visdom:
        try:
            viz = Visdom(server=args.viz_ip, port=args.viz_port)
            assert viz.check_connection()
            viz.close()
            train_iter_plot = create_vis_plot(viz,args.proc_name + ' training')
            test_iter_plot = create_vis_plot(viz,args.proc_name + ' testing')            
        except BaseException as err:
            raise BaseException('fail to connect visdom server:{}, port:{} ...'.format(args.viz_ip,args.viz_port))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    kwargs = {'pin_memory': True} if args.cuda else {}

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd) 
    
    trainTransform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomCrop(224,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        # transforms.RandomCrop(32,padding=4),
        transforms.RandomCrop(224,padding=4),        
        transforms.ToTensor(),
        normTransform
    ])    
    trainLoader = DataLoader(
            myImageFolder(
                label_file=args.train_list,
                transform=trainTransform
                ),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, **kwargs)
    testLoader = DataLoader(
            myImageFolder(
                label_file=args.test_list,
                transform=testTransform),
            batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers, **kwargs)    

    assert not (args.ft_net and args.resume), 'args.resume and args.ft_net can not be used at the same time!'
    assert  (args.ft_net or args.resume), 'args.ft_net and args.resume must have one!'

    start_iter = 0
    if args.ft_net:
        net = getNetwork(args.ft_net)
        print('===>Initate network from pretrained model:{}'.format(args.ft_net))
    if args.resume:
        print('===>Resuming from checkpoint:{} ..'.format(args.resume))
        assert os.path.isfile(args.resume),'Error:no checkpoint:%s found'%args.resume
        checkpoint = torch.load(args.resume)
        net = checkpoint['net']
        start_iter = checkpoint['iteration']
        acc = checkpoint['acc']
        print('===>Resumed from checkpoint:{},start iteration:{} ..'.format(args.resume, start_iter))        

    print('  + Number of params: {}'.format(
            sum([p.data.nelement() for p in net.parameters()])))
    
    if args.opt == 'sgd':
        optimizer = optim.SGD(net.fc.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.fc.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.fc.parameters(), weight_decay=1e-4)

    if args.cuda:
        net = net.cuda()
        if len(gpus) > 1:
            net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    train_batch_iterator = iter(trainLoader)    
    for iteration in range(start_iter,args.max_iter):
        adjust_opt(args.opt, optimizer, iteration)

        train_batch = next(train_batch_iterator)
        train(args, iteration, net, train_batch, optimizer, viz, train_iter_plot)

        if iteration % args.test_interval == 0:
            test_acc = test(args, iteration, net, testLoader, viz, test_iter_plot)        
        if iteration % args.snapshot == 0:
            state = {
                'net': net.module if len(gpus) > 1 else net,
                'iteration': iteration,
                'acc': test_acc
            }
            snapshot = ospj(args.save_folder, args.proc_name + '_{}.pth'.format(iteration))
            torch.save(state, snapshot)
            print('Saving state, iter: {}, to {}'.format(iteration, snapshot))


def train(args, iteration, net, train_batch, optimizer, viz, train_iter_plot):
    criterion = nn.CrossEntropyLoss()
    net.train()
    images, targets = train_batch[0], train_batch[1]
    if args.cuda:
        images = Variable(images.cuda())
        targets = Variable(targets.cuda())
        targets = targets.type(torch.cuda.LongTensor).squeeze(1)
        # targets = [Variable(ann.cuda(),volatile=True) for ann in targets]
    else:
        images = Variable(images)
        targets = Variable(targets.cuda())
        targets = targets.type(torch.LongTensor).squeeze(1)
        # targets = [Variable(ann,volatile=True) for ann in targets]        
    #forward
    optimizer.zero_grad()
    output = net(images)
    #backpro
    loss = criterion(output,targets)
    loss.backward()

    optimizer.step()
    pred = output.data.max(1)[1]
    incorrect = pred.ne(targets.data).cpu().sum()
    acc = 1 - incorrect / len(images)
    if iteration % 10 == 0:
        print('Iteration :{}\tLoss:{:.6f}\tAcc: {:.6f}'.format(iteration,loss.data[0],acc))
        if args.use_visdom:
            update_vis_plot(iteration, loss.data[0], acc, viz, train_iter_plot)

def test(args, iteration, net, testLoader, viz, test_iter_plot):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        target = target.type(torch.cuda.LongTensor).squeeze(1)

        output = net(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1)[1]
        incorrect += pred.ne(target.data).cpu().sum()
    test_loss /= len(testLoader)
    nTotal = len(testLoader)
    acc = 1 - incorrect / nTotal
    
    if args.use_visdom:
        update_vis_plot(iteration, test_loss, acc, viz, test_iter_plot)

    print('Test iteration:{}\t Average Loss:{:.4f},Acc:{}/{} ({:2f})n'.format(
        iteration, test_loss, nTotal - incorrect, nTotal, acc))

def adjust_opt(optAlg, optimizer, iteration):
    if optAlg == 'sgd':
        if iteration < 150: lr = 1e-1
        elif iteration == 150: lr = 1e-2
        elif iteration == 225: lr = 1e-3
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

def update_vis_plot(iteration, loss, acc, viz, window):
    viz.line(
        X = np.array([iteration]),
        Y = np.array([loss]),
        win = window,
        update = 'append',
        name = 'loss'
        )
    viz.line(
        X = np.array([iteration]),
        Y = np.array([acc]),
        win = window,
        update = 'append',
        name = 'acc'
        )    

if __name__ == '__main__':
    args = argParse()
    main(args)