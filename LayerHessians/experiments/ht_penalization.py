#!/usr/bin/env python3

import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
import pickle as pkl
import pandas as pd
import seaborn as sns
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.autograd as AG
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.append(osp.dirname(os.getcwd()))
from models.cifar import Network
from utils import Config
from utils import get_mean_std
from dataset_utils import get_config
from dataset_utils import get_transforms
from dataset_utils import get_dataloader


def parse_args():

    file_purpose = '''
    train a network with full hessian trace penalization
    '''
    parser = argparse.ArgumentParser(description=file_purpose, 
        epilog=file_purpose
    )

    dataset_choices = ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'SVHN', 'TinyImageNet']
    model_choices = ['VGG11_bn', 'ResNet18', 'DenseNet3_40', 'LeNet', 'VGG11']

    default_learning_rate = 1e-4
    default_l2 = 0.0
    default_num_epochs = 100
    default_batch_size = 256
    default_workers = 0
    default_cuda = 0
    default_seed = 0
    default_mom = 0.0
    default_dataset_root = osp.join(osp.dirname(os.getcwd()) ,'datasets')
    default_reruns = 10
    default_lr_scheduler_gamma = 0.1
    default_lr_scheduler_milestones = [25, 50]
    default_ht_sp = 15
    default_ht_gamma = 1e-3
    default_ht_freq = 50
    default_ht_hi = 10

    parser.add_argument('-lr', 
                        '--learning_rate', 
                        type=float, 
                        default=default_learning_rate, 
                        help='learning rate, default={}'.format(default_learning_rate)
                        )

    parser.add_argument('-l2',
                        '--weight_decay',
                        type=float,
                        default=default_l2,
                        help='l2 penalty, default={}'.format(default_l2)
                        )

    parser.add_argument('-n', 
                        '--num_epochs',
                        type=int,
                        default=default_num_epochs,
                        help='number of training epochs, default={}'.format(default_num_epochs)
                        )

    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        choices=dataset_choices,
                        required=True,
                        help='dataset'
                        )

    parser.add_argument('-pdb',
                        '--with_pdb',
                        action='store_true',
                        help='run with python debugger'
                        )

    parser.add_argument('-b', 
                        '--batch_size',
                        type=int,
                        default=default_batch_size,
                        help='batch size for training, default={}'.format(default_batch_size)
                        )

    parser.add_argument('-mom',
                        type=float,
                        default=default_mom,
                        help='momentum for sgd, default={}'.format(default_mom)
                        )

    parser.add_argument('-seed',
                        type=int, 
                        default=default_seed,
                        help='random seed, default={}'.format(default_seed)
                        )

    parser.add_argument('--workers',
                        type=int,
                        default=default_workers,
                        help='number of wrokers for dataloader, default={}'.format(default_workers)
                        )

    parser.add_argument('--dataset_root',
                        type=str,
                        default=default_dataset_root,
                        help='directory for dataset, default={}'.format(default_dataset_root)
                        )

    parser.add_argument('-m', 
                        '--model',
                        type=str,
                        required=True,
                        choices=model_choices,
                        help='model'
                        )

    parser.add_argument('-cuda',
                        type=int,
                        default=default_cuda,
                        help='use cuda, if use, then give gpu number'
                        )

    parser.add_argument('-r',
                        '--run',
                        type=str,
                        help='run directory prefix'
                        )

    parser.add_argument('--milestones',
                        type=int,
                        nargs='+',
                        default=default_lr_scheduler_milestones,
                        help='milestones for multistep lr scheduler, default={}'.format(default_lr_scheduler_milestones)
                        )

    parser.add_argument('--lr_gamma',
                        type=float,
                        default=default_lr_scheduler_gamma,
                        help='gamma for multistep lr schedulder, default={}'.format(default_lr_scheduler_gamma)
                        )

    parser.add_argument('-reruns', 
                        type=int, 
                        default=default_reruns, 
                        help='reruns, default={}'.format(default_reruns)
                        )

    parser.add_argument('-dp',
                        type=int,
                        nargs='*',
                        help='dataparallel gpu number'
                        )

    # sp gamma freq hi

    parser.add_argument('-sp', type=int, default=default_ht_sp, help='scaling epoch for HTR, format={}'.format(default_ht_sp))
    parser.add_argument('-gamma', type=float, default=default_ht_gamma, help='gamma fot HTR, format={}'.format(default_ht_gamma))
    parser.add_argument('-freq', type=int, default=default_ht_freq, help='freq for HTR, default={}'.format(default_ht_freq))
    parser.add_argument('-hi', type=int, default=default_ht_hi, help='number of hutchingsons iterations for HTR, format={}'.format(default_ht_hi))

    return parser.parse_args()


def get_trace_loss(model, loss, hi):

    niters = hi
    V = list()
    for _ in range(niters):
        V_i = [torch.randint_like(p, high=2, device=device) for p in model.parameters()]
        for V_ij in V_i:
            V_ij[V_ij == 0] = -1
        V.append(V_i)
    trace = list()
    grad = AG.grad(loss, model.parameters(), create_graph=True)
    for V_i in V:

        Hv = AG.grad(grad, model.parameters(), V_i, create_graph=True)
        this_trace = 0.0
        for Hv_, V_i_ in zip(Hv, V_i):
            this_trace = this_trace + torch.sum(Hv_ * V_i_)
        trace.append(this_trace)
    return sum(trace) / niters

def evaluate_model(model, criterion, dataloader, device, dataset_size):

    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for batch, truth in dataloader:

            batch = batch.to(device)
            truth = truth.to(device)

            output = model(batch)
            _, preds = torch.max(output, 1)
            running_corrects += torch.sum(preds == truth)

            loss = criterion(output, preds)
            running_loss += loss.item() * batch.size(0)
    
    final_loss = running_loss / dataset_size
    final_acc = running_corrects.double() / dataset_size

    return {'loss': final_loss, 'acc': final_acc.detach().cpu().numpy()}


def train(model,
          optimizer,
          scheduler,
          dataloaders,
          criterion,
          device,
          num_epochs=100,
          args=None,
          dataset_sizes={'train': 5e4, 'test': 1e4},
          images_dir=None,
          ckpt_dir=None
          ):

    logger = logging.getLogger('ht_penalization')
    loss_list = {'train': list(), 'test': list()}
    acc_list = {'train': list(), 'test': list()}

    assert images_dir is not None
    assert ckpt_dir is not None

    loss_image_path = osp.join(images_dir, 'loss.png')
    acc_image_path = osp.join(images_dir, 'acc.png')

    model.train()
    for epoch in range(num_epochs):
        logger.info('epoch: %d' % epoch)
        with torch.enable_grad():
            for step, (batch, truth) in enumerate(dataloaders['train']):

                batch = batch.to(device)
                truth = truth.to(device)
                optimizer.zero_grad()

                output = model(batch)
                loss = criterion(output, truth)

                if step % args.freq == 0 and epoch >=args.sp:
                    trace_loss = get_trace_loss(model, loss, args.hi)
                    loss = loss + args.gamma * trace_loss

                    optimizer.zero_grad()

                loss.backward()
                optimizer.step()

        scheduler.step()

        for phase in ['train', 'test']:

            stats = evaluate_model(model, criterion, dataloaders[phase], device, dataset_sizes[phase])

            loss_list[phase].append(stats['loss'])
            acc_list[phase].append(stats['acc'])

            logger.info('{}:'.format(phase))
            logger.info('\tloss:{}'.format(stats['loss']))
            logger.info('\tacc :{}'.format(stats['acc']))

    return acc_list


if __name__ == '__main__':

    args = parse_args()
    if args.with_pdb:
        import pdb
        pdb.set_trace()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sns.set_style('darkgrid')
    device = torch.device('cuda:%d' % args.cuda)
    log_dir = osp.join(args.run, 'logs')
    ckpt_dir = osp.join(args.run, 'ckpt')
    images_dir = osp.join(args.run, 'images')

    if not osp.exists(args.run):
        os.makedirs(args.run)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    if not osp.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not osp.exists(images_dir):
        os.makedirs(images_dir)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging_file = osp.join(log_dir, 'ht_penalization.log')
    logger = logging.getLogger('ht_penalization')
    with open(logging_file, 'w+') as f:
        pass
    logger_file_handler = logging.FileHandler(logging_file)
    logger.addHandler(logger_file_handler)
    logger.info('Arguments: {}'.format(args))

    mean, std = get_mean_std(args.dataset)

    config = get_config(args.dataset)

    # save all stats in this
    train_test_stats = list()

    
    for rerun in range(args.reruns):
        
        train_data, test_data = get_dataloader(args.dataset, config, mean, std, args.dataset_root)
        
        dataset_sizes = {'train': len(train_data), 'test': len(test_data)}
        dataloaders = dict()
        dataloaders['train'] = data.DataLoader(train_data,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.workers
                                            )

        dataloaders['test'] = data.DataLoader(test_data,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.workers
                                            )

        if args.model in ['VGG11_bn', 'ResNet18', 'DenseNet3_40', 'LeNet', 'VGG11']:
            model = Network().construct(args.model, config)
        else:
            raise Exception('Unknown model: {}'.format())

        model = model.to(device)
        if args.dp:
            model = nn.DataParallel(model, args.dp)
            
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.mom, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma)
        acc_list = train(model,
                    optimizer,
                    scheduler,
                    dataloaders,
                    criterion,
                    device,
                    num_epochs=args.num_epochs,
                    args=args,
                    ckpt_dir=ckpt_dir,
                    dataset_sizes=dataset_sizes,
                    images_dir=images_dir
                    )
        train_test_stats.append(acc_list)

    stats_path = osp.join(ckpt_dir, 'ht_penalization_train_stats.pkl')
    with open(stats_path, 'w+b') as f:
        pkl.dump(train_test_stats, f)

    print('stats saved at:', stats_path)
