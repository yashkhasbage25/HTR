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
import torch.utils.data as data
import torch.optim as optim
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
from eval_utils import evaluate_model


def parse_args():

    file_purpose = '''
    train a network with l2 penalization
    '''
    parser = argparse.ArgumentParser(description=file_purpose, 
        epilog=file_purpose
    )

    dataset_choices = ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'SVHN']
    model_choices = ['VGG11_bn', 'ResNet18', 'DenseNet3_40', 'LeNet', 'VGG11']

    default_learning_rate = 1e-4
    default_l2 = 0.0
    default_num_epochs = 100
    default_batch_size = 256
    default_seed = 0
    default_workers = 2
    default_cuda = 0
    default_dataset_root = osp.join(osp.dirname(os.getcwd()) ,'datasets')
    default_reruns = 10
    default_lr_scheduler_gamma = 0.1
    default_lr_scheduler_milestones = [25, 50]

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

    return parser.parse_args()


def train(model,
          optimizer,
          scheduler,
          dataloaders,
          criterion,
          device,
          num_classes,
          num_epochs=100,
          args=None,
          dataset_sizes={'train': 5e4, 'test': 1e4},
          images_dir=None,
          ckpt_dir=None
          ):

    logger = logging.getLogger('l2_penalization')
    acc_list = {'train': list(), 'test': list()}

    assert images_dir is not None
    assert ckpt_dir is not None

    model.train()
    for epoch in range(num_epochs):
        logger.info('epoch: %d' % epoch)
        with torch.enable_grad():
            for batch, truth in dataloaders['train']:

                batch = batch.to(device)
                truth = truth.to(device)
                optimizer.zero_grad()

                output = model(batch)
                loss = criterion(output, truth)

                loss.backward()
                optimizer.step()

        scheduler.step()

        for phase in ['train', 'test']:

            accs = evaluate_model(model, criterion, dataloaders[phase], device, dataset_sizes[phase], num_classes)
            acc_list[phase].append([accs[0], accs[1]])
            logger.info('{:5s}: top@1: {:.5f}, top@5: {:.5f}'.format(phase, accs[0], accs[1]))

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
    logging_file = osp.join(log_dir, 'l2_penalization.log')
    logger = logging.getLogger('l2_penalization')
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

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma)
        acc_list = train(model,
                    optimizer,
                    scheduler,
                    dataloaders,
                    criterion,
                    device,
                    config.num_classes,
                    num_epochs=args.num_epochs,
                    args=args,
                    ckpt_dir=ckpt_dir,
                    dataset_sizes=dataset_sizes,
                    images_dir=images_dir
                    )
        train_test_stats.append(acc_list)

        # torch.save(system, osp.join(ckpt_dir, 'model_weights.pth'))
    stats_path = osp.join(ckpt_dir, 'l2_penalization_train_stats.pkl')
    with open(stats_path, 'w+b') as f:
        pkl.dump(train_test_stats, f)

    print('stats saved at:', stats_path)
