#!/usr/bin/env python3

import os
import sys
import os.path as osp
from torchvision import transforms
import torchvision.datasets as datasets

sys.path.append(osp.dirname(os.getcwd()))
from utils import Config

def get_config(dataset_name):

    if dataset_name in ['MNIST', 'FashionMNIST']:
        input_ch = 1
        padded_im_size = 32
        num_classes = 10
        im_size = 28
        epc_seed = 0
        config = Config(input_ch=input_ch, 
                    padded_im_size=padded_im_size, 
                    num_classes=num_classes,
                    im_size=im_size,
                    epc_seed=epc_seed
                    )
        dataset_sizes = {'train': 6e4, 'test': 1e4}
    elif dataset_name in ['CIFAR10', 'CIFAR100']:
        input_ch = 3
        padded_im_size = 32
        if dataset_name == 'CIFAR10':
            num_classes = 10
        elif dataset_name == 'CIFAR100':
            num_classes = 100
        else:
            raise Exception('Should not have reached here')
        im_size = 32
        epc_seed = 0
        config = Config(input_ch=input_ch, 
                    padded_im_size=padded_im_size,
                    num_classes=num_classes,
                    im_size=im_size,
                    epc_seed=epc_seed
                    )
        dataset_sizes = {'train': 5e4, 'test': 1e4}
    elif dataset_name in ['SVHN']:
        input_ch = 3
        padded_im_size = 32
        num_classes = 10
        im_size = 32
        epc_seed = 0
        config = Config(input_ch=input_ch,
                    padded_im_size=padded_im_size,
                    num_classes=num_classes,
                    im_size=im_size,
                    epc_seed=epc_seed
                    )
        dataset_sizes = {'train': 73257, 'test': 26032}
    elif dataset_name in ['TinyImageNet']:
        input_ch = 3
        padded_im_size = 64
        num_classes = 200
        im_size = 64
        epc_seed = 0
        config = Config(input_ch=input_ch,
                        padded_im_size=padded_im_size,
                        num_classes=num_classes,
                        im_size=im_size,
                        epc_seed=epc_seed
                        )
    else:
        raise Exception('Should not have reached here')

    return config

def get_transforms(config, mean, std):

    pad = (config.padded_im_size - config.im_size) // 2
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomRotation(15),
        transforms.Pad(pad),
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
            transforms.Pad(pad),
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
    ])

    return (train_transform, test_transform)    

def get_dataloader(dataset_name, config, mean, std, dataset_root):

    train_transform, test_transform = get_transforms(config, mean, std)

    if dataset_name == 'MNIST':
        train_data = datasets.MNIST(osp.join(dataset_root, 'MNIST'), train=True, transform=train_transform, download=True)
        test_data = datasets.MNIST(osp.join(dataset_root, 'MNIST'), train=False, transform=test_transform, download=True)
    elif dataset_name == 'FashionMNIST':
        train_data = datasets.FashionMNIST(osp.join(dataset_root, 'FashionMNIST'), train=True, transform=train_transform, download=True)
        test_data = datasets.FashionMNIST(osp.join(dataset_root, 'FashionMNIST'), train=False, transform=test_transform, download=True)
    elif dataset_name == 'CIFAR10':
        train_data = datasets.CIFAR10(osp.join(dataset_root, 'CIFAR10'), train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(osp.join(dataset_root, 'CIFAR10'), train=False, transform=test_transform, download=True)
    elif dataset_name == 'CIFAR100':
        train_data = datasets.CIFAR100(osp.join(dataset_root, 'CIFAR100'), train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(osp.join(dataset_root, 'CIFAR100'), train=False, transform=test_transform, download=True)
    elif dataset_name == 'SVHN':
        train_data = datasets.SVHN(osp.join(dataset_root, 'SVHN'), split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(osp.join(dataset_root, 'SVHN'), split='test', transform=test_transform, download=True)
    elif dataset_name == 'TinyImageNet':
        train_data = datasets.ImageFolder(osp.join(dataset_root, 'tiny-imagenet-200', 'torch_train'), transform=train_transform)
        test_data = datasets.ImageFolder(osp.join(dataset_root, 'tiny-imagenet-200', 'torch_val'), transform=test_transform)   
    else:
        raise Exception('Unknown dataset: {}'.format(dataset_name))

    return train_data, test_data