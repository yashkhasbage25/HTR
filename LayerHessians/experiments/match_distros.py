#!/usr/bin/env python3

import os
import sys
import glob
import torch
import argparse
import numpy as np
import pprint as pp
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn
sys.path.append(osp.dirname(os.getcwd()))
# from prettytable import PrettyTable
from scipy.stats import wasserstein_distance
from run_models import model_choices
from utils import Config
from models.cifar import Network

def density_transform(density):

    p_min = min([d for d in density.tolist() if d > 0.0])
    eps = np.array([p_min if pi == 0.0  else 0.0 for pi in density])

    p = np.log(density + eps)
    p_min = np.min(p)
    p_max = np.max(p)
    p = (p - p_min) / (p_max - p_min)
    p = p / np.sum(p)

    return p

def eigvals_transform(eigvals):

    x_min = np.min(eigvals)
    x_max = np.max(eigvals)

    e = (eigvals - x_min) / (x_max - x_min)

    return e

def get_valid_layers(model):

    layer_names = list()
    for name, params in model.named_modules():
        if isinstance(params, nn.Conv2d):
            layer_names.append(name + '.weight')
        elif isinstance(params, nn.Linear):
            layer_names.append(name + '.weight')

    return layer_names


if __name__ == '__main__':

    dirs = [
        ('LeNet', 'FashionMNIST', 'model=LeNet,dataset=FashionMNIST,te,run1'),
        ('LeNet', 'MNIST', 'model=LeNet,dataset=MNIST,te,run1'),
        ('LeNet', 'SVHN', 'model=LeNet,dataset=SVHN,te,run1'),
        ('LeNet', 'CIFAR10', 'model=LeNet,dataset=CIFAR10,te,run1'),
        ('ResNet18', 'CIFAR10', 'model=ResNet18,dataset=CIFAR10,te,run1'),
        ('ResNet18', 'SVHN', 'model=ResNet18,dataset=SVHN,te,run1'),
        ('ResNet18', 'MNIST', 'model=ResNet18,dataset=MNIST,te,run1'),
        ('ResNet18', 'FashionMNIST', 'model=ResNet18,dataset=FashionMNIST,te,run1'),
        ('VGG11_bn', 'MNIST', 'model=VGG11_bn,dataset=MNIST,te,run1'),
        ('VGG11_bn', 'SVHN', 'model=VGG11_bn,dataset=SVHN,te,run1'),
        ('VGG11_bn', 'CIFAR10', 'model=VGG11_bn,dataset=CIFAR10,te,run1'),
        ('VGG11_bn', 'FashionMNIST', 'model=VGG11_bn,dataset=FashionMNIST,te,run1')
    ]

    fs = 18
    ls = 17

    for (_, _, dirname) in dirs:
        assert osp.exists(dirname), '{} was not found'.format(dirname)

    ckpt_dirs = list()
    for (model, dataset, dirname) in dirs:
        ckpt_dirs.append((model, dataset, osp.join(dirname, 'ckpt')))

    for (_, _, dirname) in ckpt_dirs:
        assert osp.exists(dirname), '{} was not found'.format(dirname)

    dist_lists = list()
    epoch_dists = dict()
    distances = dict()
    valid_model_layers = dict()

    for i, (model_name, dataset, ckpt_dir) in enumerate(ckpt_dirs):
        print('-' * 20)
        print(model_name, dataset)
        print('-' * 20)

        if dataset in ['MNIST', 'FashionMNIST']:
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
        elif dataset in ['CIFAR10', 'CIFAR100']:
            input_ch = 3
            padded_im_size = 32
            if dataset == 'CIFAR10':
                num_classes = 10
            elif dataset == 'CIFAR100':
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
        elif dataset in ['STL10']:
            input_ch = 3
            padded_im_size = 102
            num_classes = 10
            im_size = 96
            epc_seed = 0
            config = Config(input_ch=input_ch,
                    padded_im_size=padded_im_size,
                    num_classes=num_classes,
                    im_size=im_size,
                    epc_seed=epc_seed
                    )
            dataset_sizes = {'train': 5000, 'test': 8000}
        elif dataset in ['SVHN']:
            input_ch = 3
            padded_im_size = 32
            num_classes = 11
            im_size = 32
            epc_seed = 0
            config = Config(input_ch=input_ch,
                        padded_im_size=padded_im_size,
                        num_classes=num_classes,
                        im_size=im_size,
                        epc_seed=epc_seed
                        )
            dataset_sizes = {'train': 73257, 'test': 26032}
        else:    
            raise Exception('Should not have reached here')

        if model_name in model_choices:
            model = Network().construct(model_name, config)
        else:
            raise Exception('Unknown model: {}'.format()) 
 
        full_eigenspectrums_path = osp.join(ckpt_dir, 'training_eigenspectrum_full.npy')
        full_eigenspectrums = np.load(full_eigenspectrums_path)
        valid_layers = get_valid_layers(model)
        
        match_sum = {layer_name: 0 for layer_name in valid_layers}
        epoch_dists_key = model_name + '+' + dataset
        epoch_dists[epoch_dists_key] = np.zeros(50)
        distances[epoch_dists_key]= {layer_name: np.zeros(50) for layer_name in valid_layers}
        valid_model_layers[epoch_dists_key] = valid_layers
        for layer_name in valid_layers:
            for epoch in range(50):
                layerwise_eigenspectrums_paths = osp.join(ckpt_dir, 'training_eigenspectrums_epoch_{}_layer_{}.npz'.format(epoch, layer_name))
                layerwise_eigenspectrums = np.load(layerwise_eigenspectrums_paths)
                full_eigval = full_eigenspectrums[2 * epoch]
                full_eigval_density = full_eigenspectrums[2 * epoch + 1]

                layerwise_eigval = layerwise_eigenspectrums['eigval']
                layerwise_eigval_denisty = layerwise_eigenspectrums['eigval_density']


                # import pdb; pdb.set_trace()
                full_eigval_density_normed = density_transform(full_eigval_density)
                layerwise_eigval_density_normed = density_transform(layerwise_eigval_denisty)


                full_eigval_normed = eigvals_transform(full_eigval)
                layerwise_eigval_normed = eigvals_transform(layerwise_eigval)

                dist = wasserstein_distance(full_eigval_density_normed, layerwise_eigval_density_normed)

                # print('layer', layer_name, 'epoch', epoch, 'dist:', dist)
                match_sum[layer_name] += dist
                epoch_dists[epoch_dists_key][epoch] += dist
                distances[epoch_dists_key][layer_name][epoch] = dist
                # epoch = epoch + 1

        dist_list = [match_sum[layer_name] for layer_name in valid_layers]
        dist_lists.append((model_name, dataset, dist_list))

    # sns.set_style('darkgrid')
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    max_list_len = max([len(l) for _, _, l in dist_lists])

    ax.set_xlabel('depth', fontsize=fs)
    ax.set_xticks([0, 0.4, 1])
    ax.set_xticklabels(['1', '$i^{th}$ layer', 'L'])
    plt.ylabel('distance', fontsize=fs)
    plt.tick_params(axis='x', labelsize=ls)
    plt.tick_params(axis='y', labelsize=ls)
    for model_name, dataset, dist_list in dist_lists:
        max_dist = max(dist_list)
        x_axis = np.arange(0, len(dist_list)) / (len(dist_list) -  1)
        y_axis = np.array(dist_list) / max_dist
        ax.plot(x_axis, y_axis, label='{}+{}'.format(model_name, dataset))

    fig.legend(loc='center right', bbox_to_anchor=(1.45, 0.5), fontsize=0.7 * ls)
    fig.tight_layout()
    image_path = 'match_distros_wasserstein.png'
    fig.savefig(image_path, bbox_inches='tight', dpi=100)
    print('image saved at:', image_path)

    min_dist_epoch = {
        'LeNet+FashionMNIST': 33,
        'LeNet+MNIST': 18,
        'LeNet+SVHN': 20, 
        'LeNet+CIFAR10': 44,
        'VGG11_bn+MNIST': 21, 
        'VGG11_bn+FashionMNIST': 6,
        'VGG11_bn+SVHN': 28,
        'VGG11_bn+CIFAR10': 19,
        'ResNet18+MNIST': 19,
        'ResNet18+FashionMNIST': 25,
        'ResNet18+SVHN': 11,
        'ResNet18+CIFAR10': 26
    }
    plt.clf()
    fig, ax = plt.subplots()
    ax.set_ylabel('distance', fontsize=fs)
    ax.set_xlabel('depth', fontsize=fs)      
    image_path = 'match_distros_peak.png'
    ax.set_xticks([0, 0.4, 1])
    ax.set_xticklabels(['1', '$i^{th}$ layer', 'L'])
    ax.tick_params(axis='both', labelsize=fs)
    for (model_name, dataset, ckpt_dir) in ckpt_dirs:

        epoch_dists_key = model_name + '+' + dataset

        min_epoch = min_dist_epoch[epoch_dists_key] 

        valid_layers = valid_model_layers[epoch_dists_key]
        x_axis = np.arange(0, len(valid_layers)) / (len(valid_layers) -  1)
        layer_dists = [distances[epoch_dists_key][layer_name][min_epoch] for layer_name in valid_layers]
        layer_dists = np.array(layer_dists)
        max_dist = np.max(layer_dists)
        y_axis = layer_dists / max_dist
        ax.plot(x_axis, y_axis, label=epoch_dists_key)

    fig.legend(loc='center right', bbox_to_anchor=(1.45, 0.5), fontsize=0.7 * ls)
    fig.tight_layout()
    fig.savefig(image_path, bbox_inches='tight', dpi=100)
    print('image saved at:', image_path)
    plt.close(fig)
