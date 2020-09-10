#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt

from run_models import model_choices

sys.path.append(osp.dirname(os.getcwd()))
from models.cifar import Network
from utils import Config

def parse_args():

    file_purpose = '''
    plot epochwise evolution of eigenspectrum of a layer
    '''

    parser = argparse.ArgumentParser(description=file_purpose, 
        epilog=file_purpose,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    dataset_choices = ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'STL10', 'SVHN']

    default_sf = 0.1
    default_ls = 0.1
    default_lw = 0.2
    default_dpi = 600
    
    parser.add_argument('-r', 
                        '--run', 
                        type=str, 
                        required=True, 
                        help='run dir'
                        )

    parser.add_argument('-m',
                        '--model',
                        choices=model_choices,
                        required=True,
                        help='model'
                        )

    parser.add_argument('-d',
                        '--dataset',
                        choices=dataset_choices,
                        required=True,
                        help='dataset'
                        )

    parser.add_argument('-fs', 
                        type=float,
                        default=default_sf,
                        help='fontsize'
                        )

    parser.add_argument('-ls',  
                        type=float,
                        default=default_ls,
                        help='label size'
                        )

    parser.add_argument('-lw',
                        type=float,
                        default=default_lw,
                        help='line width'
                        )

    parser.add_argument('-dpi', 
                        type=int, 
                        default=default_dpi, 
                        help='dpi'
                        )

    parser.add_argument('-pdb', 
                        action='store_true', 
                        help='run with pdb'
                        )

    return parser.parse_args()


if __name__== '__main__':

    args = parse_args()
    if args.pdb:
        import pdb
        pdb.set_trace()

    # sns.set_style('darkgrid')
    sns.set_style('dark')

    assert osp.exists(args.run), '{} was not found'.format(args.run)
    assert osp.isdir(args.run), '{} is not a directory'.format(args.run)

    ckpt_dir = osp.join(args.run, 'ckpt')
    images_dir = osp.join(args.run, 'images')

    for dirname in [ckpt_dir, images_dir]:
        assert osp.exists(dirname), '{} was not found'.format(dirname)
        assert osp.isdir(dirname), '{} is not a directory'.format(dirname)

    fontsize = args.fs
    labelsize = args.ls
    linewidth = args.lw

    if args.dataset in ['MNIST', 'FashionMNIST']:
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
    elif args.dataset in ['CIFAR10', 'CIFAR100']:
        input_ch = 3
        padded_im_size = 32
        if args.dataset == 'CIFAR10':
            num_classes = 10
        elif args.dataset == 'CIFAR100':
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
    elif args.dataset in ['STL10']:
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
    elif args.dataset in ['SVHN']:
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

    if args.model in model_choices:
        model = Network().construct(args.model, config)
    else:
        raise Exception('Unknown model: {}'.format())

    for layer_name, _ in model.named_parameters():
        print('layer:', layer_name)
        epoch = 0
        layerwise_eigenspectrums = list()
        while True:
            layerwise_eigenspectrums_path = osp.join(ckpt_dir, 'training_eigenspectrums_epoch_{}_layer_{}.npz'.format(epoch, layer_name))
            
            if not osp.exists(layerwise_eigenspectrums_path):
                print(epoch, 'eigenspectrums were found')
                break
            eigenspectrum = np.load(layerwise_eigenspectrums_path)
            layerwise_eigenspectrums.append((epoch + 1, eigenspectrum['eigval'], eigenspectrum['eigval_density']))

            epoch = epoch + 1

        if len(layerwise_eigenspectrums) == 0:
            print('skipping layer', layer_name, 'because no npz was found for this layer')
            continue
        num_epochs = epoch

        full_eigenspectrums_path = osp.join(ckpt_dir, 'training_eigenspectrum_full.npy')
        eigenspectrums = np.load(full_eigenspectrums_path)

        full_eigenspectrums = list()

        for epoch in range(num_epochs):
            full_eigenspectrums.append((epoch + 1, eigenspectrums[2 * epoch], eigenspectrums[2 * epoch + 1]))
        
        linecolor1 = 'blue'
        linecolor2 = 'darkviolet'

        fig, axs = plt.subplots(epoch, 2, figsize=(2, epoch))

        for (i, (ax_left, ax_right)) in enumerate(axs):
            ax_left.semilogy(layerwise_eigenspectrums[i][1], layerwise_eigenspectrums[i][2], linewidth=linewidth, color=linecolor1)
            ax_left.set_title('layer epoch: {}'.format(layerwise_eigenspectrums[i][0]), fontsize=fontsize)
            ax_left.tick_params(axis='x', labelsize=labelsize)
            ax_left.tick_params(axis='y', labelsize=labelsize)

            ax_right.semilogy(full_eigenspectrums[i][1], full_eigenspectrums[i][2], linewidth=linewidth, color=linecolor2)
            ax_right.set_title('full  epoch: {}'.format(full_eigenspectrums[i][0]), fontsize=fontsize)
            ax_right.tick_params(axis='x', labelsize=labelsize)
            ax_right.tick_params(axis='y', labelsize=labelsize)

        fig.tight_layout()
        image_path = osp.join(images_dir, 'training_eigvals_epochwise_layer_{}.png'.format(layer_name))
        plt.savefig(image_path, dpi=args.dpi)

        print('image saved at:', image_path)
