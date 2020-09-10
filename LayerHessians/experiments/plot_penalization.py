#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pickle as pkl
import seaborn as sns
import os.path as osp
import matplotlib.pyplot as plt


def parse_args():

    file_purpose = '''
    plot script for ht_penalization.py, no_penalization.py and l2_penalization.py
    '''
    
    default_k = 10

    parser = argparse.ArgumentParser(description=file_purpose,
        epilog=file_purpose,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-r', '--run', type=str, required=True, help='run dir')
    parser.add_argument('-k', type=int, default=default_k, help='last k stats')
    parser.add_argument('-pdb', action='store_true', help='run with pdb')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    
    # debugging utils
    if args.pdb:
        import pdb
        pdb.set_trace()

    # directory structure
    ckpt_dir = osp.join(args.run, 'ckpt')
    images_dir = osp.join(args.run, 'images')
    
    no_penalization_path = osp.join(ckpt_dir, 'no_penalization_train_stats.pkl')
    ht_penalization_path = osp.join(ckpt_dir, 'ht_penalization_train_stats.pkl')
    l2_penalization_path = osp.join(ckpt_dir, 'l2_penalization_train_stats.pkl')

    for path in [args.run, ckpt_dir, images_dir]:
        assert osp.exists(path), '{} was not found'.format(path)

    paths_found = list()
    for path in [no_penalization_path, ht_penalization_path, l2_penalization_path]:
        if osp.exists(path):
            paths_found.append(path)

    assert len(paths_found) == 1, 'number of paths found: {}, [{}]'.format(len(paths_found), paths_found)
    
    with open(paths_found[0], 'rb') as f:
        print('path found:', paths_found)
        stats = pkl.load(f)

    train_stats = np.array([stat['train'] for stat in stats])
    test_stats = np.array([stat['test'] for stat in stats])

    assert train_stats.shape == test_stats.shape, 'shapes dont match: {} != {}'.format(train_stats.shape, test_stats.shape)
    print('# of train sequences:', train_stats.shape[0])
    print('# of test sequences:', test_stats.shape[0])

    k_train_stats = train_stats[:, -args.k:]
    k_test_stats = test_stats[:, -args.k:]

    #for k_train_stat in k_train_stats:
    #    assert k_train_stat[-1] > 0.6, 'bad train stats'
    #for k_test_stat in k_test_stats:
    #    assert k_test_stat[-1] > 0.6, 'bad test stats'

    print('train acc:', np.mean(k_train_stats), '+-', np.std(k_train_stats))
    print('test acc:', np.mean(k_test_stats), '+-', np.std(k_test_stats))

    print('gene. err.:', np.mean(k_train_stats - k_test_stats), '+-', np.std(k_train_stats - k_test_stats))

    sns.set_style('darkgrid')

    for i in range(train_stats.shape[0]):
        plt.plot(train_stats[i], label='train:%d' % i)
        plt.plot(test_stats[i], label='test:%d' % i)

    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.legend()
    plt.tight_layout()

    image_path = osp.join(images_dir, 'acc.png')
    plt.savefig(image_path, dpi=200)

    print('image saved at:', image_path)
