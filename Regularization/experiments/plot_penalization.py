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
    
    default_k = 1

    parser = argparse.ArgumentParser(description=file_purpose,
        epilog=file_purpose,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-r', '--run', type=str, required=True, help='run dir')
    # parser.add_argument('-k', type=int, default=default_k, help='last k stats')
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
    log_dir = osp.join(args.run, 'logs')
    images_dir = osp.join(args.run, 'images')
    
    no_penalization_path = osp.join(ckpt_dir, 'no_penalization_train_stats.pkl')
    ht_penalization_path = osp.join(ckpt_dir, 'ht_penalization_train_stats.pkl')
    l2_penalization_path = osp.join(ckpt_dir, 'l2_penalization_train_stats.pkl')
    combined_penalization_path = osp.join(ckpt_dir, 'combined_penalization_train_stats.pkl')

    for path in [args.run, ckpt_dir, images_dir]:
        assert osp.exists(path), '{} was not found'.format(path)

    paths_found = list()
    for path in [no_penalization_path, ht_penalization_path, l2_penalization_path, combined_penalization_path]:
        if osp.exists(path):
            paths_found.append(path)

    assert len(paths_found) == 1, 'number of paths found: {}, [{}]'.format(len(paths_found), paths_found)
    
    with open(paths_found[0], 'rb') as f:
        print('path found:', paths_found)
        stats = pkl.load(f)

    # train_stats = np.array([stat['train'] for stat in stats])
    test_stats = np.array([stat['test'] for stat in stats])
    train_stats = np.array([stat['train'] for stat in stats])

    log_file_paths = list()
    for p in [osp.join(log_dir, '{}_penalization.log'.format(prefix)) for prefix in ['no', 'ht', 'l2', 'combined']]:
        if osp.exists(p):
            log_file_paths.append(p)
    assert len(log_file_paths) <= 1, 'log_file_paths has more than one element: {}'.format(len(log_file_paths))
    if len(log_file_paths) == 0:
        print('No log file found...')
    elif len(log_file_paths) == 1:
        log_file_path = log_file_paths[0]
        with open(log_file_path) as f:
            print('log file found, arguments:')
            print(f.readline())
            assert len(f.readlines()) > 50

    # assert train_stats.shape == test_stats.shape, 'shapes dont match: {} != {}'.format(train_stats.shape, test_stats.shape)
    # print('# of train sequences:', train_stats.shape[0])
    print('# of test sequences:', test_stats.shape)

    t1 = test_stats[:, -1, 0] 
    t2 = test_stats[:, -1, 1] 

    print('acc@1: {:.3f} +- {:.3f}'.format(t1.mean(), t1.std()))
    print('acc@5: {:.3f} +- {:.3f}'.format(t2.mean(), t2.std()))
