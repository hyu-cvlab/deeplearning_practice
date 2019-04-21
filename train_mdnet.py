import sys
import pickle
import yaml
import time
import numpy as np
import os
import os.path as osp
import pandas
import torch

from utils import Options, overlap_ratio
from models.mdnet import MDNet, set_optimizer, BCELoss, Precision
from models.extractor import SampleGenerator, RegionDataset

opts = yaml.safe_load(open('options.yaml','r'))

def load_database(path='./datasets'):
    def gen_config(seq_name):
        seq_home = path
        img_dir = os.path.join(seq_home, seq_name, 'img')
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir, x) for x in img_list]
        
        gt = []
        with open(gt_path) as f:
            for line in f.readlines():
                if '\t' in line:
                    gt.append(list(map(int,line.strip().split('\t'))))
                elif ',' in line:
                    gt.append(list(map(int,line.strip().split(','))))
                elif ' ' in line:
                    gt.append(list(map(int,line.strip().split(' '))))
        gt = np.array(gt)
        # with open(gt_path) as f:
        #     print(f.readlines())
        #     gt = np.loadtxt((x.replace('\t',',') for x in f), delimiter=',')
        return img_list, gt
    f = open(osp.join(path,'tb_100.txt'))
    seq_list = f.readlines()
    names = sorted([x.split('\t')[0].strip() for x in seq_list])
    data = {}
    for name in names:
        img_list, gt = gen_config(name)
        if len(img_list) != gt.shape[0]:
            continue
        seq = {}
        seq['images'] = img_list
        seq['gt'] = gt
        data[name] = seq
    return data

def train_mdnet():

    # Init dataset
    data = load_database()
    K = len(data)
    dataset = [None] * K
    for k, seq in enumerate(data.values()):
        dataset[k] = RegionDataset(seq['images'], seq['gt'], opts)

    # Init model
    model = MDNet(opts['init_model_path'], K)
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer
    criterion = BCELoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'], opts['lr_mult'])

    # Main trainig loop
    for i in range(opts['n_cycles']):
        print('==== Start Cycle {:d}/{:d} ===='.format(i + 1, opts['n_cycles']))

        if i in opts.get('lr_decay', []):
            print('decay learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opts.get('gamma', 0.1)

        # Training
        model.train()
        prec = np.zeros(K)
        k_list = np.random.permutation(K)
        for j, k in enumerate(k_list):
            tic = time.time()
            # training
            pos_regions, neg_regions = dataset[k].next()
            if opts['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()
            pos_score = model(pos_regions)
            neg_score = model(neg_regions)

            loss = criterion(pos_score, neg_score)
            model.zero_grad()
            loss.backward()
            if 'grad_clip' in opts:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
            optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time()-tic
            print('Iter {:2d} (Domain {:2d}), Loss {:.3f}, Precision {:.3f}, Time {:.3f}'
                    .format(j, k, loss.item(), prec[k], toc))

        print('Mean Precision: {:.3f}'.format(prec.mean()))
        print('Save model to {:s}'.format(opts['model_path']))
        if opts['use_gpu']:
            model = model.cpu()
        states = {'shared_layers': model.layers.state_dict()}
        torch.save(states, opts['model_path'])
        if opts['use_gpu']:
            model = model.cuda()


if __name__ == "__main__":
    train_mdnet()
