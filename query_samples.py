import argparse
import json
import os
import csv
import time
import math

import yaml
from yaml.loader import SafeLoader

import numpy as np
from utils.dataloaders import get_dataset, get_handler
from sklearn.manifold import TSNE
from torchvision import transforms
import torch
from query_strategies import *
from torch.utils.tensorboard import SummaryWriter
from models.densenet import DenseNetClassifier
from models.vision_transformer import VisionTransformerClassifier
from models.training import Training
import matplotlib


font_size = 25
font = {'family' : 'serif',
        'size'   : font_size}

matplotlib.rc('font', **font)


ALL_STRATEGIES = [
    'RandomSampling',
    'EntropySampling',
    'BALDDropout',
    'CoreSet',
    'AdversarialDeepFool',
    'BadgeSampling',
    'CDALSampling',
    'GCNSampling'
]


def setup_config(infer_args) -> dict:
    with open(f'data_config/{infer_args.data_name}.yaml') as f:
        cfg = yaml.load(f, Loader=SafeLoader)
    cfg['loader_tr_args'] = {'batch_size': infer_args.batch_size, 'num_workers': 2}
    cfg['loader_te_args'] = {'batch_size': infer_args.batch_size, 'num_workers': 2}
    
    if infer_args.data_augmentation:
        cfg['test_transform'] = transforms.Compose(
                    [
                     transforms.Resize((cfg['img_size'], cfg['img_size'])),
                     transforms.ToTensor(),
                     transforms.Normalize(*cfg['normalize']),
                 ])
    return cfg

def select_samples():
    infer_parser = argparse.ArgumentParser(description="Query samples parser for training hyper-parrameters at ech checkpoint.")
    infer_parser.add_argument('--model', type=str, default='vit_small')
    infer_parser.add_argument('--batch_size', type=int, default=4)
    infer_parser.add_argument('--emb_size', type=int, default=256)
    infer_parser.add_argument('--data_augmentation', action='store_const', default=True, const=True)
    
    infer_parser.add_argument('--model_path', type=str, default='./trained_model/vit_small.pth')
    infer_parser.add_argument('--strategy', type=str,
                        choices=['RandomSampling', 'EntropySampling',
                                 'BALDDropout', 'CoreSet',
                                 'AdversarialDeepFool',
                                 'BadgeSampling', 'CDALSampling', 'GCNSampling', 'AlphaMixSampling',
                                 'All'],
                        default='AlphaMixSampling')
    
    infer_parser.add_argument('--n_query', type=int, default=5)
    infer_parser.add_argument('--query_growth_ratio', type=int, default=1)
    infer_parser.add_argument('--data_name', type=str, choices=['SVHN', 'HMDB51'], default='HMDB51')
    
    # AlphaMix hyper-parameters
    infer_parser.add_argument('--alpha_cap', type=float, default=0.03125)
    infer_parser.add_argument('--alpha_opt', action="store_const", default=False, const=True)
    infer_parser.add_argument('--alpha_closed_form_approx', action="store_const", default=False, const=True)

    # Gradient descent Alpha optimisation
    infer_parser.add_argument('--alpha_learning_rate', type=float, default=0.1,
                        help='The learning rate of finding the optimised alpha')
    infer_parser.add_argument('--alpha_clf_coef', type=float, default=1.0)
    infer_parser.add_argument('--alpha_l2_coef', type=float, default=0.01)
    infer_parser.add_argument('--alpha_learning_iters', type=int, default=5,
                        help='The number of iterations for learning alpha')
    infer_parser.add_argument('--alpha_learn_batch_size', type=int, default=1000000)
    
    
    
    infer_args, _ = infer_parser.parse_known_args()
    
    infer_params = setup_config(infer_args)
    
    if infer_args.strategy == 'All':
        for strategy in ALL_STRATEGIES:
            al_infer(infer_args, infer_params, strategy)
    else:
        al_infer(infer_args, infer_params, infer_args.strategy)
        
def load_network(infer_params, structure_name, model_path):
    # load network
    if len(structure_name) >= 8 and structure_name == 'densenet':
        net = DenseNetClassifier
        net_args = {'arch_name': structure_name, 'n_label': infer_params['n_label'],
                    'pretrained': True,
                    'fine_tune_layers': 1,
                    'emb_size': infer_params['emb_size'],
                    'in_channels': infer_params['in_channels'],
                    'pretrained_weights': model_path}
    else:
        net = VisionTransformerClassifier
        net_args = {'arch_name': structure_name, 'n_label': infer_params['n_label'],
                    'pretrained': True,
                    'fine_tune_layers': 1,
                    'emb_size': infer_params['emb_size'],
                    'dropout': 0,
                    'patch_size': 16,
                    'n_last_blocks': 4,
                    'avgpool_patchtokens': False,
                    'pretrained_weights': model_path}
    return net, net_args

def al_infer(infer_args, infer_params, strategy_name):
    # load dataset
    img_names, X, Y = get_dataset(infer_args.data_name, infer_params['train_dir'], infer=True)
    sample_idx = np.zeros(len(img_names), dtype=bool)
    # Update infer parameters
    infer_params['emb_size'] = infer_args.emb_size  #256
    infer_params['dim'] = np.shape(X)[1:]
    # setup Writer
    writer = SummaryWriter(log_dir='logs/')
    result_file = open('logs/' + '.csv', 'w')
    result_writer = csv.writer(result_file, quoting=csv.QUOTE_ALL)
    result_writer.writerow(["Accuracy", "Duration"])
    
    net, net_args = load_network(infer_params, infer_args.model, infer_args.model_path)
    handler = get_handler(infer_args.data_name)
    
    use_cuda = torch.cuda.is_available()
    print('Using %s device.' % ("cuda" if use_cuda else "cpu"))
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = Training(net, net_args, handler, infer_params, writer, device, init_model=True)
    
    cls = globals()[strategy_name]
    strategy = cls(X, Y, sample_idx, None, None, model, infer_args, device, writer)
    start_time = time.time()

    # Query samples
    budget = infer_args.n_query
    print('QUERYING...')
    q_idxs, embeddings, preds, probs, u_idxs, candidate_idxs = strategy.query(budget)
    
    duration = time.time() - start_time

    # update query results
    sample_idx[q_idxs] = True
    
    
if __name__ == '__main__':
    select_samples()