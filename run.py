import os
import torch
import random
import numpy as np
import scipy.io
import yaml
import argparse
import importlib
from runners.Runner import Runner
import pandas as pd
from utils.data.TextData import TextData
from utils.data import file_utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',default='scE2TM')
    parser.add_argument('-c', '--config',default='/mnt/rao/home/chenhg/Methods/scE2TM/configs/model/scE2TM.yaml')
    parser.add_argument('-k', '--num_topic', type=int, default=100)
    parser.add_argument('-n', '--num_n_neighbors', type=int, default=15)
    parser.add_argument('-scdataset', '--scdataset_name', default='Wang')
    parser.add_argument('--num_top_gene', type=int, default=10)
    parser.add_argument('--test_index', type=int, default=1)
    args = parser.parse_args()
    return args

def print_topic_genes(beta, vocab, num_top_gene):
    topic_str_list = list()
    for i, topic_dist in enumerate(beta):
        topic_genes = np.array(vocab)[np.argsort(topic_dist)][:-(num_top_gene + 1):-1]
        topic_str = ' '.join(topic_genes)
        topic_str_list.append(topic_str)
        
        print('Topic {}: {}'.format(i + 1, topic_str))
    return topic_str_list

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def main():

    GLOBAL_SEED = 1
    set_seed(GLOBAL_SEED)  
    args = parse_args()

    # loading model configuration

    file_utils.update_args(args, path=args.config)

    output_prefix = f'output/{args.scdataset_name}/{args.model}_K{args.num_topic}_{args.test_index}th'
    file_utils.make_dir(os.path.dirname(output_prefix))

    seperate_line_log = '=' * 70                                                                                                                                                                                             #print(seperate_line_log)
    print(seperate_line_log)
    print('\n' + yaml.dump(vars(args), default_flow_style=False))

    dataset_handler = TextData(args.scdataset_name, args.batch_size, args.num_n_neighbors, args.scdataset_name)
    args.vocab_size = dataset_handler.train_data.shape[1]
    args.gene_embeddings = dataset_handler.gene_embeddings

    runner = Runner(args)
    beta = runner.train(dataset_handler.train_loader, dataset_handler.test_loader, dataset_handler.vocab, num_top_gene=args.num_top_gene, test_label = dataset_handler.test_labels)   

if __name__ == '__main__':
    main()
