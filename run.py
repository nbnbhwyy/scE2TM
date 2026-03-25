import random
import torch
import numpy as np
import yaml
import argparse
import os
from pathlib import Path
from runners.Runner import Runner
from utils.data.SingleCellDataHandler import SingleCellDataHandler
from utils.data import file_utils

SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)
print(f"Working directory set to: {os.getcwd()}")

def parse_args():
    parser = argparse.ArgumentParser(description='scE2TM: Single-cell Embedded Topic Model')
    parser.add_argument('-m', '--model', default='scE2TM', help='Model name')
    parser.add_argument('-c', '--config', default='./configs/scE2TM.yaml', 
                        help='Path to config file (use relative path)')
    parser.add_argument('-k', '--num_topics', type=int, default=100, 
                        help='Number of topics')
    parser.add_argument('-n', '--num_neighbors', type=int, default=15,
                        help='Number of neighbors for graph construction')
    parser.add_argument('-d', '--dataset_name', default='Wang',
                        help='Single-cell dataset name')
    parser.add_argument('--num_top_genes', type=int, default=10,
                        help='Number of top genes per topic')
    parser.add_argument('--tac_weight', type=float, default=1.0,
                        help='Weight for TAC loss')  
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for data files')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory for output files')
    args = parser.parse_args()
    return args
def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_device(gpu_id):
    """
    Setup computation device
    
    Args:
        gpu_id: GPU ID, -1 for CPU
    Returns:
        device: torch.device object
    """
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def main():
    GLOBAL_SEED = 1
    set_seed(GLOBAL_SEED)  
    args = parse_args()

    # loading model configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    file_utils.update_args(args, path=args.config)

    # Setup device
    device = setup_device(args.gpu_id)
    args.device = device

    seperate_line_log = '=' * 70                                                                                                                                                                                             #print(seperate_line_log)
    print(seperate_line_log)
    print('\n' + yaml.dump(vars(args), default_flow_style=False))

    data_handler = SingleCellDataHandler(
        dataset_name=args.dataset_name,  
        batch_size=args.batch_size,    
        n_neighbors=args.num_neighbors,   
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device  
    )
        
    args.vocab_size = data_handler.expression_data.shape[1]
    args.gene_embeddings = data_handler.gene_embeddings

    runner = Runner(args)
    runner.train(
        train_loader=data_handler.train_loader, 
        test_loader=data_handler.test_loader, 
        gene_vocab=data_handler.gene_vocab, 
        num_top_genes=args.num_top_genes, 
        cell_labels=data_handler.test_labels
    )   
    
if __name__ == '__main__':
    main()
