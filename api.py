"""
High-level API for scE2TM.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path

# 内部导入
from .runners.Runner import Runner
from .utils.data.SingleCellDataHandler import SingleCellDataHandler
from .utils.data import file_utils


# ========== 辅助函数 ==========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_device(gpu_id):
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def to_numpy(x):
    """Convert torch Tensor to numpy array, detach if needed."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


# ========== 可 pickle 的参数容器 ==========
class _Args:
    pass


# ========== 高层训练函数 ==========
def scE2TM(
    dataset_name,
    data_dir='./data',
    output_dir='./output',
    num_topics=100,
    num_neighbors=15,
    num_top_genes=10,
    tac_weight=1.0,
    gpu_id=0,
    batch_size=512,
    learning_rate=0.001,
    epochs=500,
    beta_temp=0.2,
    en1_units=200,
    dropout=0.0,
    weight_loss_ECR=100.0,
    lr_scheduler=True,
    lr_step_size=125,
    seed=1,
    config_path=None,
    use_labels=False,
    **kwargs
):
    """
    Train an scE2TM model with a single function call.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'Wang').
    data_dir : str
        Directory containing data files.
    output_dir : str
        Directory to save outputs.
    num_topics : int
        Number of topics.
    num_neighbors : int
        Number of neighbors for graph construction.
    num_top_genes : int
        Number of top genes to extract per topic.
    tac_weight : float
        Weight for TAC loss.
    gpu_id : int
        GPU ID, -1 for CPU.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.
    epochs : int
        Number of training epochs.
    beta_temp : float
        Temperature for topic-gene distribution.
    en1_units : int
        Units in encoder hidden layer.
    dropout : float
        Dropout rate.
    weight_loss_ECR : float
        Weight for ECR loss.
    lr_scheduler : bool
        Whether to use learning rate scheduler.
    lr_step_size : int
        Step size for scheduler (if used).
    seed : int
        Random seed.
    config_path : str, optional
        Path to YAML config file. If provided, overrides the above arguments.
    use_labels : bool, optional
        If True, load cell type labels and compute evaluation metrics.
        Otherwise run in label-free mode (default).
    **kwargs
        Additional arguments passed to the model.

    Returns
    -------
    dict
        Contains:
            - topic_gene_matrix : (n_topics, n_genes)
            - cell_topic_matrix : (n_cells, n_topics)
            - topic_embeddings : (n_topics, embedding_dim)
            - gene_embeddings : (n_genes, embedding_dim)
            - model : trained model object
            - args : namespace of used arguments
    """
    # 1. 组装参数对象
    args = _Args()
    args.model = 'scE2TM'
    args.dataset_name = dataset_name
    args.data_dir = data_dir
    args.output_dir = output_dir
    args.num_topics = num_topics
    args.num_neighbors = num_neighbors
    args.num_top_genes = num_top_genes
    args.tac_weight = tac_weight
    args.gpu_id = gpu_id
    args.batch_size = batch_size
    args.learning_rate = learning_rate
    args.epochs = epochs
    args.beta_temp = beta_temp
    args.en1_units = en1_units
    args.dropout = dropout
    args.weight_loss_ECR = weight_loss_ECR
    args.lr_scheduler = lr_scheduler
    args.lr_step_size = lr_step_size
    args.use_labels = use_labels

    for k, v in kwargs.items():
        setattr(args, k, v)

    if config_path is not None and os.path.exists(config_path):
        file_utils.update_args(args, path=config_path)

    # 2. 随机种子和设备
    set_seed(seed)
    device = setup_device(gpu_id)
    args.device = device

    # 3. 数据加载
    data_handler = SingleCellDataHandler(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        n_neighbors=args.num_neighbors,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        use_labels=args.use_labels
    )

    args.vocab_size = data_handler.expression_data.shape[1]
    args.gene_embeddings = data_handler.gene_embeddings

    # 4. 训练
    runner = Runner(args)
    beta = runner.train(
        train_loader=data_handler.train_loader,
        test_loader=data_handler.test_loader,
        gene_vocab=data_handler.gene_vocab,
        num_top_genes=args.num_top_genes,
        cell_labels=data_handler.test_labels
    )

    # 5. 提取结果并转换为 numpy
    model = runner.model

    # 细胞-主题分布
    if runner.topic_distribution is not None:
        theta = to_numpy(runner.topic_distribution)
    else:
        theta = model.get_topic_distribution(data_handler.expression_data)
        theta = to_numpy(theta)

    # 嵌入
    topic_emb, gene_emb = model.get_embeddings()
    topic_emb = to_numpy(topic_emb)
    gene_emb = to_numpy(gene_emb)

    # topic-gene 矩阵
    beta = to_numpy(beta)

    # 6. 保存输出文件（确保目录存在）
    model_save_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(model_save_dir, exist_ok=True)
    import pandas as pd
    pd.DataFrame(beta).to_csv(os.path.join(model_save_dir, f'{args.dataset_name}_tg.csv'))
    pd.DataFrame(theta).to_csv(os.path.join(model_save_dir, f'{args.dataset_name}_topic_distribution.csv'))
    pd.DataFrame(topic_emb).to_csv(os.path.join(model_save_dir, f'{args.dataset_name}_topic_embedding.csv'))
    pd.DataFrame(gene_emb).to_csv(os.path.join(model_save_dir, f'{args.dataset_name}_gene_embedding.csv'))

    return {
        'topic_gene_matrix': beta,
        'cell_topic_matrix': theta,
        'topic_embeddings': topic_emb,
        'gene_embeddings': gene_emb,
        'model': model,
        'args': args,
    }