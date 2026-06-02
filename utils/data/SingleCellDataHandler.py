import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
import torch.nn as nn
import copy
import os
from umap.umap_ import nearest_neighbors
import heapq
from sklearn.utils import check_random_state

random_state = check_random_state(1)

def min_spanning_tree(knn_indices, knn_dists, n_neighbors, threshold):
    rows = np.zeros(knn_indices.shape[0] * n_neighbors, dtype=np.int32)
    cols = np.zeros(knn_indices.shape[0] * n_neighbors, dtype=np.int32)
    vals = np.zeros(knn_indices.shape[0] * n_neighbors, dtype=np.float32)
    
    pos = 0
    for i, indices in enumerate(knn_indices):
        for j, index in enumerate(indices[:threshold]):
            if index == -1:
                continue
            rows[pos] = i 
            cols[pos] = index
            vals[pos] = knn_dists[i][j]
            pos += 1
    
    matrix = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(knn_indices.shape[0], knn_indices.shape[0]))
    Tcsr = scipy.sparse.csgraph.minimum_spanning_tree(matrix)
    
    Tcsr = scipy.sparse.coo_matrix(Tcsr)
    weights_tuples = zip(Tcsr.row, Tcsr.col, Tcsr.data)
    
    sorted_weights_tuples = sorted(weights_tuples, key=lambda tup: tup[2])
    
    return sorted_weights_tuples 

def create_connected_graph(mutual_nn, total_mutual_nn, knn_indices, knn_dists, n_neighbors, connectivity):
    connected_mnn = copy.deepcopy(mutual_nn)
    
    if connectivity == "nearest":
        for i in range(len(knn_indices)): 
            if len(mutual_nn[i]) == 0:
                first_nn = knn_indices[i][1]
                if first_nn != -1:
                    connected_mnn[i].add(first_nn) 
                    connected_mnn[first_nn].add(i) 
                    total_mutual_nn += 1
        return connected_mnn

    # Create graph for mutual NN
    rows = np.zeros(total_mutual_nn, dtype=np.int32)
    cols = np.zeros(total_mutual_nn, dtype=np.int32)
    vals = np.zeros(total_mutual_nn, dtype=np.float32)
    pos = 0
    for i in connected_mnn:
        for j in connected_mnn[i]:
            rows[pos] = i 
            cols[pos] = j
            vals[pos] = 1
            pos += 1
    graph = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(knn_indices.shape[0], knn_indices.shape[0]))
    
    # Find number of connected components
    n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=graph, directed=True, return_labels=True, connection='strong')
    print(f"Number of connected components: {n_components}")
    label_mapping = {i: [] for i in range(n_components)}

    for index, component in enumerate(labels):
        label_mapping[component].append(index)

    # Find the min spanning tree with KNN
    sorted_weights_tuples = min_spanning_tree(knn_indices, knn_dists, n_neighbors, n_neighbors)
    
    # Add edges until graph is connected
    for pos, (i, j, v) in enumerate(sorted_weights_tuples):
        if connectivity == "full_tree":
            connected_mnn[i].add(j)
            connected_mnn[j].add(i) 
            
        elif connectivity == "min_tree" and labels[i] != labels[j]:
            if len(label_mapping[labels[i]]) < len(label_mapping[labels[j]]):
                i, j = j, i
                
            connected_mnn[i].add(j)
            connected_mnn[j].add(i)
            j_pos = label_mapping[labels[j]]
            for idx in j_pos:
                labels[idx] = labels[i]
            label_mapping[labels[i]].extend(j_pos)

    return connected_mnn  

def find_new_nn(knn_indices, knn_dists, knn_indices_pos, connected_mnn, n_neighbors_max, verbose=False):
    new_knn_dists = [] 
    new_knn_indices = []
    
    for i in range(len(knn_indices)): 
        min_distances = []
        min_indices = []
        # Initialize vars
        heap = [(0, 0, i)]
        mapping = {}
        
        seen = set()
        heapq.heapify(heap) 
        while len(min_distances) < n_neighbors_max and len(heap) > 0:
            dist, hop, nn = heapq.heappop(heap)
            if nn == -1:
                continue
            # For adjacent, only considering one hop away
            if nn not in seen and hop <= 1:
                min_distances.append(dist)
                min_indices.append(nn)
                seen.add(nn)
                neighbor = connected_mnn[nn]
                
                for nn_nn in neighbor:
                    if nn_nn not in seen and hop <= 0:
                        distance = 0
                        if nn_nn in knn_indices_pos[nn]:
                            pos = knn_indices_pos[nn][nn_nn]
                            distance = knn_dists[nn][pos] 
                        else:
                            pos = knn_indices_pos[nn_nn][nn]
                            distance = knn_dists[nn_nn][pos] 
                        distance += dist
                        
                        if nn_nn not in mapping:
                            mapping[nn_nn] = distance
                            heapq.heappush(heap, (distance, hop+1, nn_nn))
                        elif mapping[nn_nn] > distance:
                            mapping[nn_nn] = distance
                            heapq.heappush(heap, (distance, hop+1, nn_nn))
        
        if len(min_distances) < n_neighbors_max:
            for j in range(n_neighbors_max - len(min_distances)):
                min_indices.append(-1)
                min_distances.append(np.inf)
        
        new_knn_dists.append(min_distances)
        new_knn_indices.append(min_indices)
        
        if verbose and i % int(len(knn_dists) / 10) == 0:
            print("\tcompleted ", i, " / ", len(knn_dists), "epochs")
    return new_knn_dists, new_knn_indices

# Calculate the connected mutual nn graph
def mutual_nn_nearest(knn_indices, knn_dists, n_neighbors, n_neighbors_max, connectivity="min_tree", verbose=False):
    mutual_nn = {}
    nearest_n = {}

    knn_indices_pos = [None] * len(knn_indices)

    total = 0
    
    for i, top_vals in enumerate(knn_indices):
        nearest_n[i] = set(top_vals)
        knn_indices_pos[i] = {}
        for pos, nn in enumerate(top_vals):
            knn_indices_pos[i][nn] = pos
    
    total_mutual_nn = 0
    for i, top_vals in enumerate(knn_indices):
        mutual_nn[i] = set()
        for ind, nn in enumerate(top_vals):
            if nn != -1 and (i in nearest_n[nn] and i != nn):
                mutual_nn[i].add(nn)
                total_mutual_nn += 1

    connected_mnn = create_connected_graph(mutual_nn, total_mutual_nn, knn_indices, knn_dists, n_neighbors, connectivity)
    new_knn_dists, new_knn_indices = find_new_nn(knn_indices, knn_dists, knn_indices_pos, connected_mnn, n_neighbors_max, verbose)
    
    return connected_mnn, mutual_nn, np.array(new_knn_indices), np.array(new_knn_dists)  

class SingleCellDataset(Dataset):
    """ 
    Preprocess single-cell expression matrix and foundation embeddings.
    """
    def __init__(self, expression_data, foundation_embeddings):
        self.expression_data = expression_data 
        self.foundation_embeddings = foundation_embeddings 
        self.len = len(foundation_embeddings)
        
    def __getitem__(self, index):
        return self.expression_data[index], self.foundation_embeddings[index] 
    
    def __len__(self):
        return self.len

class ExpressionEmbeddingNeighborDataset(Dataset):
    def __init__(self, expression_data, embedding_data, expression_indices, embedding_indices):
        super(ExpressionEmbeddingNeighborDataset, self).__init__()

        self.expression_data = expression_data
        self.embedding_data = embedding_data
        self.expression_indices = expression_indices
        self.embedding_indices = embedding_indices
        assert self.expression_indices.shape[0] == len(self.expression_indices)
        assert self.embedding_indices.shape[0] == len(self.embedding_indices)

    def __len__(self):
        return len(self.expression_data)

    def __getitem__(self, index):
        anchor_expression = self.expression_data.__getitem__(index)
        anchor_embedding = self.embedding_data.__getitem__(index)

        flag = True
        while flag:
            temp_idx = np.random.choice(self.expression_indices[index], 1)[0]
            if temp_idx != -1:
                neighbor_expression_idx = temp_idx
                flag = False
        neighbor_expression = self.expression_data.__getitem__(neighbor_expression_idx)

        flag = True
        while flag:
            temp_idx = np.random.choice(self.embedding_indices[index], 1)[0] 
            if temp_idx != -1:
                neighbor_embedding_idx = temp_idx
                flag = False
        neighbor_embedding = self.embedding_data.__getitem__(neighbor_embedding_idx)

        return anchor_expression, anchor_embedding, neighbor_expression, neighbor_embedding

class SingleCellDataHandler:
    def __init__(self, dataset_name, batch_size, n_neighbors, 
                data_dir='./data', output_dir='./output', device=None, use_labels=False):
        """
        Args:
            dataset_name: dataset name
            batch_size: batch size
            n_neighbors: number of neighbors
            data_dir: data directory
            output_dir: output directory
            device: computation device
        """
        self.data_dir = data_dir
        self.use_labels = use_labels
        self.output_dir = output_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.expression_data, self.test_expression_data, self.train_labels, self.test_labels, self.gene_vocab, self.gene_embeddings, self.foundation_embeddings = self.load_data(dataset_name)
        self.vocab_size = len(self.gene_vocab)

        print(f"Number of neighbors: {n_neighbors}")
        metric = 'euclidean'
        
        # Compute nearest neighbors for expression data
        knn_indices, knn_dists, knn_search_index = nearest_neighbors(
            self.expression_data,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds={},
            angular=False,
            random_state=random_state,
            low_memory=True,
            use_pynndescent=True,
            n_jobs=4,
            verbose=True,
        )
        connected_mnn, mutual_nn, expression_indices, new_knn_dists = mutual_nn_nearest(  
            knn_indices, knn_dists, n_neighbors, n_neighbors, connectivity="min_tree", verbose=True
        )

        # Compute nearest neighbors for foundation embeddings
        knn_indices, knn_dists, knn_search_index = nearest_neighbors(
            self.foundation_embeddings.numpy(),
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds={},
            angular=False,
            random_state=random_state,
            low_memory=True,
            use_pynndescent=True,
            n_jobs=4,
            verbose=True,
        )
        connected_mnn, mutual_nn, embedding_indices, new_knn_dists = mutual_nn_nearest(  
            knn_indices, knn_dists, n_neighbors, n_neighbors, connectivity="min_tree", verbose=True
        )

        dataset = ExpressionEmbeddingNeighborDataset(
            self.expression_data, self.foundation_embeddings, expression_indices, embedding_indices
        )
        
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        test_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        
        print("===>train_size: ", self.expression_data.shape[0])
        print("===>test_size: ", self.test_expression_data.shape[0])
        print("===>vocab_size: ", self.vocab_size)
        print("===>average expression: {:.3f}".format(self.expression_data.sum(1).sum() / self.expression_data.shape[0]))
        print("===>#labels: ", len(np.unique(self.train_labels)))

        self.expression_data = torch.from_numpy(self.expression_data).to(self.device)
        self.test_expression_data = torch.from_numpy(self.test_expression_data).to(self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # 在 __init__ 末尾添加
        if self.train_labels is None:
            print("Running in label-free mode. No evaluation will be performed during training.")
        else:
            print(f"===>#labels: {len(np.unique(self.train_labels))}")

    def load_data(self, data_name):
        name = data_name
        data_path = self.data_dir + '/'
        
        expression_file = name + '_HIGHPRE.csv'
        label_file = name + '_cell_anno.csv'
        foundation_embedding_file = name + '.csv'
        
        # 加载表达数据（必须存在）
        expression_df = pd.read_csv(data_path + expression_file, sep=',', index_col=0)
        
        # ---- 标签加载：仅在启用评估模式时执行 ----
        if self.use_labels:
            label_path = data_path + label_file
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file {label_path} not found but eval mode is enabled.")
            label_df = pd.read_csv(label_path, sep=',', index_col=0)
            labels = list(label_df.iloc[:, 0])
            # 转换为数值标签（原有代码）
            label_np = np.zeros(len(labels))
            label_to_idx = {}
            label_idx = 0
            for idx, label_value in enumerate(labels):
                if label_value not in label_to_idx:
                    label_idx += 1
                    label_to_idx[label_value] = label_idx
            for idx, label_value in enumerate(labels):
                label_np[idx] = label_to_idx[label_value]
            train_labels = label_np
            test_labels = label_np
        else:
            # 默认无标签模式
            train_labels = None
            test_labels = None
            print("Running in label-free mode (use --eval to load labels).")
        
        # 后续基因词汇表、嵌入等不变
        gene_vocab = list(expression_df.columns)
        gene_embeddings = nn.init.trunc_normal_(torch.zeros(len(gene_vocab), 200), std=0.02).numpy()
        expression_data = expression_df.values.astype(np.float32)
        test_expression_data = expression_data.copy()
        foundation_embedding_path = data_path + foundation_embedding_file
        foundation_embeddings = torch.from_numpy(pd.read_csv(foundation_embedding_path, sep=',', index_col=0).values)
        
        return (expression_data, test_expression_data,
                train_labels, test_labels,
                gene_vocab, gene_embeddings, foundation_embeddings)