import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import h5py
import scipy as sp
import scanpy as sc
import scipy.sparse
import scipy.io
from utils.data import file_utils
import pandas as pd
from scipy.sparse import csr_matrix
import models.utils as utils
import torch.nn as nn
import faiss
import copy
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

            
      
  #Create graph for mutual NN
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
  
  
  #Find number of connected components
  n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=graph, directed=True, return_labels=True, connection= 'strong')
  print(n_components)
  label_mapping = {i:[] for i in range(n_components)}

  for index, component in enumerate(labels):
    label_mapping[component].append(index)



  #Find the min spanning tree with KNN
  sorted_weights_tuples = min_spanning_tree(knn_indices, knn_dists, n_neighbors, n_neighbors)
  

  #Add edges until graph is connected
  for pos,(i,j,v) in enumerate(sorted_weights_tuples):

    if connectivity == "full_tree":
      connected_mnn[i].add(j)
      connected_mnn[j].add(i) 
      
      
    elif connectivity == "min_tree" and labels[i] != labels[j]:
      if len(label_mapping[labels[i]]) < len(label_mapping[labels[j]]):
        i, j = j, i
        
      connected_mnn[i].add(j)
      connected_mnn[j].add(i)
      j_pos = label_mapping[labels[j]]
      labels[j_pos] = labels[i]
      label_mapping[labels[i]].extend(j_pos)

  return connected_mnn  
def find_new_nn(knn_indices, knn_dists, knn_indices_pos, connected_mnn, n_neighbors_max, verbose=False):
  
  new_knn_dists= [] 
  new_knn_indices = []
  
  for i in range(len(knn_indices)): 
    #print(i)
    min_distances = []
    min_indices = []
    #Initialize vars
    heap = [(0,0,i)]
    mapping = {}
          
    seen = set()
    heapq.heapify(heap) 
    while(len(min_distances) < n_neighbors_max and len(heap) >0):
      dist, hop, nn = heapq.heappop(heap)
      if nn == -1:
        continue
      #For adjacent, only considering one hop away
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
      for j in range(n_neighbors_max-len(min_distances)):
        min_indices.append(-1)
        min_distances.append(np.inf)
    
    new_knn_dists.append(min_distances)
    new_knn_indices.append(min_indices)
    
    if verbose and i % int(len(knn_dists) / 10) == 0:
      print("\tcompleted ", i, " / ", len(knn_dists), "epochs")
  return new_knn_dists, new_knn_indices

#Calculate the connected mutual nn graph
def mutual_nn_nearest(knn_indices, knn_dists, n_neighbors, n_neighbors_max, connectivity="min_tree", verbose=False):
  mutual_nn = {}
  nearest_n= {}

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

  
  connected_mnn = create_connected_graph(mutual_nn, total_mutual_nn, knn_indices, knn_dists, n_neighbors, connectivity )
  new_knn_dists, new_knn_indices = find_new_nn(knn_indices, knn_dists, knn_indices_pos, connected_mnn, n_neighbors_max, verbose)

  
  return connected_mnn, mutual_nn, np.array(new_knn_indices), np.array(new_knn_dists)  

class MyDataSet(Dataset):
    """ 
    Preproces input matrix and labels.

    """
    def __init__(self, exp, data_exa):
        self.exp = exp
        self.data_exa = data_exa
        self.len = len(data_exa)
    def __getitem__(self,index):
        return self.exp[index],self.data_exa[index]
    def __len__(self):
        return self.len

class NeighborsDataset(Dataset):
    def __init__(self, dataset_text, dataset_image, indices_text, indices_image):
        super(NeighborsDataset, self).__init__()

        self.dataset_text = dataset_text
        self.dataset_image = dataset_image
        self.indices_text = indices_text
        self.indices_image = indices_image
        assert self.indices_text.shape[0] == len(self.indices_text)
        assert self.indices_image.shape[0] == len(self.indices_image)

    def __len__(self):
        return len(self.dataset_text)

    def __getitem__(self, index):
        anchor_text = self.dataset_text.__getitem__(index)
        anchor_image = self.dataset_image.__getitem__(index)

        flag = True
        while flag:
            temps = np.random.choice(self.indices_text[index], 1)[0]
            if temps!=-1:#temps!= index and temps!=-1:
                neighbor_index_text = temps
                flag = False
        neighbor_text = self.dataset_text.__getitem__(neighbor_index_text)

        flag = True
        while flag:
            temps = np.random.choice(self.indices_image[index], 1)[0]
            if temps!=-1:#temps!= index and temps!=-1:
                neighbor_index_image = temps
                flag = False
        #neighbor_index_image = np.random.choice(self.indices_image[index], 1)[0]
        neighbor_image = self.dataset_image.__getitem__(neighbor_index_image)

        return anchor_text, anchor_image, neighbor_text, neighbor_image

class TextData:
    def __init__(self, dataset, batch_size, n_neighbors, dataset_name):
        # train_data: NxV
        # test_data: Nxv
        # gene_emeddings: VxD
        # vocab: V, ordered by gene id.

        #dataset_path = f'../data/{dataset}'
        self.train_data, self.test_data, self.train_labels, self.test_labels, self.vocab, self.gene_embeddings, data_exa = self.load_data(dataset_name)
        self.vocab_size = len(self.vocab)

        n_neighbors = n_neighbors #!!!

        new_n_neighbors = n_neighbors
        print(n_neighbors)
        metric = 'euclidean'
        knn_indices, knn_dists, knn_search_index = nearest_neighbors(
            self.train_data,
            n_neighbors=n_neighbors,
            metric = metric,
            metric_kwds = {},
            angular=False,
            random_state = random_state,
            low_memory=True,
            use_pynndescent=True,
            n_jobs=4,
            verbose=True,
        )
        connected_mnn,mutual_nn, indices_text, new_knn_dists  = mutual_nn_nearest(knn_indices, knn_dists, n_neighbors, new_n_neighbors, connectivity= "min_tree" , verbose=True)


        knn_indices, knn_dists, knn_search_index = nearest_neighbors(
            data_exa.numpy(),
            n_neighbors=n_neighbors,
            metric = metric,
            metric_kwds = {},
            angular=False,
            random_state = random_state,
            low_memory=True,
            use_pynndescent=True,
            n_jobs=4,
            verbose=True,
        )
        connected_mnn,mutual_nn, indices_image, new_knn_dists  = mutual_nn_nearest(knn_indices, knn_dists, n_neighbors, new_n_neighbors, connectivity= "min_tree" , verbose=True)


        dataset = NeighborsDataset(
            self.train_data, data_exa, indices_text, indices_image
        )
        dataloader_train = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        dataloader_test= DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        print("===>train_size: ", self.train_data.shape[0])
        print("===>test_size: ", self.test_data.shape[0])
        print("===>vocab_size: ", self.vocab_size)
        print("===>average length: {:.3f}".format(self.train_data.sum(1).sum() / self.train_data.shape[0]))
        print("===>#label: ", len(np.unique(self.train_labels)))

        self.train_data = torch.from_numpy(self.train_data)
        self.test_data = torch.from_numpy(self.test_data)
        if torch.cuda.is_available():
            self.train_data = self.train_data.cuda()
            self.test_data = self.test_data.cuda()

        self.train_loader = dataloader_train
        self.test_loader = dataloader_test

    def load_data(self, data_name):

        name = data_name
        data_path = './data/'
        dataname = name+'_HIGHPRE_5000.csv'
        labelname = name+'_cell_anno.csv'

        data = pd.read_csv(data_path+dataname, sep=',', index_col=0)
        label = list(pd.read_csv(data_path+labelname,sep=',',index_col=0).iloc[:,0])   

        # name = 'lawlor'
        # data_path = '//mnt//'
        # dataname = name+'_HIGHPRE_5000.csv'
        # labelname = name+'_cell_anno.csv'
        # data = pd.read_csv(data_path+dataname, sep=',', index_col=0)
        # label = list(pd.read_csv(data_path+labelname,sep=',',index_col=0).iloc[:,0])   

        vocab = list(data.columns)
        gene_embeddings = nn.init.trunc_normal_(torch.zeros(len(vocab), 200), std = 0.02).numpy()

        label_np = np.zeros(len(label))
        dicts_label_index = {}
        label_index = 0
        for index, value in enumerate(label):
            if value not in dicts_label_index:
                label_index += 1
                dicts_label_index[value] = label_index

        for index, value in enumerate(label):
            label_np[index] =  dicts_label_index[value]

        train_data = data.values.astype(np.float32)
        train_labels = label_np
        test_data = data.values.astype(np.float32)
        test_labels = label_np

        data_exa = torch.from_numpy(pd.read_csv(data_path+name+'.csv', sep=',', index_col=0).values)
       # data_exa = torch.from_numpy(pd.read_csv('/mnt/'+name+'.csv', sep=',', index_col=0).values)
        return train_data, test_data, train_labels, test_labels, vocab, gene_embeddings, data_exa
