import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from models.scE2TM import scE2TM
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import scanpy as sc
import pandas as pd
import gseapy as gp
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
import faiss
import time
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import f1_score,precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F

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
        neighbor_index_text = np.random.choice(self.indices_text[index], 1)[0]
        neighbor_text = self.dataset_text.__getitem__(neighbor_index_text)
        neighbor_index_image = np.random.choice(self.indices_image[index], 1)[0]
        neighbor_image = self.dataset_image.__getitem__(neighbor_index_image)

        return anchor_text, anchor_image, neighbor_text, neighbor_image

class NeighborsDataset(Dataset):
    def __init__(self, dataset_text, dataset_image, indices_text, indices_image, z):
        super(NeighborsDataset, self).__init__()

        self.dataset_text = dataset_text
        self.dataset_image = dataset_image
        self.indices_text = indices_text
        self.indices_image = indices_image
        self.z = z
        assert self.indices_text.shape[0] == len(self.indices_text)
        assert self.indices_image.shape[0] == len(self.indices_image)

    def __len__(self):
        return len(self.dataset_text)

    def __getitem__(self, index):
        anchor_text = self.dataset_text.__getitem__(index)
        anchor_image = self.dataset_image.__getitem__(index)
        z = self.z.__getitem__(index)
        neighbor_index_text = np.random.choice(self.indices_text[index], 1)[0]
        neighbor_text = self.z.__getitem__(neighbor_index_text)
        neighbor_index_image = np.random.choice(self.indices_image[index], 1)[0]
        neighbor_image = self.dataset_image.__getitem__(neighbor_index_image)

        return anchor_text, anchor_image, neighbor_text, neighbor_image, z


def mine_nearest_neighbors(features, topk=50):
    print("Computing nearest neighbors...")
    features = features.astype(np.float32)
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk + 1)  # Sample itself is included
    print("Nearest neighbors computed.")
    return indices[:, 1:]

def ext_topic_genes(beta, vocab, num_top_gene):
    topic_str_list = list()
    for i, topic_dist in enumerate(beta):
        topic_genes = np.array(vocab)[np.argsort(topic_dist)][:-(num_top_gene + 1):-1]
        topic_str = ' '.join(topic_genes)
        topic_str_list.append(topic_str)
    return topic_str_list

def compute_coherence(doc_gene, topic_gene, N, dicts_gene_tran):
    # print('computing coherence ...')    
    topic_size, gene_size = np.shape(topic_gene)
    doc_size = np.shape(doc_gene)[0]
    # find top genes'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_gene_idx = np.argpartition(topic_gene[topic_idx, :], -N)[-N:]
        topic_list.append(top_gene_idx)
    #print(topic_list)
    # compute coherence of each topic
    sum_coherence_score = 0.0
    for i in range(topic_size):
        gene_array = topic_list[i]
        sum_score = 0.0
        for n in range(N):
            if gene_array[n] in dicts_gene_tran:
                flag_n = doc_gene[:, dicts_gene_tran[gene_array[n]]] > 0
                p_n = np.sum(flag_n) / doc_size
                for l in range(n + 1, N):
                    if gene_array[l] in dicts_gene_tran:
                        flag_l = doc_gene[:, dicts_gene_tran[gene_array[l]]] > 0
                        p_l = np.sum(flag_l)
                        p_nl = np.sum(flag_n * flag_l)
                        if p_n * p_l * p_nl > 0:
                            p_l = p_l / doc_size
                            p_nl = p_nl / doc_size
                            sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
        sum_coherence_score += sum_score * (2 / (N * N - N))
    sum_coherence_score = sum_coherence_score / topic_size
    return sum_coherence_score


def TD_eva(texts):
    K = len(texts)
    T = len(texts[0].split())
    vectorizer = CountVectorizer()
    counter = vectorizer.fit_transform(texts).toarray()

    TF = counter.sum(axis=0)
    TD = (TF == 1).sum() / (K * T)

    return TD

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def f1(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

    return (precision_macro, recall_macro, f1_macro), (precision_micro, recall_micro, f1_micro)

def classification_metric(data, labels):

    X_train, X_test, y_train, y_test = train_test_split(data , labels, test_size=0.3)
    clf = LogisticRegression().fit(X_train, y_train) 
    y_pred = clf.predict(np.array(X_test))
    (precision_macro, recall_macro, f1_macro), (precision_micro, recall_micro, f1_micro)= np.round(f1(np.array(y_test), np.array(y_pred)), 5)
    print('F1 score: f1_macro = {}, f1_micro = {}'.format(f1_macro, f1_micro))
    print('precision score: precision_macro = {}, precision_micro = {}'.format(precision_macro, precision_micro))
    print('recall score: recall_macro = {}, recall_micro = {}'.format(recall_macro, recall_micro))

class Runner:
    def __init__(self, args):
        self.args = args
        self.model = ECRTM(args)
        self.theta = None
        self.dataloder = None
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.args.learning_rate,
        }

        optimizer = torch.optim.RMSprop(**args_dict)  #!!!
        return optimizer


    def evaluate(self, test_data, vocab, num_top_gene, test_label):
        with torch.no_grad():
            self.model.eval()
            beta = self.model.get_beta()
            topic_embedding, gene_embedding = self.model.get_embedding()
            beta = beta.detach().cpu().numpy()
            theta = self.test(test_data)

            topic_str_list = ext_topic_genes(beta, vocab, num_top_gene)
            TD = TD_eva(topic_str_list)
            print(f"===>TD_T{num_top_gene}: {TD:.5f}")
    
            kegg = gp.read_gmt(path="/mnt/rao/home/chenhg/Methods/ECRTM-Singlecell/data/c5.all.v2023.2.Hs.symbols.gmt")
            gene_set = []
            for value in kegg.values():
                gene_set.extend(value)
            gene_set = list(set(gene_set))
            dicts_gene_index = {}
            for index, value in enumerate(gene_set):
                dicts_gene_index[value] = index
            bg_data = np.zeros((len(kegg),len(gene_set)))
            for index, values in enumerate(kegg.values()):
                for value in values:
                    bg_data[index][dicts_gene_index[value]] = 1
  
            dicts_gene_tran = {}
            for index, value in enumerate(vocab):
                if value in dicts_gene_index:
                    dicts_gene_tran[index] = dicts_gene_index[value]

            TC = compute_coherence(bg_data, beta, num_top_gene, dicts_gene_tran)
            print(f"===>TC_T{num_top_gene}: {TC:.5f}")
            # pd.DataFrame(beta).to_csv('./output/'+self.args.scdataset_name+'/'+self.args.scdataset_name+'_tg'+'.csv') 
            # pd.DataFrame(theta).to_csv('./output/'+self.args.scdataset_name+'/'+self.args.scdataset_name+'_embedding'+'.csv') 
            # pd.DataFrame(topic_embedding.detach().cpu().numpy()).to_csv('./output/'+self.args.scdataset_name+'/'+self.args.scdataset_name+'_topic_embedding'+'.csv') 
            # pd.DataFrame(gene_embedding.detach().cpu().numpy()).to_csv('./output/'+self.args.scdataset_name+'/'+self.args.scdataset_name+'_gene_embedding'+'.csv') 
            adata = sc.AnnData(theta)
            adata.obs['cell_type'] = test_label
            sc.pp.pca(adata)
            sc.pp.neighbors(adata,use_rep = 'X')
            maxn = 2
            minn= 0
            list_value = []
            for x in  range(minn, maxn*10):
                sc.tl.louvain(adata,resolution=x/10.0,random_state=0)
                list_value.append(adjusted_rand_score(adata.obs['cell_type'],adata.obs['louvain']))
            sc.tl.louvain(adata,resolution=list_value.index(max(list_value))*0.1)
            print("d-scIGM   Adjusted_rand_score   "+str(adjusted_rand_score(adata.obs['cell_type'],adata.obs['louvain']))+"   Adjusted_mutual_info_score   "+str(adjusted_mutual_info_score(adata.obs['cell_type'],adata.obs['louvain']))+"   ASW   "+str(silhouette_score(adata.X,adata.obs['cell_type'])))
            from sklearn import metrics
            def purity_score(y_true, y_pred):
                # compute contingency matrix (also called confusion matrix)
                contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
                # return purity
                return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
            print(purity_score(adata.obs['cell_type'],adata.obs['louvain']))
            print("DB   "+str(davies_bouldin_score(adata.X,adata.obs['cell_type'])))  
            
            classification_metric(theta, test_label)
            print("d-scIGM   Adjusted_rand_score   "+str(adjusted_rand_score(adata.obs['cell_type'],np.argmax(adata.X, axis=1)))+"   Adjusted_mutual_info_score   "+str(adjusted_mutual_info_score(adata.obs['cell_type'],np.argmax(adata.X, axis=1))))
            print(purity_score(adata.obs['cell_type'],np.argmax(adata.X, axis=1)))
            return theta
    
    def make_lr_scheduler(self, optimizer,):
        lr_scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma = 0.5, verbose=True) 
        return lr_scheduler

    def train(self, data_loader, test_data, vocab, num_top_gene, test_label):
        optimizer = self.make_optimizer()
       # theta = None
        if "lr_scheduler" in self.args:
            print("===>Warning: use lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(data_loader.dataset)
        start = time.time()
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            #for batch_data, data_exa in data_loader:
            for iter, (text, image, neigh_text, neigh_image) in enumerate(data_loader):

                rst_dict = self.model(text.cuda(), image.float().cuda(), neigh_text.cuda(), neigh_image.float().cuda(),flag=epoch)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()#torch.ones_like(batch_loss)
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * len(text)

            if 'lr_scheduler' in self.args:
                lr_scheduler.step()

            output_log = f'Epoch: {epoch:03d}'
            for key in loss_rst_dict:
                output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'
            print(output_log)
            if epoch%100 == 0:
                # print(epoch)
                end = time.time()
                print("Method   time "+str(end - start))
                self.theta = self.evaluate(test_data, vocab, num_top_gene, test_label)

        beta = self.model.get_beta().detach().cpu().numpy()

        return beta

    def test(self, input_data):
        theta = None
        with torch.no_grad():
            self.model.eval()
            for ext, image, neigh_text, neigh_image in input_data:
                batch_theta = self.model.get_theta(ext.cuda())

                if theta == None:
                    theta = batch_theta.cpu()
                else:
                    theta = torch.cat((theta,batch_theta.cpu()))
        return  theta.numpy()
