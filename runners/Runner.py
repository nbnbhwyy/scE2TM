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
from sklearn.metrics import f1_score,precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset
import os

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
        neighbor_expression_idx = np.random.choice(self.expression_indices[index], 1)[0]
        neighbor_expression = self.expression_data.__getitem__(neighbor_expression_idx)
        neighbor_embedding_idx = np.random.choice(self.embedding_indices[index], 1)[0]
        neighbor_embedding = self.embedding_data.__getitem__(neighbor_embedding_idx)

        return anchor_expression, anchor_embedding, neighbor_expression, neighbor_embedding

class ExpressionEmbeddingNeighborDatasetWithLatent(Dataset):
    def __init__(self, expression_data, embedding_data, expression_indices, embedding_indices, latent_variables):
        super(ExpressionEmbeddingNeighborDatasetWithLatent, self).__init__()

        self.expression_data = expression_data
        self.embedding_data = embedding_data
        self.expression_indices = expression_indices
        self.embedding_indices = embedding_indices
        self.latent_variables = latent_variables
        assert self.expression_indices.shape[0] == len(self.expression_indices)
        assert self.embedding_indices.shape[0] == len(self.embedding_indices)

    def __len__(self):
        return len(self.expression_data)

    def __getitem__(self, index):
        anchor_expression = self.expression_data.__getitem__(index)
        anchor_embedding = self.embedding_data.__getitem__(index)
        latent_var = self.latent_variables.__getitem__(index)
        
        neighbor_expression_idx = np.random.choice(self.expression_indices[index], 1)[0]
        neighbor_latent = self.latent_variables.__getitem__(neighbor_expression_idx)
        
        neighbor_embedding_idx = np.random.choice(self.embedding_indices[index], 1)[0]
        neighbor_embedding = self.embedding_data.__getitem__(neighbor_embedding_idx)

        return anchor_expression, anchor_embedding, neighbor_latent, neighbor_embedding, latent_var


def mine_nearest_neighbors(features, topk=50):
    print("Computing nearest neighbors...")
    features = features.astype(np.float32)
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatIP(dim)
    if torch.cuda.is_available():
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk + 1)  # Sample itself is included
    print("Nearest neighbors computed.")
    return indices[:, 1:]

def extract_topic_genes(beta, gene_vocab, num_top_genes):
    topic_str_list = list()
    for i, topic_dist in enumerate(beta):
        topic_genes = np.array(gene_vocab)[np.argsort(topic_dist)][:-(num_top_genes + 1):-1]
        topic_str = ' '.join(topic_genes)
        topic_str_list.append(topic_str)
    return topic_str_list

def compute_topic_coherence(pathway_matrix, topic_gene_matrix, num_top_genes, gene_index_mapping):
    topic_size, gene_size = np.shape(topic_gene_matrix)
    pathway_size = np.shape(pathway_matrix)[0]
    
    # find top genes' index for each topic
    topic_gene_indices = []
    for topic_idx in range(topic_size):
        top_gene_idx = np.argpartition(topic_gene_matrix[topic_idx, :], -num_top_genes)[-num_top_genes:]
        topic_gene_indices.append(top_gene_idx)
    
    # compute coherence for each topic
    total_coherence = 0.0
    for i in range(topic_size):
        gene_array = topic_gene_indices[i]
        topic_coherence = 0.0
        for n in range(num_top_genes):
            if gene_array[n] in gene_index_mapping:
                flag_n = pathway_matrix[:, gene_index_mapping[gene_array[n]]] > 0
                p_n = np.sum(flag_n) / pathway_size
                for l in range(n + 1, num_top_genes):
                    if gene_array[l] in gene_index_mapping:
                        flag_l = pathway_matrix[:, gene_index_mapping[gene_array[l]]] > 0
                        p_l = np.sum(flag_l)
                        p_nl = np.sum(flag_n * flag_l)
                        if p_n * p_l * p_nl > 0:
                            p_l = p_l / pathway_size
                            p_nl = p_nl / pathway_size
                            topic_coherence += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
        total_coherence += topic_coherence * (2 / (num_top_genes * num_top_genes - num_top_genes))
    total_coherence = total_coherence / topic_size
    return total_coherence


def compute_topic_diversity(topic_strings):
    K = len(topic_strings)
    T = len(topic_strings[0].split())
    vectorizer = CountVectorizer()
    counter = vectorizer.fit_transform(topic_strings).toarray()

    TF = counter.sum(axis=0)
    TD = (TF == 1).sum() / (K * T)

    return TD

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def compute_f1_scores(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

    return (precision_macro, recall_macro, f1_macro), (precision_micro, recall_micro, f1_micro)

def evaluate_classification(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data , labels, test_size=0.3)
    clf = LogisticRegression().fit(X_train, y_train) 
    y_pred = clf.predict(np.array(X_test))
    (precision_macro, recall_macro, f1_macro), (precision_micro, recall_micro, f1_micro) = np.round(compute_f1_scores(np.array(y_test), np.array(y_pred)), 5)  # f1 -> compute_f1_scores
    print('F1 score: f1_macro = {}, f1_micro = {}'.format(f1_macro, f1_micro))
    print('precision score: precision_macro = {}, precision_micro = {}'.format(precision_macro, precision_micro))
    print('recall score: recall_macro = {}, recall_micro = {}'.format(recall_macro, recall_micro))

class Runner:
    """Main runner class for training and evaluation"""
    def __init__(self, args):
        self.args = args
        self.model = scE2TM(args)
        self.topic_distribution = None 
        self.dataloader = None

    def make_optimizer(self,):
        optimizer_config = {  
            'params': self.model.parameters(),
            'lr': self.args.learning_rate,
        }

        optimizer = torch.optim.RMSprop(**optimizer_config)
        return optimizer

    def evaluate(self, test_loader, gene_vocab, num_top_genes, cell_labels):
        with torch.no_grad():
            model_save_dir = os.path.join(self.args.output_dir, self.args.dataset_name)
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(self.model, os.path.join(model_save_dir, f'{self.args.dataset_name}.pth'))
            
            self.model.eval()
            beta = self.model.get_topic_gene_distribution()
            topic_embedding, gene_embedding = self.model.get_embeddings()
            beta = beta.detach().cpu().numpy()
            topic_distribution = self.test(test_loader)

            # 计算 topic diversity（不依赖标签）
            topic_strings = extract_topic_genes(beta, gene_vocab, num_top_genes)
            topic_diversity = compute_topic_diversity(topic_strings)
            print(f"===>Topic Diversity (top {num_top_genes}): {topic_diversity:.5f}")

            # 计算 topic coherence（不依赖标签）
            gene_set_path = './data/msigdb.v2024.1.Hs.symbols.gmt'
            kegg = gp.read_gmt(path=gene_set_path)
            all_genes = []
            for value in kegg.values():
                all_genes.extend(value)
            all_genes = list(set(all_genes))
            gene_to_idx = {}
            for index, value in enumerate(all_genes):
                gene_to_idx[value] = index
            pathway_matrix = np.zeros((len(kegg), len(all_genes)))
            for index, values in enumerate(kegg.values()):
                for value in values:
                    pathway_matrix[index][gene_to_idx[value]] = 1
            gene_idx_mapping = {}
            for index, gene in enumerate(gene_vocab):
                if gene in gene_to_idx:
                    gene_idx_mapping[index] = gene_to_idx[gene]
            topic_coherence = compute_topic_coherence(pathway_matrix, beta, num_top_genes, gene_idx_mapping)
            print(f"===>Topic Coherence (top {num_top_genes}): {topic_coherence:.5f}")

            # 保存输出文件（不依赖标签）
            pd.DataFrame(beta).to_csv(os.path.join(model_save_dir, f'{self.args.dataset_name}_tg.csv'))
            pd.DataFrame(topic_distribution).to_csv(os.path.join(model_save_dir, f'{self.args.dataset_name}_topic_distribution.csv'))
            pd.DataFrame(topic_embedding.detach().cpu().numpy()).to_csv(os.path.join(model_save_dir, f'{self.args.dataset_name}_topic_embedding.csv'))
            pd.DataFrame(gene_embedding.detach().cpu().numpy()).to_csv(os.path.join(model_save_dir, f'{self.args.dataset_name}_gene_embedding.csv'))

            # ===== 依赖标签的评估部分 =====
            if cell_labels is not None:
                adata = sc.AnnData(topic_distribution)
                adata.obs['cell_type'] = cell_labels
                sc.pp.pca(adata)
                sc.pp.neighbors(adata, use_rep='X')
                max_resolution = 2
                min_resolution = 0
                ari_scores = []
                for res in range(min_resolution, max_resolution * 10):
                    sc.tl.louvain(adata, resolution=res / 10.0, random_state=0)
                    ari_scores.append(adjusted_rand_score(adata.obs['cell_type'], adata.obs['louvain']))
                best_resolution = np.argmax(ari_scores) * 0.1
                sc.tl.louvain(adata, resolution=best_resolution, random_state=0)

                ari = adjusted_rand_score(adata.obs['cell_type'], adata.obs['louvain'])
                ami = adjusted_mutual_info_score(adata.obs['cell_type'], adata.obs['louvain'])
                asw = silhouette_score(adata.X, adata.obs['cell_type'])
                purity = purity_score(adata.obs['cell_type'], np.argmax(adata.X, axis=1))
                print(f"scE2TM ARI: {ari:.4f}, AMI: {ami:.4f}, ASW: {asw:.4f}, Purity: {purity:.4f}")

                evaluate_classification(topic_distribution, cell_labels)
                print(f"scE2TM Purity: {purity:.4f}")
            else:
                print("No cell labels provided. Skipping clustering and classification evaluation.")

            return topic_distribution
    
    def make_lr_scheduler(self, optimizer,):
        lr_scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=0.5, verbose=True) 
        return lr_scheduler

    def train(self, train_loader, test_loader, gene_vocab, num_top_genes, cell_labels):
        """
        Train the model
        
        Args:
            train_loader: training data loader
            test_loader: test data loader
            gene_vocab: vocabulary of gene names
            num_top_genes: number of top genes per topic
            cell_labels: ground truth cell type labels
        Returns:
            beta: topic-gene distribution
        """
        
        optimizer = self.make_optimizer()
 
        # if "lr_scheduler" in self.args:
        if hasattr(self.args, "lr_scheduler"):
            print("===>Warning: use lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(train_loader.dataset)
        start = time.time()
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            loss_dict = defaultdict(float)

            for iter, (expression, embedding, neighbor_expression, neighbor_embedding) in enumerate(train_loader):

                expression = expression.to(self.args.device)
                embedding = embedding.float().to(self.args.device)
                neighbor_expression = neighbor_expression.to(self.args.device)
                neighbor_embedding = neighbor_embedding.float().to(self.args.device)
                
                result_dict = self.model(expression, embedding, neighbor_expression, neighbor_embedding, epoch=epoch)
                batch_loss = result_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in result_dict:
                    loss_dict[key] += result_dict[key] * len(expression)

            # if 'lr_scheduler' in self.args:
            if hasattr(self.args, "lr_scheduler"):
                lr_scheduler.step()

            output_log = f'Epoch: {epoch:03d}'
            for key in loss_dict:
                output_log += f' {key}: {loss_dict[key] / data_size:.3f}'
            print(output_log)
            if epoch % 100 == 0:
                elapsed_time = time.time() - start 
                print("Method time: " + str(elapsed_time))
                self.topic_distribution = self.evaluate(test_loader, gene_vocab, num_top_genes, cell_labels)  
        beta = self.model.get_topic_gene_distribution().detach().cpu().numpy() 

        return beta

    def test(self, test_loader):  
        all_theta = []
        with torch.no_grad():
            self.model.eval()
            for expression, embedding, neighbor_expression, neighbor_embedding in test_loader:
                expression = expression.to(self.args.device)
                batch_theta = self.model.get_topic_distribution(expression) 

                if len(all_theta) == 0:
                    all_theta = batch_theta.cpu()
                else:
                    all_theta = torch.cat((all_theta, batch_theta.cpu()))
        return all_theta.numpy()