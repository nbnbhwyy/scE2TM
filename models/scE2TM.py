import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.ECR import ECR

def entropy(logit):
    """Calculate entropy loss"""
    logit = logit.mean(dim=0)
    logit_ = torch.clamp(logit, min=1e-9)
    b = logit_ * torch.log(logit_)
    return -b.sum()


def consistency_loss(anchors, neighbors):
    b, n = anchors.size()
    similarity = torch.bmm(anchors.view(b, 1, n), neighbors.view(b, n, 1)).squeeze()
    ones = torch.ones_like(similarity)
    consistency_loss = F.binary_cross_entropy(similarity, ones, reduction='mean') 
    return consistency_loss


class DistillLoss(nn.Module):
    def __init__(self, num_classes, temperature, device):  
        super(DistillLoss, self).__init__()
        self.num_classes = num_classes  
        self.temperature = temperature
        self.mask = self.mask_correlated_clusters(num_classes).to(device)  
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.device = device

    def mask_correlated_clusters(self, num_classes):  
        N = 2 * num_classes  
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(num_classes):  
            mask[i, num_classes + i] = 0  
            mask[num_classes + i, i] = 0  
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.num_classes  
        c = torch.cat((c_i, c_j), dim=0)
        c = F.normalize(c, dim=1)
        sim = c @ c.T / self.temperature
        sim_i_j = torch.diag(sim, self.num_classes)  
        sim_j_i = torch.diag(sim, -self.num_classes)  
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels) / N
        return loss

class CrossModalClusterHead(nn.Module):
    def __init__(self, input_dim=512, num_clusters=100): 
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, num_clusters),
        )

    def forward(self, features, return_logits=False): 
        logits = self.cluster_head(features)
        if not return_logits:
            logits = F.softmax(logits, dim=1)
        return logits

class scE2TM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device 
        self.args = args
        self.beta_temp = args.beta_temp

        self.prior_alpha = 1 * np.ones((1, args.num_topics)).astype(np.float32)
        self.prior_mu = nn.Parameter(torch.as_tensor((np.log(self.prior_alpha).T - np.mean(np.log(self.prior_alpha), 1)).T))
        self.prior_var = nn.Parameter(torch.as_tensor((((1.0 / self.prior_alpha) * (1 - (2.0 / args.num_topics))).T + (1.0 / (args.num_topics * args.num_topics)) * np.sum(1.0 / self.prior_alpha, 1)).T))

        self.prior_mu.requires_grad = False  
        self.prior_var.requires_grad = False 
        
        self.encoder_fc1 = nn.Linear(args.vocab_size, args.en1_units) 
        self.encoder_fc2 = nn.Linear(args.en1_units, args.en1_units)
        self.encoder_mu = nn.Linear(args.en1_units, args.num_topics)
        self.encoder_logvar = nn.Linear(args.en1_units, args.num_topics)
        
        self.cross_modal_cluster_head = CrossModalClusterHead( 
            input_dim=512, 
            num_clusters=args.num_topics
        )
        
        self.distill_loss = DistillLoss(
            num_classes=args.num_topics, 
            temperature=0.5, 
            device=args.device
        ) 

        self.encoder_dropout = nn.Dropout(args.dropout) 
        self.classifier = nn.Sequential(  
            nn.Softmax(dim=1)
        )
        self.encoder = nn.Sequential(
            nn.Linear(args.vocab_size, args.en1_units),
            nn.BatchNorm1d(args.en1_units),
            nn.Tanh(),
            nn.Linear(args.en1_units, args.en1_units),
        )

        self.mean_bn = nn.BatchNorm1d(args.num_topics)
        self.logvar_bn = nn.BatchNorm1d(args.num_topics)
        self.decoder_bn = nn.BatchNorm1d(args.vocab_size)

        self.gene_embeddings = torch.from_numpy(args.gene_embeddings).float()
        self.gene_embeddings = nn.Parameter(F.normalize(self.gene_embeddings))

        self.topic_embeddings = torch.empty((args.num_topics, self.gene_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.02)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.ecr_loss = ECR(self.args.weight_loss_ECR)
        self.to(args.device)
     
    def get_topic_gene_distribution(self):
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.gene_embeddings) 
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def get_embeddings(self):
        return self.topic_embeddings, self.gene_embeddings

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode_expression(self, input, return_both=True): 
        e1 = self.encoder(input) 

        e1 = self.encoder_dropout(e1)
        mu = self.mean_bn(self.encoder_mu(e1)) 
        logvar = self.logvar_bn(self.encoder_logvar(e1))
        z = self.reparameterize(mu, logvar)

        kl_loss = self.compute_kl_loss(mu, logvar)
        if return_both: 
            return z, kl_loss
        else:
            class_dist = self.classifier(z)
            return class_dist

    def get_topic_distribution(self, input):
        theta, kl_loss = self.encode_expression(input, return_both=True)
        if self.training:
            return theta, kl_loss
        else:
            return theta

    def compute_kl_loss(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.prior_var  
        diff = mu - self.prior_mu 
        diff_term = diff * diff / self.prior_var  
        logvar_division = self.prior_var.log() - logvar  
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.args.num_topics) 
        KLD = KLD.mean()
        return KLD

    def compute_ecr_loss(self):
        cost = self.pairwise_euclidean_distance(self.topic_embeddings, self.gene_embeddings)  
        ecr_loss = self.ecr_loss(cost) 
        return ecr_loss
    
    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def forward(self, expression, foundation_embedding, neighbor_expression, neighbor_embedding, epoch):
        z, kl_loss = self.encode_expression(expression, return_both=True) 
        theta = F.softmax(z, dim=1)
        beta = self.get_topic_gene_distribution()
        reconstruction = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1) 
        recon_loss = -(expression * reconstruction.log()).sum(axis=1).mean()  
        topic_model_loss = recon_loss + kl_loss 
        
        ecr_loss = self.compute_ecr_loss() 

        cluster_logits_embedding = self.cross_modal_cluster_head(foundation_embedding) 
        neighbor_cluster_logits_embedding = self.cross_modal_cluster_head(neighbor_embedding)
        neighbor_cluster_logits_expression = self.encode_expression(neighbor_expression, return_both=False) 
        cluster_logits_expression = self.encode_expression(expression, return_both=False) 
        
        distill_loss_val = (self.distill_loss(cluster_logits_embedding, neighbor_cluster_logits_expression) + 
                           self.distill_loss(cluster_logits_expression, neighbor_cluster_logits_embedding))
        consist_loss_val = consistency_loss(cluster_logits_expression, cluster_logits_embedding)
        entropy_loss_val = entropy(cluster_logits_expression) + entropy(cluster_logits_embedding)
        
        cross_modal_loss = distill_loss_val + consist_loss_val - 5 * entropy_loss_val  
        total_loss = topic_model_loss + ecr_loss + cross_modal_loss  

        result_dict = {
            'loss': total_loss, 
            'topic_model_loss': topic_model_loss,  
            'cross_modal_loss': cross_modal_loss,  
            'ecr_loss': ecr_loss, 
        }

        return result_dict