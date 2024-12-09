import numpy as np
import torch
import geomloss
from torch import nn
import torch.nn.functional as F
import torch.distributions as distributions
from models.ECR import ECR
from torch.distributions import Normal, Independent

def entropy(logit):
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
    def __init__(self, class_num, temperature):
        super(DistillLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.mask = self.mask_correlated_clusters(class_num).cuda()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)
        c = F.normalize(c, dim=1)
        sim = c @ c.T / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels) / N
        return loss

class ClusterHead(nn.Module):
    def __init__(self, in_dim=512, num_clusters=10):
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_head_image = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, num_clusters),
        )

    def forward(self, image, flag=0):
        logit_image = self.cluster_head_image(image)
        if flag==0:
            logit_image = F.softmax(logit_image, dim=1)
        return logit_image

class ECRTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = 0
        self.args = args
        self.beta_temp = args.beta_temp

        self.a = 1 * np.ones((1, args.num_topic)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / args.num_topic))).T + (1.0 / (args.num_topic * args.num_topic)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False
        self.fc11 = nn.Linear(args.vocab_size, args.en1_units)
        self.fc12 = nn.Linear(args.en1_units, args.en1_units)
        self.fc21 = nn.Linear(args.en1_units, args.num_topic)
        self.fc22 = nn.Linear(args.en1_units, args.num_topic)
        self.CH = ClusterHead(in_dim=512, num_clusters=100).cuda()
        self.distill_loss = DistillLoss(class_num = 100, temperature = 0.5) 

        self.fc1_dropout = nn.Dropout(args.dropout)
        self.cla = nn.Sequential(
            nn.Softmax(dim=1)
        )
        self.enc = nn.Sequential(
            nn.Linear(args.vocab_size, args.en1_units),
            nn.BatchNorm1d(args.en1_units),
            nn.Tanh(),
            nn.Linear(args.en1_units, args.en1_units),
        )

        self.mean_bn = nn.BatchNorm1d(args.num_topic)
        self.logvar_bn = nn.BatchNorm1d(args.num_topic)
        self.decoder_bn = nn.BatchNorm1d(args.vocab_size)

        self.gene_embeddings = torch.from_numpy(args.gene_embeddings).float()
        self.gene_embeddings = nn.Parameter(F.normalize(self.gene_embeddings))

        self.topic_embeddings = torch.empty((args.num_topic, self.gene_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std = 0.02)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.ECR = ECR(self.args.weight_loss_ECR)

    def get_beta(self):
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.gene_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def get_embedding(self):
        return self.topic_embeddings,self.gene_embeddings

    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, input, flag = 1):
        e1 = self.enc(input) 

        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)

        loss_KL = self.compute_loss_KL(mu, logvar)
        if flag == 1:
            return z, loss_KL
        else:
            cla = self.cla(z)
            return cla

    def get_theta(self, input):
        theta, loss_KL = self.encode(input)
        if self.training:
            return theta, loss_KL
        else:
            return theta

    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.args.num_topic)
        KLD = KLD.mean()
        return KLD

    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(self.topic_embeddings, self.gene_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR
    
    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def forward(self, cellint, cellext, neigh_cellint, neigh_cellext, flag = 0, z2 = None):
        z, loss_KL = self.encode(cellint)
        theta = F.softmax(z, dim=1)
        beta = self.get_beta()
        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(cellint * recon.log()).sum(axis=1).mean()
        loss_RE = recon_loss + 1*loss_KL
        loss_ECR = self.get_loss_ECR()


        logit_cellext = self.CH(cellext)
        neigh_logit_cellext = self.CH(neigh_cellext)
        neigh_logit_cellint = self.encode(neigh_cellint, flag = 0)
        logit_cellint = self.encode(cellint, flag = 0)
        loss_nei = self.distill_loss(logit_cellext, neigh_logit_cellint) + self.distill_loss(logit_cellint, neigh_logit_cellext)
        loss_con = consistency_loss(logit_cellint, logit_cellext)
        loss_reg = entropy(logit_cellint) + entropy(logit_cellext)
        loss_CME = loss_nei + 1*loss_con - 5* loss_reg 
        loss = loss_RE + loss_ECR + 1*loss_CME

        rst_dict = {
            'loss': loss,
            'loss_TM': loss_RE,
            'loss_ECR': loss_ECR
        }

        return rst_dict
