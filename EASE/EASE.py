import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EASE:
    def __init__(self, args):
        self.reg = args.reg_weight
        self.device = args.device
    
    def fit(self, X):
        G = X.T @ X
        diagIndices = torch.eye(G.shape[0]) == 1
        G[diagIndices] += self.reg

        P = G.inverse()
        B = P / (-1 * P.diag())
        B[diagIndices] = 0
        
        self.B = B

    def forward(self, X):
        output = X @ self.B
        return output
    

    def predict(self, dataset):
        X = dataset.sparse_matrix.todense()
        user2rec = {}

        recon_mat = self.pred.cpu()
        score = recon_mat * torch.from_numpy(1 - X)
        rec_list = score.argsort(dim = 1)

        for user, rec in enumerate(rec_list):
            up = rec[-10:].cpu().numpy().tolist()
            user2rec[user] = up
        
        return user2rec

class MultiEASE(nn.Module):
    def __init__(self, args, num_items):
        super(MultiEASE, self).__init__()
        self.reg = args.reg_weight
        self.device = args.device
        self.B = nn.Parameter(torch.zeros(num_items, num_items))    
        
        diag_mask = 1 - torch.eye(num_items)
        self.register_buffer("diag_mask", diag_mask)
       

    def forward(self, X):
        X.to(self.device)
        output = X @ self.B
        output = F.log_softmax(output, dim=-1)

        return output
    
    def calculate_loss(self, X, output):
        loss = -(output * X).sum(dim=-1).mean()    
        reg_loss = self.reg * torch.sum(self.B ** 2 * self.diag_mask)
        
        return loss + reg_loss