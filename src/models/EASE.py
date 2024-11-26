import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src.utils.utils import get_hit, get_ndcg, get_recall

class EASE:
    def __init__(self, args):
        self.reg = args.reg_weight
        self.device = args.device
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        # matrix -> tensor
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)
    
    def fit(self, dataset):
        X = dataset.sparse_matrix
        X = self._convert_sp_mat_to_sp_tensor(X)
        G = X.to_dense().t() @ X.to_dense()
        diagIndices = torch.eye(G.shape[0]) == 1
        G[diagIndices] += self.reg

        P = G.inverse()
        B = P / (-1 * P.diag())
        B[diagIndices] = 0
        
        self.B = B
        self.pred = X.to_dense() @ B

    def evaluate(self, dataset):
        X = dataset.sparse_matrix.todense()
        mat = torch.from_numpy(X)

        NDCG = 0.0 # NDCG@10
        HIT = 0.0 # HIT@10
        RECALL = 0.0 # RECALL@10

        recon_mat = self.pred.cpu()
        recon_mat[mat == 1] = -np.inf
        rec_list = recon_mat.argsort(dim = 1)

        user_valid = dataset.user_valid
        user_list = user_valid.keys()
        for user, rec in enumerate(rec_list):
            if user in user_list:
                uv = user_valid[user]
                up = rec[-10:].cpu().numpy().tolist()[::-1]
                NDCG += get_ndcg(pred_list = up, true_list = uv)
                HIT += get_hit(pred_list = up, true_list = uv)
                RECALL += get_recall(pred_list = up, true_list = uv)

        NDCG /= len(user_list)
        HIT /= len(user_list)
        RECALL /= len(user_list)

        return NDCG, HIT, RECALL
    

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