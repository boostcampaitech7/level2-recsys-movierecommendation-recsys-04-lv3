import os
import torch
import numpy as np
from tqdm import tqdm
from src.utils.utils import get_hit, get_ndcg, get_recall, convert_sp_mat_to_sp_tensor


class EASETrainer:
    def __init__(self, model, args):
        self.model = model
        self.device = args.device

    def train(self, data_loader):
        loss_val = 0.0
        for mat, users in data_loader:
            mat = mat.to('cuda')
            self.model.fit(mat)
        return loss_val

    def evaluate(self, data_loader, user_valid):
        NDCG, HIT, RECALL = 0.0, 0.0, 0.0

        mat = next(iter(data_loader))[0]
        mat = mat.to('cuda')
        
        recon_mat = self.model.forward(mat)
        recon_mat[mat == 1] = -np.inf
        rec_list = recon_mat.argsort(dim = 1)

        user_list = user_valid.keys()
        for user, rec in enumerate(rec_list):
            if user in user_list:
                uv = user_valid[user]
                up = rec[-10:].cpu().numpy().tolist()[::-1]
                NDCG += get_ndcg(pred_list = up, true_list = uv)
                HIT += get_hit(pred_list = up, true_list = uv)
                RECALL += get_recall(pred_list = up, true_list = uv)

        NDCG /= len(data_loader.dataset)
        HIT /= len(data_loader.dataset)
        RECALL /= len(data_loader.dataset)

        return NDCG, HIT, RECALL

    def save_model(self, path, filename):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.B, os.path.join(path, filename))



class MultiEASETrainer:
    def __init__(self, model, optimizer, args):
        self.model = model
        self.optimizer = optimizer
        self.device = args.device

    def train(self, data_loader):
        self.model.train()
        loss_val = 0

        for mat, users in data_loader:
            mat = mat.to('cuda')
            
            output = self.model.forward(mat)
            
            self.optimizer.zero_grad()
            loss = self.model.calculate_loss(mat, output)

            loss_val += loss.item()

            loss.backward()
            self.optimizer.step()

        loss_val /= len(data_loader)
        return loss_val

    def evaluate(self, data_loader, user_valid):
        self.model.eval()
        NDCG, HIT, RECALL = 0.0, 0.0, 0.0

        with torch.no_grad():
            for mat, users in data_loader:
                mat = mat.to('cuda')
                
                output = self.model.forward(mat)
                output[mat == 1] = -np.inf
                rec_list = output.argsort(dim=1)

                for user, rec in zip(users, rec_list):
                    uv = user_valid[user.item()]
                    up = rec[-10:].cpu().numpy().tolist()[::-1]
                    NDCG += get_ndcg(pred_list=up, true_list=uv)
                    HIT += get_hit(pred_list=up, true_list=uv)
                    RECALL += get_recall(pred_list=up, true_list=uv)

        NDCG /= len(data_loader.dataset)
        HIT /= len(data_loader.dataset)
        RECALL /= len(data_loader.dataset)

        return NDCG, HIT, RECALL

    def save_model(self, path, filename):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, filename))


def full_sort_predict(model, data_loader):
    num_users = len(data_loader.dataset)
    preds = np.zeros((num_users, 10), dtype=int) 

    for batch, users in data_loader:
        batch = batch.to('cuda')
        pred = model.forward(batch)

        scores = pred * (1 - batch)
        scores = scores.argsort(dim=1)
        pred = scores[:, :10] 

        preds[users] = pred.cpu().numpy()
    return preds