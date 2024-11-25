import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import bottleneck as bn
from torch.utils.tensorboard import SummaryWriter

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class Trainer:
    def __init__(self, model, optimizer, config, device):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.update_count = 0
        self.early_stopping = EarlyStopping(patience=10, verbose=True)
        
        self._create_directories()

    def _create_directories(self):
        directories = [
            self.config.model_save_path,
            self.config.final_prediction_path,
            self.config.log_dir
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    @torch.cuda.amp.autocast()
    def train_one_epoch(self, data_loader, epoch):
        self.model.train()
        train_loss = 0.0
        start_time = time.time()
        scaler = torch.cuda.amp.GradScaler()

        for batch_idx, data in enumerate(data_loader):
            data = data.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            
            if self.config.total_anneal_steps > 0:
                anneal = min(self.config.anneal_cap,
                           1. * self.update_count / self.config.total_anneal_steps)
            else:
                anneal = self.config.anneal_cap
            
            with torch.cuda.amp.autocast():
                recon_batch, mu, logvar = self.model(data)
                loss = self.loss_function(recon_batch, data, mu, logvar, anneal)
            
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            train_loss += loss.item()
            self.update_count += 1
            
            if batch_idx % self.config.log_interval == 0 and batch_idx > 0:
                self._log_progress(epoch, batch_idx, len(data_loader), train_loss, start_time)
                start_time = time.time()
                train_loss = 0.0
                
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        return train_loss / len(data_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device, non_blocking=True)
                recon_batch, mu, logvar = self.model(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)

    @torch.cuda.amp.autocast()
    def generate_recommendations(self, dataset, n_items=10):
        """
        Generate recommendations for all users
        dataset: original MovieLensDataset instance (not the subset)
        """
        self.model.eval()
        recommendations = []
        
        # Load user mapping
        user_mapping_df = pd.read_csv(self.config.user_id_mapping_path)
        user_id_dict = dict(zip(user_mapping_df['user_idx'], user_mapping_df['user']))
        
        n_users = len(user_mapping_df)  # 전체 사용자 수
        batch_size = 100
        
        with torch.no_grad():
            for start_idx in range(0, n_users, batch_size):
                end_idx = min(start_idx + batch_size, n_users)
                
                # Get the full dataset's sparse matrix for these users
                user_data = torch.stack([
                    torch.FloatTensor(dataset.sparse_matrix[i].toarray().flatten()) 
                    for i in range(start_idx, end_idx)
                ]).to(self.device, non_blocking=True)
                
                recon_batch, _, _ = self.model(user_data)
                recon_batch = recon_batch.cpu().numpy()
                
                # Set scores of items in training set to -inf
                for i, idx in enumerate(range(start_idx, end_idx)):
                    user_train = dataset.sparse_matrix[idx].toarray().flatten()
                    recon_batch[i][user_train > 0] = -np.inf
                
                # Get top N items
                top_items = bn.argpartition(-recon_batch, n_items, axis=1)[:, :n_items]
                top_scores = np.take_along_axis(-recon_batch, top_items, axis=1)
                
                # Add recommendations to the list
                for i, (items, scores) in enumerate(zip(top_items, top_scores)):
                    user_idx = start_idx + i
                    original_user = user_id_dict[user_idx]
                    
                    for item, score in zip(items, scores):
                        recommendations.append({
                            'user': original_user,
                            'item': int(item),
                            'score': float(-score)
                        })
                
                if start_idx % 1000 == 0:
                    torch.cuda.empty_cache()
        
        return pd.DataFrame(recommendations)

    def _log_progress(self, epoch, batch_idx, data_loader_size, train_loss, start_time):
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
              'loss {:4.2f}'.format(
                  epoch, batch_idx, data_loader_size,
                  elapsed * 1000 / self.config.log_interval,
                  train_loss / self.config.log_interval))
        
        self.writer.add_scalar('training_loss', 
                             train_loss / self.config.log_interval,
                             epoch * data_loader_size + batch_idx)

    def loss_function(self, recon_x, x, mu, logvar, anneal=1.0):
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return BCE + anneal * KLD

    def save_checkpoint(self, epoch, loss, best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'update_count': self.update_count
        }
        if best:
            path = os.path.join(self.config.model_save_path, f'best_model.pt')
        else:
            path = os.path.join(self.config.model_save_path, f'model_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        
    