import os
import pandas as pd
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class MovieLensDataset(Dataset):
    def __init__(self, data_path, transform=None, save_mapping=True):
        self.transform = transform
        self.data = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
        
        # Create mappings for users and items to continuous indices
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(self.data['user'].unique())}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(self.data['item'].unique())}
        
        self.n_users = len(self.user_mapping)
        self.n_items = len(self.item_mapping)
        
        # Save mappings if needed
        if save_mapping:
            self._save_mappings(data_path)

        # Map users and items to continuous indices
        self.data['user'] = self.data['user'].map(self.user_mapping)
        self.data['item'] = self.data['item'].map(self.item_mapping)
        
        # Create sparse matrix
        rows = self.data['user'].values
        cols = self.data['item'].values
        values = np.ones_like(rows)
        
        self.sparse_matrix = sparse.csr_matrix((values, (rows, cols)), 
                                             shape=(self.n_users, self.n_items))

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        user_data = self.sparse_matrix[idx].toarray().flatten()
        return torch.FloatTensor(user_data)

    def _save_mappings(self, data_path):
        # Save user and item mappings as CSV files
        user_mapping_df = pd.DataFrame(list(self.user_mapping.items()), 
                                     columns=['user', 'user_idx'])
        item_mapping_df = pd.DataFrame(list(self.item_mapping.items()), 
                                     columns=['item', 'item_idx'])
        
        user_mapping_df.to_csv(os.path.join(data_path, 'user_id_mapping.csv'), 
                              index=False)
        item_mapping_df.to_csv(os.path.join(data_path, 'item_id_mapping.csv'), 
                              index=False)

def get_data_loader(config):
    # 전체 데이터셋 생성
    dataset = MovieLensDataset(config.data_path)
    
    # Set n_users and n_items in the config
    config.n_users = dataset.n_users
    config.n_items = dataset.n_items
    
    # 학습/검증 데이터 분할 (90% 학습, 10% 검증)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # GPU 메모리 사용 최적화
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset.n_users, dataset.n_items