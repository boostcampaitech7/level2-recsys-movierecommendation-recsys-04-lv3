import os
import pandas as pd
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import Dataset, DataLoader

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
    dataset = MovieLensDataset(config.data_path)
    
    # Set n_users and n_items in the config
    config.n_users = dataset.n_users
    config.n_items = dataset.n_items
    
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    return data_loader, dataset.n_users, dataset.n_items