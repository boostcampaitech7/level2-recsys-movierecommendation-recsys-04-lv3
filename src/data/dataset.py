import os
import pandas as pd
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict

class MovieLensDataset(Dataset):
    def __init__(self, config, data_type = 'train'):
        self.config = config
        self.data_type = data_type
        self.data = pd.read_csv(os.path.join(self.config.data_dir, 'train_ratings.csv'))
        
        # Create mappings for users and items to continuous indices
        self.user_encoder = {user_id: idx for idx, user_id in enumerate(self.data['user'].unique())}
        self.item_encoder = {item_id: idx for idx, item_id in enumerate(self.data['item'].unique())}

        self.user_decoder = {idx: user_id for user_id, idx in self.user_encoder.items()}
        self.item_decoder = {idx: item_id for item_id, idx in self.item_encoder.items()}
        
        self.n_users = len(self.user_encoder)
        self.n_items = len(self.item_encoder)

        # Map users and items to continuous indices
        self.data['user'] = self.data['user'].map(self.user_encoder)
        self.data['item'] = self.data['item'].map(self.item_encoder)

        self.user_train, self.user_valid = self._generate_sequence_data()
        self.sparse_matrix = self.make_sparse_matrix()
        
    
    def _generate_sequence_data(self):
        users = defaultdict(list)
        user_train, user_valid = {}, {}
        
        for user, item in zip(self.data['user'], self.data['item']):
            users[user].append(item)
        
        for user in users:
            # np.random.seed(self.config.seed)
            user_total = users[user]
            valid_samples = int(len(user_total) * self.config.valid_size)
            valid = np.random.choice(user_total, size=valid_samples, replace=False).tolist()
            train = list(set(user_total) - set(valid))

            user_train[user] = train
            user_valid[user] = valid

        return user_train, user_valid

    def make_sparse_matrix(self):
        X = sparse.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        
        if self.data_type in ['train', 'valid']: 
            for user in self.user_train.keys():
                item_list = self.user_train[user]
                X[user, item_list] = 1.0
        
        if self.data_type == 'valid':
            for user in self.user_valid.keys():
                item_list = self.user_valid[user]
                X[user, item_list] = 1.0

        return X.tocsr()
    
    def __len__(self):
        return self.n_users

    def __getitem__(self, index):
        user_data = self.sparse_matrix[index].toarray().flatten()
        return torch.FloatTensor(user_data)
    