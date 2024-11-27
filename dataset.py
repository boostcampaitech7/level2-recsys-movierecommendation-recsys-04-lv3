# dataset.py
import torch
from torch.utils.data import Dataset

class MultiVAEDataset(Dataset):
    def __init__(self, user_item_matrix, train_mask, test_mask):
        self.user_item_matrix = user_item_matrix
        self.train_mask = train_mask
        self.test_mask = test_mask

    def __len__(self):
        return len(self.user_item_matrix)

    def __getitem__(self, idx):
        return (
            self.user_item_matrix[idx], 
            self.train_mask[idx], 
            self.test_mask[idx]
        )