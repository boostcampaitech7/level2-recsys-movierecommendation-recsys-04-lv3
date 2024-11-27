# utils.py
import os
import logging
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def setup_logging(log_dir, log_filename='training_log.txt'):
    """로깅 설정 함수"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path),  # 파일 핸들러
            logging.StreamHandler()  # 콘솔 핸들러
        ]
    )
    return logging.getLogger()

def load_rating_csv_to_interaction_matrix(file_path, device):
    df = pd.read_csv(file_path)
    
    unique_users = df['user'].unique()
    unique_items = df['item'].unique()
    
    user_id_map = {user: idx for idx, user in enumerate(unique_users)}
    item_id_map = {item: idx for idx, item in enumerate(unique_items)}
    
    df['user_idx'] = df['user'].map(user_id_map)
    df['item_idx'] = df['item'].map(item_id_map)
    
    interaction_matrix = csr_matrix(
        (np.ones(len(df)), (df['user_idx'], df['item_idx'])), 
        shape=(len(unique_users), len(unique_items))
    ).toarray()
    
    interaction_matrix = torch.tensor(interaction_matrix, dtype=torch.float32, device=device)
    
    return interaction_matrix, user_id_map, item_id_map

def stratified_split(interaction_matrix, test_size=0.2, seed=42):
    torch.manual_seed(seed)
    train_matrix = interaction_matrix.clone()
    test_matrix = torch.zeros_like(interaction_matrix)
    
    for user in range(interaction_matrix.shape[0]):
        interacted_items = torch.nonzero(interaction_matrix[user]).squeeze()
        
        if len(interacted_items) > 1:
            num_test_items = max(1, int(len(interacted_items) * test_size))
            test_items = torch.randperm(len(interacted_items))[:num_test_items]
            test_items = interacted_items[test_items]
            
            train_matrix[user, test_items] = 0
            test_matrix[user, test_items] = interaction_matrix[user, test_items]
    
    return train_matrix, test_matrix

def recall_at_k(predictions, test_matrix, k=10):
    recalls = []
    
    for user in range(predictions.shape[0]):
        test_items = torch.nonzero(test_matrix[user]).squeeze()
        
        if len(test_items) == 0:
            continue
        
        top_k_items = predictions[user].topk(k).indices
        
        # GPU에서 교집합 찾기
        hit_count = len(torch.tensor(list(set(top_k_items.cpu().numpy()) & set(test_items.cpu().numpy()))))
        recall = hit_count / len(test_items)
        recalls.append(recall)
    
    return torch.mean(torch.tensor(recalls)).item() if recalls else 0