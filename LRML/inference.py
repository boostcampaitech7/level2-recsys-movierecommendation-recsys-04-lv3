from lrml import LRML
from tqdm import tqdm
from dataset import PairwiseDataset
from utils import set_seed
import pandas as pd
import numpy as np
import torch
import argparse
import yaml

if __name__ == '__main__':
     # argument parser
    args =  argparse.ArgumentParser()
    args.add_argument("--config", default="config.yaml")
    args.add_argument("--model_path", default="LRML_model.pth")
    args.add_argument("--output_path", default="recommendations.csv")
    parsed_args = args.parse_args()
    
    # config file load
    with open(parsed_args.config) as f:
        config = yaml.load(f)    
    
    set_seed(config['seed'])
    data = pd.read_csv(config['data_path'])

    user_map = {user: idx for idx, user in enumerate(data['user'].unique())}
    item_map = {item: idx for idx, item in enumerate(data['item'].unique())}
    idx_user_map = {idx: user for user, idx in user_map.items()}
    idx_item_map = {idx: item for item, idx in item_map.items()}

    dataset = PairwiseDataset(
        data, 
        user_map=user_map, 
        item_map=item_map,
        idx_user_map=idx_user_map,
        idx_item_map=idx_item_map,
        test=True
    )

    # 모델 로드
    model = LRML(dataset.num_users, dataset.num_items,
                embedding_dim=config['embedding_dim'], 
                memory_size=config['memory_size'],
                margin=config['margin'],
                reg_weight=config['reg_weight']
                )
    model.load_state_dict(torch.load(parsed_args.model_path))
    model = model.to('cuda')

    # 추천 생성
    all_users = torch.arange(dataset.num_users).to(config['device'])
    all_items = torch.arange(dataset.num_items).to(config['device'])

    # Create a meshgrid of all user-item combinations
    user_tensor, item_tensor = torch.meshgrid(all_users, all_items, indexing='ij')
    user_tensor = user_tensor.flatten().to(config['device'])
    item_tensor = item_tensor.flatten().to(config['device'])

    # Compute scores in batches to manage memory
    all_scores = []
    batch_size = 100000  # Use the chunk_size from your model

    with torch.no_grad():
        for i in tqdm(range(0, len(user_tensor), batch_size), desc='cal score'):
            batch_users = user_tensor[i:i+batch_size]
            batch_items = item_tensor[i:i+batch_size]
            
            # Compute scores for this batch
            batch_scores = model(batch_users, batch_items).cpu().numpy()
            all_scores.append(batch_scores)

    # Concatenate all scores
    all_scores = np.concatenate(all_scores)
    all_scores = all_scores.reshape(dataset.num_users, dataset.num_items)

    matrix = dataset.create_interaction_matrix()
    results = []
    k = 10
    for user in tqdm(range(dataset.num_users), desc='get top 10'):
        user_pred_scores = all_scores[user]
        
        # 이미 train에 쓰인 아이템은 제거    
        user_pred_scores[matrix[user].toarray()[0] == 1] = -float('inf')
        top_k = np.argsort(user_pred_scores)[::-1][:k]
        # top_k_set 저장
        for item in top_k:
            results.append([user, item])
            
    results = pd.DataFrame(results, columns=['user', 'item'])
    results['user'] = results['user'].map(dataset.idx_user_map)
    results['item'] = results['item'].map(dataset.idx_item_map)
    results.to_csv(parsed_args.output_path, index=False)
    print("submission saved")