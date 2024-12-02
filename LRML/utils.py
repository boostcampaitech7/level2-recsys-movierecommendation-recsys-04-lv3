from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import torch

def split_data_sklearn(data, test_num):
    user_map = {user: idx for idx, user in enumerate(data['user'].unique())}
    item_map = {item: idx for idx, item in enumerate(data['item'].unique())}
    idx_user_map = {idx: user for user, idx in user_map.items()}
    idx_item_map = {idx: item for item, idx in item_map.items()}
    
    test_data = []
    train_data = []
    
    for user, group in tqdm(data.groupby('user'), desc='train valid split', total=len(data.groupby('user'))):
        if len(group) > test_num:
            # scikit-learn의 train_test_split 사용
            train_interactions, test_interactions = train_test_split(
                group, 
                test_size=test_num/len(group), 
                shuffle=True, 
                random_state=42  # 재현성을 위한 랜덤 시드
            )
        else:
            # 그룹 크기가 작으면 랜덤 샘플링 대신 마지막 항목 사용
            test_interactions = group.tail(1)
            train_interactions = group.iloc[:-1]
        
        test_data.append(test_interactions)
        train_data.append(train_interactions)
    
    return pd.concat(train_data), pd.concat(test_data), user_map, item_map, idx_user_map, idx_item_map

def set_seed(seed):
    random.seed(seed)  # Python 기본 random seed 설정
    np.random.seed(seed)  # Numpy random seed 설정
    torch.manual_seed(seed)  # PyTorch CPU 연산에 대한 seed 설정
    torch.cuda.manual_seed(seed)  # PyTorch GPU 연산에 대한 seed 설정
    torch.cuda.manual_seed_all(seed)  # Multi-GPU 환경에서 모든 GPU에 대해 seed 설정
    torch.backends.cudnn.deterministic = True  # CuDNN에서 determinism 보장
    torch.backends.cudnn.benchmark = False  # 성능 최적화 기능 비활성화