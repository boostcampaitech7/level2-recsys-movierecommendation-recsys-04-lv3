import os
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse import csr_matrix
from typing import Tuple, Dict

def setup_logging(log_dir: str, log_filename: str = 'training_log.txt') -> logging.Logger:
    """
    로깅 설정 함수입니다.
    
    Args:
        log_dir (str): 로그 파일을 저장할 디렉토리 경로.
        log_filename (str): 로그 파일 이름 (기본값: 'training_log.txt').

    Returns:
        logging.Logger: 로거 객체.
    """
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

def load_rating_csv_to_interaction_matrix(file_path: str, device: torch.device) -> Tuple[torch.Tensor, Dict[int, int], Dict[int, int]]:
    """
    CSV 파일에서 사용자-아이템 상호작용 행렬을 생성합니다.
    
    Args:
        file_path (str): 사용자-아이템 데이터가 포함된 CSV 파일 경로.
        device (torch.device): 텐서를 저장할 디바이스 (예: 'cpu' 또는 'cuda').

    Returns:
        Tuple[torch.Tensor, Dict[int, int], Dict[int, int]]: 
            - interaction_matrix (torch.Tensor): 사용자-아이템 상호작용 행렬.
            - user_id_map (Dict[int, int]): 사용자 ID와 인덱스의 매핑.
            - item_id_map (Dict[int, int]): 아이템 ID와 인덱스의 매핑.
    """
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

def stratified_split(interaction_matrix: torch.Tensor, test_size: float = 0.2, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    사용자-아이템 상호작용 행렬을 훈련과 테스트로 분할하는 함수입니다.
    
    Args:
        interaction_matrix (torch.Tensor): 사용자-아이템 상호작용 행렬.
        test_size (float): 테스트 데이터 비율 (기본값: 0.2).
        seed (int): 랜덤 시드를 고정하기 위한 값 (기본값: 42).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - train_matrix (torch.Tensor): 훈련 데이터 행렬.
            - test_matrix (torch.Tensor): 테스트 데이터 행렬.
    """
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

def recall_at_k(predictions: torch.Tensor, test_matrix: torch.Tensor, k: int = 10) -> float:
    """
    주어진 예측값에 대해 recall@k를 계산하는 함수입니다.
    
    Args:
        predictions (torch.Tensor): 모델의 예측값.
        test_matrix (torch.Tensor): 테스트 데이터 행렬.
        k (int): top-k 아이템 (기본값: 10).

    Returns:
        float: recall@k 값.
    """
    recalls = []
    
    for user in range(predictions.shape[0]):
        test_items = torch.nonzero(test_matrix[user]).squeeze()
        
        if len(test_items) == 0:
            continue
        
        top_k_items = predictions[user].topk(k).indices
        
        hit_count = len(torch.tensor(list(set(top_k_items.cpu().numpy()) & set(test_items.cpu().numpy()))))
        recall = hit_count / min(len(test_items), k)
        recalls.append(recall)
    
    return torch.mean(torch.tensor(recalls)).item() if recalls else 0

def get_timestamp_filename(prefix: str = '', extension: str = '.txt') -> str:
    """
    현재 날짜와 시간을 포함하는 파일 이름을 생성합니다.
    
    Args:
        prefix (str): 파일 이름 앞에 붙일 접두사 (기본값: '').
        extension (str): 파일 확장자 (기본값: '.txt').

    Returns:
        str: 생성된 파일 이름.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{extension}"
