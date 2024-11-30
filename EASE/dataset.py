import os
import pandas as pd
import numpy as np
from box import Box
from scipy import sparse
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import Dict, List, Tuple

class MovieLensDataset(Dataset):
    """MovieLens 데이터셋을 처리하기 위한 PyTorch Dataset 클래스.
    
    사용자-아이템 상호작용 데이터를 로드하고 전처리하며, 훈련 및 검증을 위한 데이터 분할을 수행한다.

    Attributes:
        config (object): 데이터셋 설정 정보를 포함하는 구성 객체
        data_type (str): 현재 데이터셋의 유형 ('train' 또는 'valid')
    
    Methods:
        _generate_sequence_data(): 사용자별 훈련 및 검증 데이터 생성
        make_sparse_matrix(): 사용자-아이템 상호작용 희소 행렬 생성
        __len__(): 데이터셋의 사용자 수 반환
        __getitem__(): 특정 사용자의 아이템 상호작용 데이터 반환
    """

    def __init__(self, config: Box, data_type: str = 'train') -> None:
        """MovieLensDataset 클래스 초기화.

        Args:
            config (object): 데이터셋 구성 설정
            data_type (str, optional): 데이터셋 유형. 기본값은 'train'.
        """
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
        
    
    def _generate_sequence_data(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """사용자별 훈련 및 검증 데이터 생성.

        Returns:
            tuple: (user_train, user_valid) 딕셔너리
            - user_train (dict): 각 사용자의 훈련 데이터
            - user_valid (dict): 각 사용자의 검증 데이터
        """
        users = defaultdict(list)
        user_train, user_valid = {}, {}
        
        for user, item in zip(self.data['user'], self.data['item']):
            users[user].append(item)
        
        for user in users:
            user_total = users[user]
            valid_samples = int(len(user_total) * self.config.valid_size)
            valid = np.random.choice(user_total, size=valid_samples, replace=False).tolist()
            train = list(set(user_total) - set(valid))

            user_train[user] = train
            user_valid[user] = valid

        return user_train, user_valid

    def make_sparse_matrix(self) -> sparse.csr_matrix:
        """사용자-아이템 상호작용 희소 행렬 생성.

        Returns:
            scipy.sparse.csr_matrix: 사용자-아이템 상호작용 희소 행렬
        """
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
    
    def __len__(self) -> int:
        """데이터셋의 사용자 수를 반환.

        Returns:
            int: 사용자 수
        """
        return self.n_users

    def __getitem__(self, index: int) -> torch.FloatTensor:
        """특정 사용자의 아이템 상호작용 데이터 반환.

        Args:
            index (int): 사용자 인덱스

        Returns:
            torch.FloatTensor: 사용자의 아이템 상호작용 데이터
        """
        user_data = self.sparse_matrix[index].toarray().flatten()
        return torch.FloatTensor(user_data)
    