import torch
import numpy as np
from torch.utils.data import Dataset

class MultiVAEDataset(Dataset):
    """
    MultiVAE 모델의 사용자-아이템 행렬과 마스크 정보를 다루는 Dataset 클래스입니다.
    """

    def __init__(self, 
                 user_item_matrix: torch.Tensor, 
                 train_mask: torch.Tensor, 
                 test_mask: torch.Tensor) -> None:
        """
        Dataset을 초기화하는 함수입니다.
        
        Args:
            user_item_matrix (torch.Tensor): 사용자-아이템 상호작용 행렬로, 각 행은 사용자에 해당하며 열은 아이템에 해당합니다.
            train_mask (torch.Tensor): 훈련 데이터에 대한 마스크로, 1은 훈련 데이터이고 0은 아닌 데이터입니다.
            test_mask (torch.Tensor): 테스트 데이터에 대한 마스크로, 1은 테스트 데이터이고 0은 아닌 데이터입니다.
        """
        self.user_item_matrix: torch.Tensor = user_item_matrix  # 사용자-아이템 상호작용 행렬
        self.train_mask: torch.Tensor = train_mask  # 훈련 마스크
        self.test_mask: torch.Tensor = test_mask  # 테스트 마스크

    def __len__(self) -> int:
        """
        Dataset의 길이를 반환합니다. 여기서는 사용자 수에 해당합니다.
        
        Returns:
            int: 데이터셋의 크기 (사용자 수)
        """
        return len(self.user_item_matrix)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        주어진 인덱스에 해당하는 사용자-아이템 상호작용 행렬과 훈련/테스트 마스크를 반환합니다.
        
        Args:
            idx (int): 데이터셋에서 가져올 인덱스
        
        Returns:
            tuple: (사용자-아이템 행렬, 훈련 마스크, 테스트 마스크)
                각 항목은 torch.Tensor 형태입니다.
        """
        return (
            self.user_item_matrix[idx], 
            self.train_mask[idx], 
            self.test_mask[idx]
        )
