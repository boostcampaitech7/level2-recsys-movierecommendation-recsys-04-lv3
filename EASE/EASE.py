import numpy as np
from box import Box
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, Any

class EASE:
    """ 
    추천 시스템을 위한 EASE(Embarrassingly Shallow Autoencoder) 모델을 정의한다.
    
    Attributes:
        reg: 정규화 가중치 (float)
        device: 계산에 사용되는 장치 (torch.device)
        B: 학습된 추천 가중치 행렬 (torch.Tensor)
    
    Methods:
        fit: 모델을 데이터에 맞춰 학습
        forward: 입력 행렬에 대한 추천 점수 계산
        predict: 데이터셋에 대한 추천 리스트 생성
    """
    def __init__(self, args: Box):
        self.reg = args.reg_weight
        self.device = args.device
    
    def fit(self, X: torch.Tensor) -> None:
        """ 
        추천 모델의 가중치를 학습한다.
        
        Args:
            X: 사용자-아이템 상호작용 행렬 (torch.Tensor)
        """
        G = X.T @ X
        diagIndices = torch.eye(G.shape[0]) == 1
        G[diagIndices] += self.reg

        P = G.inverse()
        B = P / (-1 * P.diag())
        B[diagIndices] = 0
        
        self.B = B

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ 
        입력 행렬에 대한 추천 점수를 계산한다.
        
        Args:
            X: 입력 행렬 (torch.Tensor)
        
        Returns:
            추천 점수 행렬 (torch.Tensor)
        """
        output = X @ self.B
        return output
    

    def predict(self, dataset: Dataset) -> Dict[int, list]:
        """ 
        데이터셋에 대한 사용자별 추천 리스트를 생성한다.
        
        Args:
            dataset: 추천을 수행할 데이터셋
        
        Returns:
            사용자별 추천 아이템 리스트 (Dict[int, list])
        """
        X = dataset.sparse_matrix.todense()
        user2rec = {}

        recon_mat = self.pred.cpu()
        score = recon_mat * torch.from_numpy(1 - X)
        rec_list = score.argsort(dim = 1)

        for user, rec in enumerate(rec_list):
            up = rec[-10:].cpu().numpy().tolist()
            user2rec[user] = up
        
        return user2rec

class MultiEASE(nn.Module):
    """ 
    다중 채널 추천을 위한 MultiEASE 모델을 정의한다.
    
    Attributes:
        reg: 정규화 가중치 (float)
        device: 계산에 사용되는 장치 (torch.device)
        B: 학습 가능한 추천 가중치 행렬 (nn.Parameter)
        diag_mask: 대각선 요소 제외를 위한 마스크 (torch.Tensor)
    
    Methods:
        forward: 입력 행렬에 대한 로그 소프트맥스 추천 점수 계산
        calculate_loss: 모델의 손실 함수 계산
    """
    def __init__(self, args: Box, num_items: int):
        super(MultiEASE, self).__init__()
        self.reg = args.reg_weight
        self.device = args.device
        self.B = nn.Parameter(torch.zeros(num_items, num_items))    
        
        diag_mask = 1 - torch.eye(num_items)
        self.register_buffer("diag_mask", diag_mask)
       

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ 
        입력 행렬에 대한 로그 소프트맥스 추천 점수를 계산한다.
        
        Args:
            X: 입력 행렬 (torch.Tensor)
        
        Returns:
            로그 소프트맥스 변환된 추천 점수 행렬 (torch.Tensor)
        """
        X.to(self.device)
        output = X @ self.B
        output = F.log_softmax(output, dim=-1)

        return output
    
    def calculate_loss(self, X: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """ 
        모델의 손실 함수를 계산한다.
        
        Args:
            X: 원본 상호작용 행렬 (torch.Tensor)
            output: 모델의 출력 행렬 (torch.Tensor)
        
        Returns:
            계산된 손실 값 (기본 손실 + 정규화 손실) (torch.Tensor)
        """
        loss = -(output * X).sum(dim=-1).mean()    
        reg_loss = self.reg * torch.sum(self.B ** 2 * self.diag_mask)
        
        return loss + reg_loss