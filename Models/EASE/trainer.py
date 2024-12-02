import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any, Optional
from box import Box
from utils import get_hit, get_ndcg, get_recall


class EASETrainer:
    def __init__(self, model: Any, args: Box) -> None:
        """ 
        EASE(Embarrassingly Shallow Autoencoder) 모델을 훈련하고 평가하는 트레이너 클래스.
    
        Attributes:
            model: 훈련할 EASE 모델 (Any)
            device: 모델이 실행될 장치 (str)
        
        Methods:
            train: 모델을 데이터로 훈련
            evaluate: 모델의 성능 평가
            save_model: 훈련된 모델 저장
        """
        self.model = model
        self.device = args.device

    def train(self, data_loader: DataLoader) -> float:
        """
        모델 훈련 메서드

        Args:
            data_loader (DataLoader): 훈련 데이터 로더

        Returns:
            float: 손실값
        """
        loss_val = 0.0
        for mat, users in data_loader:
            mat = mat.to(self.device)
            self.model.fit(mat)
        return loss_val

    def evaluate(self, data_loader: DataLoader, user_valid: Dict[int, List[int]], k: int) -> Tuple[float, float, float]:
        """
        모델 성능 평가 메서드

        Args:
            data_loader (DataLoader): 평가 데이터 로더
            user_valid (Dict[int, List[int]]): 검증용 사용자 상호작용 데이터
            k (int): Top-k 추천 개수

        Returns:
            Tuple[float, float, float]: NDCG, HIT, RECALL 점수
        """
        NDCG, HIT, RECALL = 0.0, 0.0, 0.0

        mat = next(iter(data_loader))[0]
        mat = mat.to(self.device)
        
        recon_mat = self.model.forward(mat)
        recon_mat[mat == 1] = -np.inf
        rec_list = recon_mat.argsort(dim = 1)

        user_list = user_valid.keys()
        for user, rec in enumerate(rec_list):
            if user in user_list:
                uv = user_valid[user]
                up = rec[-10:].cpu().numpy().tolist()[::-1]
                NDCG += get_ndcg(pred_list = up, true_list = uv)
                HIT += get_hit(pred_list = up, true_list = uv)
                RECALL += get_recall(pred_list = up, true_list = uv, k = k)

        NDCG /= len(data_loader.dataset)
        HIT /= len(data_loader.dataset)
        RECALL /= len(data_loader.dataset)

        return NDCG, HIT, RECALL

    def save_model(self, path: str, filename: str) -> None:
        """
        모델 저장 메서드

        Args:
            path (str): 모델 저장 경로
            filename (str): 저장할 파일명
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.B, os.path.join(path, filename))



class MultiEASETrainer:
    """ 
    다중 채널 EASE 모델을 훈련하고 평가하는 트레이너 클래스.
    
    Attributes:
        model: 훈련할 다중 EASE 모델 (Any)
        optimizer: 모델 최적화기 (torch.optim.Optimizer)
        device: 모델이 실행될 장치 (str)
    
    Methods:
        train: 모델을 데이터로 훈련
        evaluate: 모델의 성능 평가
        save_model: 훈련된 모델 저장
    """
    def __init__(self, model: Any, optimizer: torch.optim.Optimizer, args: Box) -> None:
        """
        Args:
            model (Any): 훈련할 모델
            optimizer (torch.optim.Optimizer): 모델 최적화기
            args (Box): 모델 설정 및 하이퍼파라미터
        """
        self.model = model
        self.optimizer = optimizer
        self.device = args.device

    def train(self, data_loader: DataLoader) -> float:
        """
        모델 훈련 메서드

        Args:
            data_loader (DataLoader): 훈련 데이터 로더

        Returns:
            float: 평균 손실값
        """
        self.model.train()
        loss_val = 0

        for mat, users in data_loader:
            mat = mat.to(self.device)
            
            output = self.model.forward(mat)
            
            self.optimizer.zero_grad()
            loss = self.model.calculate_loss(mat, output)

            loss_val += loss.item()

            loss.backward()
            self.optimizer.step()

        loss_val /= len(data_loader)
        return loss_val

    def evaluate(self, data_loader: DataLoader, user_valid: Dict[int, List[int]], k: int) -> Tuple[float, float, float]:
        """
        모델 성능 평가 메서드

        Args:
            data_loader (DataLoader): 평가 데이터 로더
            user_valid (Dict[int, List[int]]): 검증용 사용자 상호작용 데이터
            k (int): Top-k 추천 개수

        Returns:
            Tuple[float, float, float]: NDCG, HIT, RECALL 점수
        """
        self.model.eval()
        NDCG, HIT, RECALL = 0.0, 0.0, 0.0

        with torch.no_grad():
            for mat, users in data_loader:
                mat = mat.to(self.device)
                
                output = self.model.forward(mat)
                output[mat == 1] = -np.inf
                rec_list = output.argsort(dim=1)

                for user, rec in zip(users, rec_list):
                    uv = user_valid[user.item()]
                    up = rec[-k:].cpu().numpy().tolist()[::-1]
                    NDCG += get_ndcg(pred_list=up, true_list=uv)
                    HIT += get_hit(pred_list=up, true_list=uv)
                    RECALL += get_recall(pred_list=up, true_list=uv, k=k)

        NDCG /= len(data_loader.dataset)
        HIT /= len(data_loader.dataset)
        RECALL /= len(data_loader.dataset)

        return NDCG, HIT, RECALL

    def save_model(self, path: str, filename: str) -> None:
        """
        모델 저장 메서드

        Args:
            path (str): 모델 저장 경로
            filename (str): 저장할 파일명
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, filename))


def full_sort_predict(model: Any, data_loader: DataLoader, k: int, device: str) -> np.ndarray:
    """
    전체 데이터셋에 대한 추천 예측 수행

    Args:
        model (Any): 추천 모델
        data_loader (DataLoader): 데이터 로더
        k (int): Top-k 추천 개수
        device (str): 모델이 실행될 장치 ('cuda' 또는 'cpu')

    Returns:
        np.ndarray: 각 사용자에 대한 top-k 추천 아이템 배열
    """
    num_users = len(data_loader.dataset)
    preds = np.zeros((num_users, k), dtype=int) 

    for batch, users in data_loader:
        batch = batch.to(device)
        pred = model.forward(batch)

        scores = pred * (1 - batch)
        scores = scores.argsort(dim=1)
        pred = scores[:, -k:] 

        preds[users] = pred.cpu().numpy()
    return preds