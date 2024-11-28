import os
import logging
import torch
import torch.nn.functional as F
import pandas as pd

from config import Config
from early_stopping import EarlyStopping
from utils import recall_at_k, get_timestamp_filename


def calculate_loss(x_recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, 
                   train_mask: torch.Tensor, update: int, total_anneal_steps: int, 
                   anneal_cap: float) -> torch.Tensor:
    """
    VAE 손실을 계산합니다. 
    복원 손실과 KL 발산 손실을 포함하며, annealing이 적용됩니다.

    Args:
        x_recon (torch.Tensor): 복원된 행렬로, 크기는 (배치 크기, 아이템 수)입니다.
        mu (torch.Tensor): VAE 잠재 공간에서의 평균값, 크기는 (배치 크기, 잠재 차원)입니다.
        logvar (torch.Tensor): VAE 잠재 공간에서의 로그 분산, 크기는 (배치 크기, 잠재 차원)입니다.
        train_mask (torch.Tensor): 입력에서 관측된 항목을 나타내는 마스크로, 크기는 (배치 크기, 아이템 수)입니다.
        update (int): 훈련 단계 수로, annealing 계산에 사용됩니다.
        total_anneal_steps (int): annealing을 적용할 총 단계 수입니다.
        anneal_cap (float): annealing의 최대 값입니다.

    Returns:
        torch.Tensor: 배치에 대한 총 손실(복원 손실 + KL 발산 손실)을 반환합니다.
    """
    if total_anneal_steps > 0:
        anneal = min(anneal_cap, 1.0 * update / total_anneal_steps)
    else:
        anneal = anneal_cap

    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)) * anneal
    recon_loss = -(F.log_softmax(x_recon, 1) * train_mask).sum(1).mean()
    
    return recon_loss + kl_loss

def train_multiVAE(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_matrix: torch.Tensor, 
                   optimizer: torch.optim.Optimizer, config: Config, device: torch.device, logger: logging.Logger, 
                   early_stopping: EarlyStopping) -> float:
    """
    MultiVAE 모델을 훈련시키고 검증 데이터에서 최상의 Recall@k를 얻습니다.

    Args:
        model (torch.nn.Module): 훈련할 VAE 모델입니다.
        train_loader (torch.utils.data.DataLoader): 훈련 데이터 로더입니다.
        val_matrix (torch.Tensor): 검증 데이터 행렬입니다.
        optimizer (torch.optim.Optimizer): 모델의 옵티마이저입니다.
        config (Config): 설정 파일에서 로드한 구성입니다.
        device (torch.device): 훈련에 사용할 장치(CPU 또는 GPU)입니다.
        logger (logging.Logger): 로깅을 위한 로거입니다.
        early_stopping (EarlyStopping): 조기 종료를 위한 객체입니다.

    Returns:
        float: 최상의 Recall@k 값입니다.
    """
    val_matrix_tensor = val_matrix.to(device)
    val_mask_tensor = (val_matrix > 0).float().to(device)
    
    update_count = 0
    best_recall = 0
    
    os.makedirs(config.model_save_dir, exist_ok=True)
    
    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        
        for user_items, train_mask, _ in train_loader:
            user_items = user_items.to(device)
            train_mask = train_mask.float().to(device)
            
            optimizer.zero_grad()
            
            x_recon, mu, logvar = model(user_items)
            
            update_count += 1
            train_loss = calculate_loss(x_recon, mu, logvar, train_mask, 
                                        update_count, config.total_anneal_steps, config.anneal_cap)
            train_loss.backward()
            
            optimizer.step()
            
            total_train_loss += train_loss.item()
        
        model.eval()
        with torch.no_grad():
            val_predictions, val_mu, val_logvar = model(val_matrix_tensor)
            val_loss = calculate_loss(val_predictions, val_mu, val_logvar, val_mask_tensor, 
                                      update_count, config.total_anneal_steps, config.anneal_cap)
            val_recall = recall_at_k(val_predictions, val_matrix, k=config.recall_k)
        
        logger.info(f"Epoch [{epoch+1}/{config.epochs}], "
                    f"Train Loss: {total_train_loss/len(train_loader):.4f}, "
                    f"Validation Loss: {val_loss.item():.4f}, "
                    f"Validation Recall@{config.recall_k}: {val_recall:.4f}")
        
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_recall': best_recall
            }, os.path.join(config.model_save_dir, 'best_model.pth'))
        
        if early_stopping(val_recall):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return best_recall

def generate_recommendations(model: torch.nn.Module, interaction_matrix: torch.Tensor, 
                             user_id_map: dict, item_id_map: dict, config: Config, 
                             device: torch.device) -> pd.DataFrame:
    """
    훈련된 모델을 사용하여 사용자에게 추천을 생성합니다.

    Args:
        model (torch.nn.Module): 훈련된 VAE 모델입니다.
        interaction_matrix (torch.Tensor): 사용자-아이템 상호작용 행렬입니다.
        user_id_map (dict): 사용자 ID를 인덱스로 매핑한 딕셔너리입니다.
        item_id_map (dict): 아이템 ID를 인덱스로 매핑한 딕셔너리입니다.
        config (Config): 설정 파일에서 로드한 구성입니다.
        device (torch.device): 추천을 생성할 때 사용할 장치(CPU 또는 GPU)입니다.

    Returns:
        pd.DataFrame: 추천된 사용자-아이템 쌍의 DataFrame을 반환합니다.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        predictions, _, _ = model(interaction_matrix)
        
        reverse_item_map = {idx: item for item, idx in item_id_map.items()}
        
        recommendations = []
        for user_idx in range(interaction_matrix.shape[0]):
            interacted_items = torch.nonzero(interaction_matrix[user_idx]).squeeze()
            
            user = list(user_id_map.keys())[list(user_id_map.values()).index(user_idx)]
            
            user_predictions = predictions[user_idx]
            
            non_interacted_mask = torch.ones_like(user_predictions, dtype=torch.bool)
            non_interacted_mask[interacted_items] = False
            
            filtered_predictions = user_predictions.masked_fill(~non_interacted_mask, float('-inf'))
            top_k_items = filtered_predictions.topk(config.top_k_recommendations).indices
            
            for item_idx in top_k_items:
                rec_item = reverse_item_map[item_idx.item()]
                recommendations.append({
                    'user': user,
                    'item': rec_item
                })
        
        rec_df = pd.DataFrame(recommendations)
        recommendations_filename = get_timestamp_filename(prefix='user_recommendations', extension='.csv')
        rec_df.to_csv(os.path.join(config.output_dir, recommendations_filename), index=False)
        
        return rec_df
