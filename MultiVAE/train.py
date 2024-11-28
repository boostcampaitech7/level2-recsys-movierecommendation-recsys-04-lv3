# train.py
import os
import torch
import torch.nn.functional as F
import pandas as pd
from utils import recall_at_k, get_timestamp_filename

def calculate_loss(x_recon, mu, logvar, train_mask, update, total_anneal_steps, anneal_cap):
    # Annealing 계산
    if total_anneal_steps > 0:
        anneal = min(anneal_cap, 1.0 * update / total_anneal_steps)
    else:
        anneal = anneal_cap

    # KL Divergence Loss (Annealing 적용)
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)) * anneal
    
    # Reconstruction Loss (Cross Entropy)
    recon_loss = -(F.log_softmax(x_recon, 1) * train_mask).sum(1).mean()
    
    return recon_loss + kl_loss

def train_multiVAE(model, train_loader, val_matrix, optimizer, config, device, logger, early_stopping):
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
        
        # Best 모델 저장
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_recall': best_recall
            }, os.path.join(config.model_save_dir, 'best_model.pth'))
        
        # Early Stopping 체크
        if early_stopping(val_recall):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return best_recall

def generate_recommendations(model, interaction_matrix, user_id_map, item_id_map, config, device):
    # 출력 디렉토리 확인
    os.makedirs(config.output_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # 이미 GPU에 있는 텐서 사용
        predictions, _, _ = model(interaction_matrix)
        
        # 원래 아이템 ID로 매핑
        reverse_item_map = {idx: item for item, idx in item_id_map.items()}
        
        # 추천 결과 저장할 리스트
        recommendations = []
        for user_idx in range(interaction_matrix.shape[0]):
            # 사용자의 기존 상호작용 아이템 찾기
            interacted_items = torch.nonzero(interaction_matrix[user_idx]).squeeze()
            
            # 원본 user ID 찾기
            user = list(user_id_map.keys())[list(user_id_map.values()).index(user_idx)]
            
            # 모델 예측 결과 가져오기
            user_predictions = predictions[user_idx]
            
            # 기존에 상호작용하지 않은 아이템만 선택
            non_interacted_mask = torch.ones_like(user_predictions, dtype=torch.bool)
            non_interacted_mask[interacted_items] = False
            
            # 필터링된 예측 결과에서 top-k 선택
            filtered_predictions = user_predictions.masked_fill(~non_interacted_mask, float('-inf'))
            top_k_items = filtered_predictions.topk(config.top_k_recommendations).indices
            
            # 각 추천 아이템에 대해 개별 행 생성
            for item_idx in top_k_items:
                rec_item = reverse_item_map[item_idx.item()]
                recommendations.append({
                    'user': user,
                    'item': rec_item
                })
        
        # CSV로 저장
        rec_df = pd.DataFrame(recommendations)
        recommendations_filename = get_timestamp_filename(prefix='user_recommendations', extension='.csv')
        rec_df.to_csv(os.path.join(config.output_dir, recommendations_filename), index=False)
        
        return rec_df