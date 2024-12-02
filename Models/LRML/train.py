from dataset import PairwiseDataset
from tqdm import tqdm
from lrml import LRML
import torch
import numpy as np
import logging

def evaluate_model(model: LRML, test_data: PairwiseDataset, train_matrix, k=10, device='cuda'):
    """
    주어진 LRML 모델을 recall at k를 사용하여 평가합니다.
    Args:
        model (LRML): 평가할 LRML 모델.
        test_data (PairwiseDataset): 사용자-아이템 상호작용을 포함하는 테스트 데이터셋.
        train_matrix (scipy.sparse.csr_matrix): 각 사용자에 대해 학습에 사용된 아이템을 나타내는 학습 행렬.
        k (int, optional): recall 계산을 위해 고려할 상위 아이템의 수. 기본값은 10.
        device (str, optional): 연산을 수행할 장치 ('cuda' 또는 'cpu'). 기본값은 'cuda'.
    Returns:
        float: 테스트 데이터셋의 모든 사용자에 대한 평균 recall at k.
    """
    
    model.eval()
    # Prepare users and items for scoring
    all_users = torch.arange(test_data.num_users).to(device)
    all_items = torch.arange(test_data.num_items).to(device)
    
    # Create a meshgrid of all user-item combinations
    user_tensor, item_tensor = torch.meshgrid(all_users, all_items, indexing='ij')
    user_tensor = user_tensor.flatten().to(device)
    item_tensor = item_tensor.flatten().to(device)
    
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
    all_scores = all_scores.reshape(test_data.num_users, test_data.num_items)
    
    recalls = []
    
    for user in tqdm(range(test_data.num_users), desc='cal recall'):
        user_true_interaction = test_data.user_positive[user]
        user_pred_scores = all_scores[user]
        
        # 이미 train에 쓰인 아이템은 제거
        user_pred_scores[train_matrix[user].toarray()[0] == 1] = -float('inf')
        top_k_set = set(np.argsort(user_pred_scores)[::-1][:k])
        recall = len(top_k_set & user_true_interaction) / min(k, len(user_true_interaction))
        recalls.append(recall)

    return np.mean(recalls)


def train_model(model: LRML, train_loader, test_dataset, num_epochs, device, learning_rate=0.001):
    """
    LRML 모델을 주어진 학습 데이터로 학습하고 테스트 데이터셋에서 평가합니다.
    Args:
        model (LRML): 학습할 LRML 모델.
        train_loader (DataLoader): 학습 데이터에 대한 DataLoader.
        test_dataset (Dataset): 평가를 위한 데이터셋.
        num_epochs (int): 모델을 학습할 에포크 수.
        device (torch.device): 모델을 실행할 장치 (예: 'cpu' 또는 'cuda').
        learning_rate (float, optional): 옵티마이저의 학습률. 기본값은 0.001.
    Returns:
        None
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.008, steps_per_epoch=1, epochs=num_epochs, pct_start=0.2)
    model = model.to(device)
    
    # 학습 데이터의 상호작용 행렬 생성
    train_matrix = train_loader.dataset.create_interaction_matrix()
    best_recall = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}') as pbar:
            for step, batch in enumerate(pbar):
                users = batch['user'].to(device)
                items = batch['item'].to(device)
                neg_users = batch['neg_user'].to(device)
                neg_items = batch['neg_item'].to(device)
                optimizer.zero_grad()
                loss = model.training_step(users, items, neg_users, neg_items)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{total_loss / (step + 1):.8f}'})
                
        scheduler.step()
            
        # Evaluation
        recall = evaluate_model(model, test_dataset, train_matrix, k=10, device=device)
        
        # Save best model
        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), 'LRML_model.pth')
             
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Recall@10: {recall:.4f}, Average Loss: {avg_loss:.8f}')
        logging.info(f'Epoch {epoch + 1}, Recall@10: {recall:.4f}, Average Loss: {avg_loss:.8f}')
