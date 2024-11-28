# main.py
import torch
from torch.utils.data import DataLoader
import os
import time

from MultiVAE.config import Config
from MultiVAE.utils import setup_logging, load_rating_csv_to_interaction_matrix, stratified_split, recall_at_k, get_timestamp_filename
from MultiVAE.early_stopping import EarlyStopping
from MultiVAE.dataset import MultiVAEDataset
from MultiVAE.model import MultiVAE
from MultiVAE.train import train_multiVAE, generate_recommendations

def main():
    # 전체 시작 시간 기록
    total_start_time = time.time()

    # 설정 로드
    config = Config('config.yaml')   
    
    # 로깅 설정
    log_filename = get_timestamp_filename(prefix='training_log', extension='.txt')
    logger = setup_logging(config.log_dir, log_filename)
    
    logger.info("===== Config Parameters =====")
    if hasattr(config, '_config') and isinstance(config._config, dict):
        config_table = "\n".join([f"{key:30}: {value}" for key, value in config._config.items()])
    else:
        config_table = "\n".join([f"{key:30}: {value}" for key, value in vars(config).items()])
    logger.info(config_table)
    logger.info("=============================\n")
    
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 데이터 로드 시작 시간
    data_load_start = time.time()
    
    # 데이터 로드
    interaction_matrix, user_id_map, item_id_map = load_rating_csv_to_interaction_matrix(config.rating_file_path, device)
    
    data_load_end = time.time()
    data_load_time = data_load_end - data_load_start
    logger.info(f"데이터 로드 시간: {data_load_time:.2f}초")
    
    logger.info(f"상호작용 행렬 shape: {interaction_matrix.shape}")
    logger.info(f"총 유저 수: {len(user_id_map)}")
    logger.info(f"총 아이템 수: {len(item_id_map)}")
    
    # 훈련/테스트 데이터 전략적 분할 시작 시간
    split_start = time.time()
    
    # 훈련/테스트 데이터 전략적 분할
    train_matrix, test_matrix = stratified_split(
        interaction_matrix, 
        test_size=config.test_size, 
        seed=config.seed
    )
    
    split_end = time.time()
    split_time = split_end - split_start
    logger.info(f"데이터 분할 시간: {split_time:.2f}초")
    
    dataset = MultiVAEDataset(train_matrix, train_matrix, test_matrix)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # 모델 및 옵티마이저 초기화
    num_items = interaction_matrix.shape[1]
    model = MultiVAE(
        num_items, 
        hidden_dim=config.hidden_dim, 
        latent_dim=config.latent_dim, 
        dropout_prob=config.dropout_prob
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Early Stopping 초기화
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience, 
        min_delta=config.early_stopping_delta, 
        mode='max'
    )
    
    # 모델 훈련 시작 시간
    train_start = time.time()
    
    # 모델 훈련 (로거, early_stopping 전달)
    train_multiVAE(model, train_loader, test_matrix, optimizer, config, device, logger, early_stopping)
    
    train_end = time.time()
    train_time = train_end - train_start
    logger.info(f"모델 훈련 시간: {train_time:.2f}초")
    
    # 최종 모델 로드
    checkpoint = torch.load(os.path.join(config.model_save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 최종 평가 시작 시간
    eval_start = time.time()
    
    # 최종 평가
    model.eval()
    with torch.no_grad():
        predictions, _, _ = model(train_matrix)
        
        final_recall = recall_at_k(predictions, test_matrix, k=config.recall_k)
        logger.info(f"Final Test Recall@{config.recall_k}: {final_recall:.4f}")
    
    eval_end = time.time()
    eval_time = eval_end - eval_start
    logger.info(f"모델 평가 시간: {eval_time:.2f}초")
    
    # 추천 생성 시작 시간
    rec_start = time.time()
    
    # 최종 추천 생성
    generate_recommendations(model, train_matrix, user_id_map, item_id_map, config, device)
    
    rec_end = time.time()
    rec_time = rec_end - rec_start
    logger.info(f"추천 생성 시간: {rec_time:.2f}초")
    
    # 총 실행 시간 계산
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    logger.info(f"총 실행 시간: {total_execution_time:.2f}초")

if __name__ == "__main__":
    main()