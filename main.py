import time
import torch
import torch.optim as optim
import pandas as pd
import logging
from config import Config
from data_loader import get_data_loader
from model import MultiVAE
from trainer import Trainer

def main():
    start_time = time.time()
    
    log_file = str(start_time)+'.log'

    # 로그 설정
    logging.basicConfig(
        level=logging.INFO,  # 로그 레벨 설정
        format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 포맷 설정
        handlers=[
            logging.FileHandler(log_file),  # 파일로 로그 저장
            logging.StreamHandler()  # 콘솔에도 출력
        ]
    )
    
    logging.info('---------- 1. Initialize configuration')
    config = Config()

    logging.info('---------- 2. Set device')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'----------   device: {device}')
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logging.info('---------- 3. Get data loader and dataset info')
    train_loader, val_loader, n_users, n_items = get_data_loader(config)
    
    full_dataset = train_loader.dataset.dataset
    
    encoder_dims = [n_items] + config.encoder_dims + [config.latent_dim * 2]
    decoder_dims = [config.latent_dim] + config.encoder_dims[::-1] + [n_items]

    logging.info('---------- 4. Initialize model')
    model = MultiVAE(encoder_dims, decoder_dims, dropout=config.dropout_prob).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)

    logging.info('---------- 5. Initialize trainer')
    trainer = Trainer(model, optimizer, config, device)

    logging.info('---------- 6. Training')
    best_loss = float('inf')
    best_recall = 0
    best_ndcg = 0
    early_stop = False
    
    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss, train_recall, train_ndcg = trainer.train_one_epoch(train_loader, epoch)
        logging.info(f'Epoch {epoch}:')
        logging.info(f'Training Loss: {train_loss:.4f}, Recall@10: {train_recall:.4f}, NDCG@10: {train_ndcg:.4f}')
        
        # Validate
        val_loss, val_recall, val_ndcg = trainer.evaluate(val_loader)
        logging.info(f'Validation Loss: {val_loss:.4f}, Recall@10: {val_recall:.4f}, NDCG@10: {val_ndcg:.4f}')
        
        # Save regular checkpoint
        trainer.save_checkpoint(epoch, val_loss)
        
        # Update best metrics
        if val_recall > best_recall:
            best_recall = val_recall
            best_ndcg = val_ndcg
            trainer.save_checkpoint(epoch, val_loss, best=True)
            logging.info(f'New best model saved! Recall@10: {val_recall:.4f}, NDCG@10: {val_ndcg:.4f}')
        
        # Early stopping
        trainer.early_stopping(val_loss)
        if trainer.early_stopping.early_stop:
            logging.info("Early stopping triggered")
            early_stop = True
            break

    logging.info('---------- 7. Generate final recommendations ----------')
    checkpoint = torch.load(f'{config.model_save_path}/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    recommendations = trainer.generate_recommendations(full_dataset)
    recommendations.to_csv(
        f'{config.final_prediction_path}/recommendations.csv',
        index=False
    )

    end_time = time.time() 
    total_time = end_time - start_time
    logging.info('---------- Done! ----------')
    logging.info(f'Total time taken: {total_time:.2f} seconds')
    
    if early_stop:
        logging.info(f"Training stopped early at epoch {epoch}")
    logging.info(f"Best validation metrics - Recall@10: {best_recall:.4f}, NDCG@10: {best_ndcg:.4f}")

if __name__ == "__main__":
    main()