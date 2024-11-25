import time
import torch
import torch.optim as optim
import pandas as pd
from config import Config
from data_loader import get_data_loader
from model import MultiVAE
from trainer import Trainer

def main():
    start_time = time.time()
    
    print('1. Initialize configuration')
    config = Config()

    print('2. Set device')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'   device: {device}')
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print('3. Get data loader and dataset info')
    train_loader, val_loader, n_users, n_items = get_data_loader(config)
    
    # 원본 데이터셋 가져오기 (추천 생성용)
    full_dataset = train_loader.dataset.dataset  # Subset의 원본 데이터셋
    
    # 모델 아키텍처 설정
    encoder_dims = [n_items] + config.encoder_dims + [config.latent_dim * 2]
    decoder_dims = [config.latent_dim] + config.encoder_dims[::-1] + [n_items]

    print('4. Initialize model')
    model = MultiVAE(encoder_dims, decoder_dims, dropout=config.dropout_prob).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)

    print('5. Initialize trainer')
    trainer = Trainer(model, optimizer, config, device)

    print('6. Training')
    best_loss = float('inf')
    early_stop = False
    
    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss = trainer.train_one_epoch(train_loader, epoch)
        print(f'Epoch {epoch} Training Loss: {train_loss:.4f}')
        
        # Validate
        val_loss = trainer.evaluate(val_loader)
        print(f'Epoch {epoch} Validation Loss: {val_loss:.4f}')
        
        # Save regular checkpoint
        trainer.save_checkpoint(epoch, val_loss)
        
        # Check if this is the best model
        if val_loss < best_loss:
            best_loss = val_loss
            trainer.save_checkpoint(epoch, val_loss, best=True)
            print(f'New best model saved! Loss: {val_loss:.4f}')
        
        # Early stopping
        trainer.early_stopping(val_loss)
        if trainer.early_stopping.early_stop:
            print("Early stopping triggered")
            early_stop = True
            break

    print('7. Generate final recommendations')
    # 베스트 모델 불러오기
    checkpoint = torch.load(f'{config.model_save_path}/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    recommendations = trainer.generate_recommendations(full_dataset)  # 원본 데이터셋 사용
    recommendations.to_csv(
        f'{config.final_prediction_path}/recommendations.csv',
        index=False
    )

    end_time = time.time() 
    total_time = end_time - start_time
    print('Done!')
    print(f'Total time taken: {total_time:.2f} seconds')
    
    if early_stop:
        print(f"Training stopped early at epoch {epoch}")
    print(f"Best validation loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()