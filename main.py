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
    print(f'device: {device}')

    print('3. Get data loader and dataset info')
    data_loader, n_users, n_items = get_data_loader(config)
    
    # 모델 아키텍처 설정
    # 인코더: input_dim -> hidden_dims -> latent_dim*2
    encoder_dims = [n_items] + config.encoder_dims + [config.latent_dim * 2]
    # 디코더: latent_dim -> hidden_dims (reverse) -> output_dim
    decoder_dims = [config.latent_dim] + config.encoder_dims[::-1] + [n_items]

    print('4. Initialize model')
    model = MultiVAE(encoder_dims, decoder_dims, dropout=config.dropout_prob).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)

    print('5. Initialize trainer')
    trainer = Trainer(model, optimizer, config, device)

    print('6. Training')
    best_loss = float('inf')
    for epoch in range(1, config.epochs + 1):
        trainer.train_one_epoch(data_loader, epoch)
        
        # Save regular checkpoint
        trainer.save_checkpoint(epoch)
        
        # Calculate loss for entire dataset
        total_loss = 0
        model.eval()
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss = trainer.loss_function(recon_batch, data, mu, logvar)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch} Average Loss: {avg_loss}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            trainer.save_checkpoint(epoch, best=True)

    print('7. Generate final recommendations')
    recommendations = trainer.generate_recommendations(data_loader.dataset)
    recommendations.to_csv(
        f'{config.final_prediction_path}/recommendations.csv',
        index=False
    )

    end_time = time.time() 
    total_time = end_time - start_time
    print('Done!')
    print(f'Total time taken: {total_time:.2f} seconds')

if __name__ == "__main__":
    main()