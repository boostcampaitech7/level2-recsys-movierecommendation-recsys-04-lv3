import pandas as pd
import numpy as np
import torch
import bottleneck as bn

def generate_recommendations(model, train_data, n_items=10):
    model.eval()
    N = train_data.shape[0]
    predictions = []
    
    with torch.no_grad():
        for start_idx in range(0, N, 1000):  # Process in batches to avoid memory issues
            end_idx = min(start_idx + 1000, N)
            data = train_data[start_idx:end_idx]
            data_tensor = naive_sparse2tensor(data).to(next(model.parameters()).device)
            
            recon_batch, _, _ = model(data_tensor)
            recon_batch = recon_batch.cpu().numpy()
            
            # Exclude items that are in the training set
            recon_batch[data.nonzero()] = -np.inf
            
            # Get top N items
            idx = bn.argpartition(-recon_batch, n_items, axis=1)[:, :n_items]
            predictions.extend(idx)
    
    return np.array(predictions)

def save_recommendations(predictions, user_mapping, item_mapping, output_file):
    recommendations = []
    for user_idx, items in enumerate(predictions):
        original_user_id = list(user_mapping.keys())[list(user_mapping.values()).index(user_idx)]
        item_recommendations = [list(item_mapping.keys())[list(item_mapping.values()).index(item_idx)] 
                              for item_idx in items]
        recommendations.append({
            'user': original_user_id,
            'recommendations': item_recommendations
        })
    
    df = pd.DataFrame(recommendations)
    df.to_csv(output_file, index=False)