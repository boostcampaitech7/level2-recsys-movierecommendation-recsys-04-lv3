# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiVAE(nn.Module):
    def __init__(self, num_items, hidden_dim=600, latent_dim=200, dropout_prob=0.5):
        super(MultiVAE, self).__init__()
        
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_items)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            return mu + epsilon * std
        else:
            return mu

    def forward(self, x):
        # 이미 GPU에 있는 입력을 가정
        x = F.normalize(x, dim=1)
        
        # Encode
        h = self.encoder(x)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar