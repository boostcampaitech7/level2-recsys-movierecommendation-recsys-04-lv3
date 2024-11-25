import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiVAE(nn.Module):
    def __init__(self, encoder_dims, decoder_dims, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.latent_dim = decoder_dims[0]

        # 인코더 레이어 생성
        self.encoder_layers = nn.ModuleList([
            nn.Linear(d_in, d_out) for d_in, d_out in zip(encoder_dims[:-1], encoder_dims[1:])
        ])
        
        # 디코더 레이어 생성
        self.decoder_layers = nn.ModuleList([
            nn.Linear(d_in, d_out) for d_in, d_out in zip(decoder_dims[:-1], decoder_dims[1:])
        ])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def forward(self, input_data):
        mu, logvar = self.encode(input_data)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input_data):
        h = F.normalize(input_data, p=2, dim=1)
        h = self.drop(h)
        
        # 인코더를 통과
        for i, layer in enumerate(self.encoder_layers[:-1]):
            h = layer(h)
            h = F.tanh(h)
            h = self.drop(h)
        
        # 마지막 레이어에서 mu와 logvar 생성
        h = self.encoder_layers[-1](h)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        
        # 디코더를 통과
        for i, layer in enumerate(self.decoder_layers[:-1]):
            h = layer(h)
            h = F.tanh(h)
            h = self.drop(h)
        
        # 마지막 레이어
        return self.decoder_layers[-1](h)

    def init_weights(self):
        for layer in self.encoder_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.bias, std=0.001)
        
        for layer in self.decoder_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.bias, std=0.001)
