class Config:
    def __init__(self):
        self.data_path = '../../data/train/'
        self.model_save_path = 'saved_models/'
        self.log_dir = 'logs/'
        self.final_prediction_path = 'predictions/'
        self.lr = 1e-3
        self.wd = 0.00
        self.batch_size = 500
        self.epochs = 100
        self.total_anneal_steps = 200000
        self.anneal_cap = 0.2
        self.seed = 42
        self.log_interval = 100
        
        # VAE architecture
        self.encoder_dims = [200]  # 인코더 레이어의 차원
        self.latent_dim = 16      # 잠재 공간의 차원
        self.dropout_prob = 0.5
        
        # n_users and n_items will be set dynamically during data loading
        self.n_users = None
        self.n_items = None
        
        self.user_id_mapping_path = '../../data/train/user_id_mapping.csv'