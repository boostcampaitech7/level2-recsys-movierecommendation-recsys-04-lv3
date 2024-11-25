import yaml

class Config:
    def __init__(self, config_file='config.yaml'):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.data_path = config['data_path']
        self.user_id_mapping_path = config['user_id_mapping_path']
        self.model_save_path = config['model_save_path']
        self.log_dir = config['log_dir']
        self.final_prediction_path = config['final_prediction_path']
        self.lr = float(config['lr'])  # 타입을 float로 변환
        self.wd = float(config['wd'])  # 타입을 float로 변환
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.total_anneal_steps = config['total_anneal_steps']
        self.anneal_cap = config['anneal_cap']
        self.seed = config['seed']
        self.log_interval = config['log_interval']
        
        # VAE architecture
        self.encoder_dims = config['vae_architecture']['encoder_dims']
        self.latent_dim = config['vae_architecture']['latent_dim']
        self.dropout_prob = config['vae_architecture']['dropout_prob']
        
        # n_users and n_items will be set dynamically during data loading
        self.n_users = config['n_users']
        self.n_items = config['n_items']
