import argparse
import logging
import yaml

import pandas as pd
from dataset import PairwiseDataset
from torch.utils.data import DataLoader

from LRML.lrml import LRML
from LRML.train import train_model
from LRML.utils import split_data_sklearn, set_seed
     
     
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,  
        format="%(asctime)s - %(levelname)s - %(message)s",  
        filename="log_LRML.log", 
        filemode="a",
    )
    logging.info("=====start run_lrml.py=====")
    
    # argument parser
    args =  argparse.ArgumentParser()
    args.add_argument("--config", default="config.yaml")
    parsed_args = args.parse_args()
    
    # config file load
    with open(parsed_args.config) as f:
        config = yaml.load(f)    
    
    set_seed(config['seed'])

    # 데이터 로드 및 분할
    data = pd.read_csv(config['data_path'])
    train_data, test_data, user_map, item_map, idx_user_map, idx_item_map = split_data_sklearn(data, config['test_num'])
    
    # Dataset 생성 - 동일한 매핑 전달
    logging.info(f'users: {len(user_map)}, items: {len(item_map)}')
    train_dataset = PairwiseDataset(
        train_data, 
        user_map=user_map, 
        item_map=item_map,
        idx_user_map=idx_user_map,
        idx_item_map=idx_item_map,
        cache_size=config['cache_size']
    )
    test_dataset = PairwiseDataset(
        test_data, 
        user_map=user_map, 
        item_map=item_map,
        idx_user_map=idx_user_map,
        idx_item_map=idx_item_map,
        test=True
    )
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    logging.info("start training")
    # 모델 생성 및 학습
    model = LRML(train_dataset.num_users,
                 train_dataset.num_items,
                 embedding_dim=config['embedding_dim'], 
                 memory_size=config['memory_size'],
                 margin=config['margin'],
                 reg_weight=config['reg_weight']
                 )
    
    train_model(model, train_loader, test_dataset, config['epoch'], config['device'], config['learning_rate'])
    print("Training completed")