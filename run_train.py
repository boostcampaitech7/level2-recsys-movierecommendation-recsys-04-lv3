import argparse
import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MovieLensDataset
from src.models.EASE import EASE
from src.utils.utils import check_path, set_seed, generate_submission_file


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)

    # model args
    parser.add_argument("--model", default="ease", type=str)    
    parser.add_argument("--reg_weight", default=250.0, type=float)    

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--valid_size", default=0.1, type=float)   
    parser.add_argument("--num_workers", default=4, type=int)   

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )

    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and args.cuda

    # save model args
    args_str = f"{args.model}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = MovieLensDataset(args, data_type="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = args.batch_size, 
        shuffle = True,
        pin_memory = True,
        num_workers = args.num_workers,
    ) # ease에서는 안쓰임

    valid_dataset = MovieLensDataset(args, data_type="valid")
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size = args.batch_size, 
        shuffle = True,
        pin_memory = True,
        num_workers = args.num_workers,
    ) # ease에서는 안쓰임

    if args.model == 'ease':
        model = EASE(args)
        model.fit(train_dataset)
        ndcg, hit, recall =  model.evaluate(train_dataset)
        print(f'NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}| RECALL@10: {recall:.5f}')
    
    elif args.model == 'multiease':
        # 구현 예정
        pass

    model = EASE(args)
    model.fit(valid_dataset)
    rec_pred = model.predict(valid_dataset)
    submission = generate_submission_file(valid_dataset, rec_pred)
    submission.to_csv(os.path.join(args.output_dir, args.model +'_submission_'+ pd.Timestamp.now().strftime("%Y-%m-%d %H%M%S") +'.csv'), index=False)

if __name__ == "__main__":
    main()
