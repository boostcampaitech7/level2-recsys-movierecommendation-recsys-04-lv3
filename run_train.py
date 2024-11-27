import argparse
import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MovieLensDataset
from src.models.EASE import EASE, MultiEASE
from src.utils.utils import check_path, set_seed, generate_submission_file
from trainer import EASETrainer, MultiEASETrainer, full_sort_predict
from logging import getLogger


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data/train/", type=str)
    parser.add_argument("--output_dir", default="./saved/submission/", type=str)
    parser.add_argument("--model_dir", default="./saved/model/", type=str)

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

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")

    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--early_stopping", type=int, default=10, help="round of early stopping")

    

    args = parser.parse_args()
    logger = getLogger()

    set_seed(args.seed)
    check_path(args.model_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and args.cuda
    torch.cuda.empty_cache()

    # save model args
    args_str = f"{args.model}"
    args.log_file = os.path.join(args.model_dir, args_str + ".txt")
    print(str(args))

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.model_dir, checkpoint)

    print('----------------Train test split-------------------')
    train_dataset = MovieLensDataset(args, data_type="train")

    num_items = train_dataset.n_items
    num_users = train_dataset.n_users
    user_valid = train_dataset.user_valid
    
    batch_size = args.batch_size if args.model != 'ease' else num_users
    shuffle = True if args.model != 'ease' else False

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size, 
        shuffle = shuffle,
        pin_memory = True,
        num_workers = args.num_workers,
    ) # ease에서는 안쓰임

    valid_dataset = MovieLensDataset(args, data_type="valid")
    valid_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size, 
        shuffle = False,
        pin_memory = True,
        num_workers = args.num_workers,
    ) # ease에서는 안쓰임

    print('----------------Train model-------------------')
    if args.model == 'ease':
        model = EASE(args)
        trainer = EASETrainer(model, args)
        args.epochs = 1
    elif args.model == 'multiease':
        model = MultiEASE(args, num_items).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        trainer = MultiEASETrainer(model, optimizer, args)
        
    # Training
    best_recall = 0
    early_stop = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train(train_dataloader)
        ndcg, hit, recall = trainer.evaluate(train_dataloader, user_valid)

        print(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}| RECALL@10: {recall:.5f}')

        if best_recall < recall:
            best_recall = recall
            trainer.save_model(args.checkpoint_path, args.model)
            early_stop = 0
        else:
            early_stop += 1
            if early_stop == args.early_stopping:
                print('early stop!')
                break
    

    print('----------------submission file-------------------')   
    pred = full_sort_predict(model, valid_dataloader)
    submission = generate_submission_file(valid_dataset, pred)
    submission.to_csv(os.path.join(args.output_dir, args.model +'_submission_'+ pd.Timestamp.now().strftime("%Y-%m-%d %H:%M") +'.csv'), index=False)

if __name__ == "__main__":
    main()
