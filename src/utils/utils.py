import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from collections import defaultdict


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def convert_sp_mat_to_sp_tensor(X):
    # matrix -> tensor
    coo = X.tocoo().astype(np.float32)
    i = torch.LongTensor(np.mat([coo.row, coo.col]))
    v = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(i, v, coo.shape).to("cuda")


def generate_submission_file(dataset, preds):
    submission = []
    for user,items in enumerate(preds):
        for item in items:
            submission.append(
                {   
                    'user' : dataset.user_decoder[user],
                    'item' : dataset.item_decoder[item],
                }
            )

    submission = pd.DataFrame(submission)
    return submission



def get_ndcg(pred_list, true_list):
    idcg = sum((1 / np.log2(rank + 2) for rank in range(1, len(pred_list))))
    dcg = 0
    for rank, pred in enumerate(pred_list):
        if pred in true_list:
            dcg += 1 / np.log2(rank + 2)
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

def get_hit(pred_list, true_list):
    hit = 0
    hit_list = set(true_list) & set(pred_list)
    if len(hit_list) > 0:
        hit = 1
    return hit

def get_recall(pred_list, true_list):
    recall_list = set(true_list) & set(pred_list)
    recall = len(recall_list) / len(true_list)
    return recall