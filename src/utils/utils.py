import numpy as np
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


def get_ndcg(pred_list, true_list):
    idcg = sum((1 / np.log2(rank + 2) for rank in range(1, len(pred_list))))
    dcg = 0
    for rank, pred in enumerate(pred_list):
        if pred in true_list:
            dcg += 1 / np.log2(rank + 2)
    ndcg = dcg / idcg

    return ndcg


def recall_at_10(pred_list, true_list):
    pred_list_top_10 = pred_list[:10]
    true_positive = len(set(pred_list_top_10) & set(true_list))
    actual_true = len(true_list)
    recall = true_positive / actual_true if actual_true > 0 else 0

    return recall
