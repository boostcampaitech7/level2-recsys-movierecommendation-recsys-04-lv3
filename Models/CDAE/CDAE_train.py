import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.CDAE_dataset import MakeMatrixDataSet, MovieDataSet
from src.models.CDAE import CDAE
from src.utils.utils import get_ndcg, recall_at_10

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, criterion, optimizer, data_loader, make_matrix_data_set):
    model.train()
    loss_val = 0
    for users in data_loader:
        mat = make_matrix_data_set.make_matrix(users)
        mat = mat.to(device)
        recon_mat = model(mat, users.view(-1).to(device))

        optimizer.zero_grad()
        loss = criterion(recon_mat, mat)

        loss_val += loss.item()

        loss.backward()
        optimizer.step()

    loss_val /= len(data_loader)

    return loss_val


def evaluate(model, criterion, data_loader, user_train, user_valid, make_matrix_data_set):
    model.eval()

    loss_val = 0.0  # Validation loss
    NDCG = 0.0  # NDCG@10
    RECALL = 0.0  # Recall@10
    with torch.no_grad():
        for users in data_loader:
            mat = make_matrix_data_set.make_matrix(users)
            mat = mat.to(device)

            recon_mat = model(mat, users.view(-1).to(device))

            loss = criterion(recon_mat, mat)
            loss_val += loss.item()

            recon_mat[mat == 1] = -np.inf
            rec_list = recon_mat.argsort(dim=1)

            for user, rec in zip(users, rec_list):
                uv = user_valid[user.item()]
                up = rec[-10:].cpu().numpy().tolist()[::-1]
                NDCG += get_ndcg(pred_list=up, true_list=uv)
                RECALL += recall_at_10(pred_list=up, true_list=uv)

    loss_val /= len(data_loader.dataset)
    NDCG /= len(data_loader.dataset)
    RECALL /= len(data_loader.dataset)

    return loss_val, NDCG, RECALL


def predict(config):
    params = config["params"]

    make_matrix_data_set = MakeMatrixDataSet(config=config)

    item_encoder, item_decoder, user_encoder, user_decoder = make_matrix_data_set.get_encoder_decoder_data()
    num_items, num_users = make_matrix_data_set.get_item_user_num()

    movie_test_dataset = MovieDataSet(
        num_user=make_matrix_data_set.num_user,
    )
    movie_test_dataloader = DataLoader(
        movie_test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=params["num_workers"],
    )

    # Load saved model
    model = CDAE(
        num_users=num_users,
        num_items=num_items,
        num_hidden_units=params["num_hidden_units"],
        corruption_ratio=params["corruption_ratio"],
    )
    print(model)
    model = torch.load(config["saved_path"], weights_only=False)
    model.eval()

    submission_df = pd.DataFrame(columns=["user", "item"])

    with torch.no_grad():
        for users in tqdm(movie_test_dataloader):
            mat = make_matrix_data_set.make_matrix(users, train=False)
            mat = mat.to(device)

            recon_mat = model(mat, users.view(-1).to(device))
            recon_mat[mat == 1] = -np.inf  # 이미 평가한 항목(1로 표시)을 추천에서 제외하기 위해 -무한대로 설정
            rec_list = recon_mat.argsort(dim=1)
            up = rec_list[0][-10:].cpu().numpy().tolist()[::-1]  # 상위 10개 추천 항목을 추출하고 역순으로 정렬

            user = user_decoder[users.item()]
            for item_id in up:
                item = item_decoder[item_id]
                submission_df.loc[submission_df.shape[0]] = [user, item]

    submission_df.to_csv(config["submission_path"], index=False)

    return submission_df
