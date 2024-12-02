import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import CDAE_train
from CDAE_dataset import MakeMatrixDataSet, MovieDataSet
from CDAE import CDAE
from mlflow_setup import MlflowManager
from utils import load_config

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)


def main():
    ml = MlflowManager(user_name="lagom", experiment_name="CDAE")

    with ml.start_run(run_name=None):

        CONFIG_PATH = "/data/ephemeral/home/level2-recsys-movierecommendation-recsys-04-lv3/config/CDAE.yaml"

        config = load_config(CONFIG_PATH)
        params = config["params"]
        print(config)

        ml.log_params(params)  # Save MLflow log params

        make_matrix_data_set = MakeMatrixDataSet(config=config)
        user_train, user_valid = make_matrix_data_set.get_train_valid_data()
        num_items, num_users = make_matrix_data_set.get_item_user_num()

        movie_dataset = MovieDataSet(
            num_user=make_matrix_data_set.num_user,
        )
        movie_dataloader = DataLoader(
            movie_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=params["num_workers"],
        )

        model = CDAE(
            num_users=num_users,
            num_items=num_items,
            num_hidden_units=params["num_hidden_units"],
            corruption_ratio=params["corruption_ratio"],
        )
        model.to(device)
        print(model)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        max_recall = 0
        print("\nTrain...")
        for epoch in tqdm(range(1, params["num_epochs"] + 1)):

            train_loss = CDAE_train.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                data_loader=movie_dataloader,
                make_matrix_data_set=make_matrix_data_set,
            )

            val_loss, ndcg, recall = CDAE_train.evaluate(
                model=model,
                criterion=criterion,
                data_loader=movie_dataloader,
                user_train=user_train,
                user_valid=user_valid,
                make_matrix_data_set=make_matrix_data_set,
            )

            # scheduler.step()

            print(f"Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| Val loss: {val_loss:.5f}")
            print(f"Epoch: {epoch:3d}| NDCG@10: {ndcg:.5f}| RECALL@10: {recall:.5f}")

            # Save
            if max_recall < recall:
                max_recall = recall
                torch.save(model, config["saved_path"])

            # Save MLflow log metric
            ml.log_metric("train_loss", train_loss, step=epoch)
            ml.log_metric("val_loss", val_loss, step=epoch)
            ml.log_metric("ndcg", ndcg, step=epoch)
            ml.log_metric("recall", recall, step=epoch)

            ml.log_metric("LR", optimizer.param_groups[0]["lr"], step=epoch)

            ml.log_model(model, "CDAE", type="torch")

    # Predict
    print("\nPredict...")
    df = CDAE_train.predict(config)
    print(df.head())


if __name__ == "__main__":
    main()
