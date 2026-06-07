import os
import pickle

import lightgbm
import mlflow
import mlflow.lightgbm
import pandas as pd
from dotenv import load_dotenv

import optuna
import optuna.integration.mlflow as optuna_mlflow

from metrics import map_at_k, ndcg_at_k, precision_at_k, recall_at_k


load_dotenv()


mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))
mlflow.set_experiment("reranker_training_optuna_v2")

ID_COLS = {"uid", "item_id", "label"}
MODEL_PATH = "lightgbm_reranker_optuna.txt"

train_data = pd.read_csv("reranker-train.csv", index_col=0)
train_data["has_like"] = train_data["has_like"].fillna(0).astype(int)

test_data = pd.read_csv("reranker-test.csv", index_col=0)
test_data["has_like"] = test_data["has_like"].fillna(0).astype(int)

feature_cols = [c for c in train_data.columns if c not in ID_COLS]

train_data = train_data.sort_values("uid").reset_index(drop=True)
test_data = test_data.sort_values("uid").reset_index(drop=True)

train_groups = train_data.groupby("uid", sort=False).size().tolist()
test_groups = test_data.groupby("uid", sort=False).size().tolist()

train_set = lightgbm.Dataset(train_data[feature_cols], label=train_data["label"], group=train_groups)
test_set = lightgbm.Dataset(test_data[feature_cols], label=test_data["label"], group=test_groups, reference=train_set)

test_true = (
    test_data[test_data["label"] == 1]
    .groupby("uid")["item_id"]
    .apply(set)
    .to_dict()
)

KS = [10, 20, 50, 100]


def make_recommend_fn(model, df):
    grouped = {uid: grp for uid, grp in df.groupby("uid", sort=False)}

    def recommend_fn(uid, _matrix, _user2id, _id2item, k=10):
        cands = grouped.get(uid)
        if cands is None or cands.empty:
            return []
        scores = model.predict(cands[feature_cols], num_iteration=model.best_iteration)
        return cands.assign(score=scores).sort_values("score", ascending=False)["item_id"].head(k).tolist()

    return recommend_fn


def log_ranking_metrics(model, df):
    recommend_fn = make_recommend_fn(model, df)
    common = dict(test_true=test_true, user_item_matrix=None, user2id={}, id2item={}, recommend_fn=recommend_fn)
    metrics = {}
    for k in KS:
        metrics[f"recall_at_{k}"] = recall_at_k(**common, k=k)
        metrics[f"precision_at_{k}"] = precision_at_k(**common, k=k)
        metrics[f"ndcg_at_{k}"] = ndcg_at_k(**common, k=k)
        metrics[f"map_at_{k}"] = map_at_k(**common, k=k)
    mlflow.log_metrics(metrics)
    return metrics


def objective(trial):
    with mlflow.start_run(run_name=f'trial_{trial.number}'):
        params = {
            "objective": "lambdarank",
            "metric": ["ndcg", "map"],
            "ndcg_at": [10, 20, 50, 100],
            "map_at": [10, 20, 50, 100],
            "verbose": -1,
            "seed": 42,
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }


        model = lightgbm.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[train_set, test_set],
            valid_names=["train", "test"],
            callbacks=[
                lightgbm.early_stopping(50, verbose=False),
            ],
        )

        # for split in ["train", "test"]:
        #     for metric_key, values in evals_result[split].items():
        #         mlflow_name = f"{split}_{metric_key.replace('@', '_at_')}"
        #         for step, value in enumerate(values):
        #             mlflow.log_metric(mlflow_name, value, step=step)

        best_score = model.best_score["test"]["ndcg@10"]
        # mlflow.log_metric("best_ndcg_at_10", best_score)
        # mlflow.log_metric("best_iteration", model.best_iteration)

        log_ranking_metrics(model, test_data)

    return best_score


mlflow_callback = optuna_mlflow.MLflowCallback(
    tracking_uri=os.getenv("MLFLOW_URI"),
    metric_name="ndcg_at_10",
    create_experiment=False,
    mlflow_kwargs={"experiment_name": "reranker_training_optuna_v2"},
)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)#, callbacks=[mlflow_callback])

print(f"\nbest trial: {study.best_trial.number}")
print(f"best ndcg@10: {study.best_value:.6f}")
print(f"best params: {study.best_params}")

best_params = {
    "objective": "lambdarank",
    "metric": ["ndcg", "map"],
    "ndcg_eval_at": [10, 20, 50, 100],
    "map_eval_at": [10, 20, 50, 100],
    "verbose": -1,
    "seed": 42,
    **study.best_params,
}

with mlflow.start_run(run_name="best_trial_final_model"):
    mlflow.log_params(best_params)

    evals_result = {}

    best_model = lightgbm.train(
        best_params,
        train_set,
        num_boost_round=500,
        valid_sets=[train_set, test_set],
        valid_names=["train", "test"],
        callbacks=[
            lightgbm.early_stopping(50, verbose=False),
            lightgbm.log_evaluation(period=10),
            lightgbm.record_evaluation(evals_result),
        ],
    )

    for split in ["train", "test"]:
        for metric_key, values in evals_result[split].items():
            mlflow_name = f"{split}_{metric_key.replace('@', '_at_')}"
            for step, value in enumerate(values):
                mlflow.log_metric(mlflow_name, value, step=step)

    best_model.save_model(MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
    mlflow.log_metric("best_iteration", best_model.best_iteration)
    metrics = log_ranking_metrics(best_model, test_data)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    with open("reranker_feature_cols_optuna.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    mlflow.log_artifact("reranker_feature_cols_optuna.pkl")
