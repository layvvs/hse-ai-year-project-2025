import os
import pickle

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import GroupShuffleSplit


from metrics import (
    hit_rate_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))
mlflow.set_experiment("reranker_training")

ID_COLS = {"uid", "item_id", "label"}
FEATURE_COLS_PATH = "reranker_feature_cols.pkl"
MODEL_PATH = "lightgbm_reranker.txt"


LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10, 20, 50, 100],
    "learning_rate": 0.01186277223185042,
    'lambda_l1': 2.2731670711510562e-06,
    'lambda_l2': 0.0003365582024289719,
    "num_leaves": 51,
    "min_child_samples": 53,
    "feature_fraction": 0.997794,
    "bagging_fraction": 0.642,
    "bagging_freq": 10,
    "verbose": -1,
    "seed": 42,
}

NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 50


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df["has_like"] = df["has_like"].fillna(0).astype(int)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS]


def prepare_ranking_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    df_sorted = df.sort_values("uid").reset_index(drop=True)
    groups = df_sorted.groupby("uid", sort=False).size().tolist()
    return df_sorted, groups


def build_test_true(df: pd.DataFrame) -> dict:
    return (
        df[df["label"] == 1]
        .groupby("uid")["item_id"]
        .apply(set)
        .to_dict()
    )


def make_rerank_recommend_fn(model, candidates_df: pd.DataFrame, feature_cols: list[str]):
    grouped = {
        uid: group
        for uid, group in candidates_df.groupby("uid", sort=False)
    }

    def recommend(uid, user_item_matrix, user2id, id2item, k=10):
        user_cands = grouped.get(uid)
        if user_cands is None or user_cands.empty:
            return []

        scores = model.predict(user_cands[feature_cols], num_iteration=model.best_iteration)
        ranked = user_cands.assign(score=scores).sort_values("score", ascending=False)
        return ranked["item_id"].head(k).tolist()

    return recommend


train_df = load_dataset("reranker-train.csv")
raw_test_df = load_dataset("reranker-test.csv")

gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, holdout_idx = next(gss.split(raw_test_df, groups=raw_test_df["uid"]))
val_df = raw_test_df.iloc[val_idx].copy()
holdout_df = raw_test_df.iloc[holdout_idx].copy()

feature_cols = get_feature_cols(train_df)

train_df, train_groups = prepare_ranking_data(train_df)
val_df, val_groups = prepare_ranking_data(val_df)
holdout_df, holdout_groups = prepare_ranking_data(holdout_df)

X_train = train_df[feature_cols]
y_train = train_df["label"]
X_val = val_df[feature_cols]
y_val = val_df["label"]

holdout_true = build_test_true(holdout_df)

with mlflow.start_run(run_name="lightgbm_reranker_item_filters_user_filters"):
    mlflow.lightgbm.autolog(
        log_models=False,
        log_input_examples=False,
        log_datasets=False,
    )

    mlflow.log_params(
        {
            "model_type": "lgbm_ranker",
            "objective": "lambdarank",
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "holdout_rows": len(holdout_df),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "holdout_groups": len(holdout_groups),
            "n_features": len(feature_cols),
            "num_boost_round": NUM_BOOST_ROUND,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        }
    )

    train_set = lgb.Dataset(X_train, label=y_train, group=train_groups)
    val_set = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_set)

    model = lgb.train(
        LGBM_PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=10),
        ],
    )

    recommend_fn = make_rerank_recommend_fn(model, holdout_df, feature_cols)

    common_kwargs = dict(
        test_true=holdout_true,
        user_item_matrix=None,
        user2id={},
        id2item={},
        recommend_fn=recommend_fn,
    )

    metrics_10 = {
        "recall_at_10": recall_at_k(**common_kwargs, k=10),
        "precision_at_10": precision_at_k(**common_kwargs, k=10),
        "ndcg_at_10": ndcg_at_k(**common_kwargs, k=10),
        "map_at_10": map_at_k(**common_kwargs, k=10),
        "hit_rate_at_10": hit_rate_at_k(**common_kwargs, k=10),
    }
    mlflow.log_metrics(metrics_10)

    metrics_100 = {
        "recall_at_100": recall_at_k(**common_kwargs, k=100),
        "precision_at_100": precision_at_k(**common_kwargs, k=100),
        "ndcg_at_100": ndcg_at_k(**common_kwargs, k=100),
        "map_at_100": map_at_k(**common_kwargs, k=100),
        "hit_rate_at_100": hit_rate_at_k(**common_kwargs, k=100),
    }
    mlflow.log_metrics(metrics_100)

    model.save_model(MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)

    with open(FEATURE_COLS_PATH, "wb") as f:
        pickle.dump(feature_cols, f)
    mlflow.log_artifact(FEATURE_COLS_PATH)

    print(f"best iteration: {model.best_iteration}")
    for name, value in {**metrics_10, **metrics_100}.items():
        print(f"{name}: {value:.6f}")
