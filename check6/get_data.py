from __future__ import annotations

import numpy as np
import pandas as pd
from datasets import load_from_disk
from scipy.sparse import csr_matrix

EVENTS = {"like", "dislike", "unlike", "undislike", "listen"}
LISTEN_MIN = 0.5

W = {
    "like": 1.0,
    "dislike": 0.1,
    "unlike": -0.7,
    "undislike": 0.2,
}


def load_df(path: str) -> pd.DataFrame:
    ds = load_from_disk(path)
    try:
        df = ds["train"].to_pandas()
    except Exception:
        df = ds.to_pandas()
    return df


def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["event_type"] = df["event_type"]
    df["uid"] = df["uid"]
    df["item_id"] = df["item_id"]

    df = df[df["event_type"].isin(EVENTS)].copy()

    x = df["played_ratio_pct"].astype(float) / 100.0

    df["played_ratio"] = x.clip(upper=1.0)

    m = (df["event_type"] == "listen") & (df["played_ratio"] >= LISTEN_MIN)
    df = df[(df["event_type"] != "listen") | m].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def filt_users(df: pd.DataFrame, q1: float = 0.7, q2: float = 0.95) -> pd.DataFrame:
    s = df.groupby("uid").size()
    low = s.quantile(q1)
    high = s.quantile(q2)

    ids = s[(s >= low) & (s <= high)].index
    res = df[df["uid"].isin(ids)].copy().reset_index(drop=True)
    return res


def split_df(df: pd.DataFrame, test_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    t_max = df["timestamp"].max()
    border = t_max - test_days * 24 * 60 * 60

    train_df = df[df["timestamp"] < border].copy().reset_index(drop=True)
    test_df = df[df["timestamp"] >= border].copy().reset_index(drop=True)
    return train_df, test_df


def add_w(
        df: pd.DataFrame,
        organic_k: float = 0.8,
        decay_days: float = 90.0,
        listen_k: float = 0.9,
) -> pd.DataFrame:
    df = df.copy()

    df["base_w"] = df["event_type"].map(W).fillna(0.0)

    m = df["event_type"] == "listen"
    df.loc[m, "base_w"] = df.loc[m, "played_ratio"] * listen_k

    df["org_k"] = np.where(df["is_organic"] == 1, 1.0, organic_k)

    t_max = df["timestamp"].max()
    age_days = (t_max - df["timestamp"]) / (24 * 60 * 60)
    df["time_k"] = 1.0 / (1.0 + age_days / decay_days)

    df["w"] = df["base_w"] * df["org_k"] * df["time_k"]
    return df


def make_tbl(df: pd.DataFrame) -> pd.DataFrame:
    res = (
        df.groupby(["uid", "item_id"], as_index=False)
        .agg(
            w=("w", "sum"),
            ts=("timestamp", "max"),
            n=("item_id", "size"),
        )
        .reset_index(drop=True)
    )

    res["w"] = res["w"].clip(lower=0.0)
    res["w"] = np.log1p(res["w"])
    return res


def make_map(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    users = df["uid"].drop_duplicates().tolist()
    items = df["item_id"].drop_duplicates().tolist()

    user2id = {x: i for i, x in enumerate(users)}
    item2id = {x: i for i, x in enumerate(items)}
    id2item = {i: x for x, i in item2id.items()}

    return user2id, item2id, id2item


def make_mat(df: pd.DataFrame, user2id: dict, item2id: dict) -> csr_matrix:
    rows = df["uid"].map(user2id).to_numpy()
    cols = df["item_id"].map(item2id).to_numpy()
    vals = df["w"].astype("float32").to_numpy()

    shape = (len(user2id), len(item2id))
    res = csr_matrix((vals, (rows, cols)), shape=shape, dtype=np.float32)
    return res


def make_true(test_df: pd.DataFrame) -> dict:
    df = test_df[test_df["event_type"].isin({"like", "listen"})].copy()

    x = df["played_ratio_pct"].astype(float) / 100.0

    df["played_ratio"] = x.clip(upper=1.0)

    m = (df["event_type"] == "listen") & (df["played_ratio"] >= LISTEN_MIN)
    df = df[(df["event_type"] != "listen") | m].copy()

    res = df.groupby("uid")["item_id"].apply(set).to_dict()
    return res


def prepare(
        path: str,
        test_days: int = 14,
        q1: float = 0.7,
        q2: float = 0.95,
        organic_k: float = 0.8,
        decay_days: float = 90.0,
        listen_k: float = 0.9,
) -> dict:
    df = load_df(path)
    df = prep_df(df)
    df = filt_users(df, q1=q1, q2=q2)

    train_df, test_df = split_df(df, test_days=test_days)
    train_df = add_w(
        train_df,
        organic_k=organic_k,
        decay_days=decay_days,
        listen_k=listen_k,
    )

    inter_df = make_tbl(train_df)
    user2id, item2id, id2item = make_map(inter_df)
    user_item_matrix = make_mat(inter_df, user2id, item2id)
    test_true = make_true(test_df)

    res = {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "inter_df": inter_df,
        "user_item_matrix": user_item_matrix,
        "user2id": user2id,
        "item2id": item2id,
        "id2item": id2item,
        "test_true": test_true,
    }
    return res
