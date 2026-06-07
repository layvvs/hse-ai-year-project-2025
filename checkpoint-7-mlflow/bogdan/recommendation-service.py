from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "check6"))


import pickle
from contextlib import asynccontextmanager
from functools import lru_cache


import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from scipy.sparse import load_npz
from models import ELSA


RERANKER_PATH = ROOT / "mlflow-logging/reranker-1.pkl"
ELSA_PATH = ROOT / "mlflow-logging/elsa_training_item_filters_user_filters.pkl"
MATRIX_PATH = ROOT / "data/user_item_matrix_item_filters_user_filters.npz"
MAPPINGS_PATH = ROOT / "data/mappings_item_filters_user_filters.pkl"
ENTROPY_PATH = ROOT / "data/users-entropy.csv"
TRAIN_CSV = ROOT / "mlflow-logging/reranker-train.csv"
TEST_CSV = ROOT / "mlflow-logging/reranker-test.csv"

CANDIDATE_K = 100
DEFAULT_K = 10

USER_FEATURE_COLS = [
    "user_n_events", "n_unique_items", "user_n_likes", "n_listens", "user_like_rate",
    "user_avg_played_ratio", "user_organic_ratio", "entropy", "entropy_norm", "n_items",
    "matrix_n_positive", "matrix_total_weight",
]
ITEM_FEATURE_COLS = [
    "item_n_events", "n_unique_users", "item_n_likes", "item_like_rate",
    "item_avg_played_ratio", "avg_track_length", "item_organic_ratio", "matrix_popularity",
]
UI_FEATURE_COLS = [
    "ui_n_events", "mean_played_ratio", "has_like", "days_before_test",
    "organic_ratio_ui", "interaction_weight",
]


class RecommendationItem(BaseModel):
    item_id: int
    score: float
    rank: int


class RecommendationResponse(BaseModel):
    uid: int
    k: int
    recommendations: list[RecommendationItem]


class RecommendationRequest(BaseModel):
    uid: int
    k: int = Field(default=DEFAULT_K, ge=1, le=CANDIDATE_K)


class FeatureStore:
    def __init__(
        self,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        ui_features: pd.DataFrame,
    ):
        self.user_features = user_features.set_index("uid")
        self.item_features = item_features.set_index("item_id")
        self.ui_features = ui_features.set_index(["uid", "item_id"])

    @classmethod
    def from_artifacts(cls, matrix, mappings: dict) -> "FeatureStore":
        user_features, item_features, ui_features = _bootstrap_features_from_csv()
        user_features = _extend_user_features(user_features, matrix, mappings)
        item_features = _extend_item_features(item_features, matrix, mappings)
        ui_features = _extend_ui_features(ui_features, matrix, mappings)
        return cls(user_features, item_features, ui_features)

    def build_candidates_frame(self, uid: int, item_ids: list[int]) -> pd.DataFrame:
        candidates = pd.DataFrame({"uid": uid, "item_id": item_ids})
        candidates["elsa_rank"] = range(len(item_ids))

        candidates = candidates.merge(
            self.user_features.reset_index(), on="uid", how="left"
        )
        candidates = candidates.merge(
            self.item_features.reset_index(), on="item_id", how="left"
        )
        candidates = candidates.merge(
            self.ui_features.reset_index(), on=["uid", "item_id"], how="left"
        )
        return _fill_ui_defaults(candidates)


class RecommendationEngine:
    def __init__(
        self,
        elsa: ELSA,
        reranker_model,
        feature_cols: list[str],
        best_iteration: int,
        user_item_matrix,
        user2id: dict,
        id2item: dict,
        feature_store: FeatureStore,
    ):
        self.elsa = elsa
        self.reranker_model = reranker_model
        self.feature_cols = feature_cols
        self.best_iteration = best_iteration
        self.user_item_matrix = user_item_matrix
        self.user2id = user2id
        self.id2item = id2item
        self.feature_store = feature_store

    def recommend(self, uid: int, k: int = DEFAULT_K) -> list[RecommendationItem]:
        if uid not in self.user2id:
            raise KeyError(f"uid={uid} не найден в mappings")

        item_ids = self.elsa.predict(
            uid, self.user_item_matrix, self.user2id, self.id2item, k=CANDIDATE_K
        )
        if not item_ids:
            return []

        candidates = self.feature_store.build_candidates_frame(uid, item_ids)
        scores = self.reranker_model.predict(
            candidates[self.feature_cols],
            num_iteration=self.best_iteration,
        )
        ranked = (
            candidates.assign(score=scores)
            .sort_values("score", ascending=False)
            .head(k)
        )

        return [
            RecommendationItem(item_id=int(row.item_id), score=float(row.score), rank=i + 1)
            for i, row in enumerate(ranked.itertuples(index=False))
        ]


def _fill_ui_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["has_like"] = out["has_like"].fillna(0).astype(int)
    out["interaction_weight"] = out["interaction_weight"].fillna(0.0)
    out["ui_n_events"] = out["ui_n_events"].fillna(0)
    out["days_before_test"] = out["days_before_test"].fillna(-1)
    return out


def _bootstrap_features_from_csv() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    for path in (TRAIN_CSV, TEST_CSV):
        if path.exists():
            parts.append(pd.read_csv(path, index_col=0))

    if not parts:
        return (
            pd.DataFrame(columns=["uid", *USER_FEATURE_COLS]),
            pd.DataFrame(columns=["item_id", *ITEM_FEATURE_COLS]),
            pd.DataFrame(columns=["uid", "item_id", *UI_FEATURE_COLS]),
        )

    df = pd.concat(parts, ignore_index=True)
    user_features = df[["uid", *USER_FEATURE_COLS]].drop_duplicates("uid")
    item_features = df[["item_id", *ITEM_FEATURE_COLS]].drop_duplicates("item_id")
    ui_features = df[["uid", "item_id", *UI_FEATURE_COLS]].drop_duplicates(["uid", "item_id"])
    return user_features, item_features, ui_features


def _extend_user_features(
    user_features: pd.DataFrame,
    matrix,
    mappings: dict,
) -> pd.DataFrame:
    id2user = mappings["id2user"]
    matrix_n_positive = (matrix > 0).sum(axis=1).A1
    matrix_total_weight = matrix.sum(axis=1).A1

    matrix_users = pd.DataFrame(
        {
            "uid": [id2user[i] for i in range(matrix.shape[0])],
            "matrix_n_positive": matrix_n_positive,
            "matrix_total_weight": matrix_total_weight,
        }
    )

    if ENTROPY_PATH.exists():
        entropy = pd.read_csv(ENTROPY_PATH, index_col=0)
        matrix_users = matrix_users.merge(
            entropy[["uid", "entropy", "entropy_norm", "n_items"]],
            on="uid",
            how="left",
        )

    if user_features.empty:
        return matrix_users

    merged = matrix_users.merge(user_features, on="uid", how="left", suffixes=("_matrix", "_csv"))
    for col in USER_FEATURE_COLS:
        csv_col = f"{col}_csv"
        matrix_col = f"{col}_matrix"
        if csv_col in merged.columns:
            if matrix_col in merged.columns:
                merged[col] = merged[csv_col].combine_first(merged[matrix_col])
            else:
                merged[col] = merged[csv_col]
        elif matrix_col in merged.columns:
            merged[col] = merged[matrix_col]
    keep = ["uid"] + [c for c in USER_FEATURE_COLS if c in merged.columns]
    return merged[keep]


def _extend_item_features(
    item_features: pd.DataFrame,
    matrix,
    mappings: dict,
) -> pd.DataFrame:
    id2item = mappings["id2item"]
    matrix_popularity = matrix.sum(axis=0).A1

    matrix_items = pd.DataFrame(
        {
            "item_id": [id2item[i] for i in range(matrix.shape[1])],
            "matrix_popularity": matrix_popularity,
        }
    )

    if item_features.empty:
        return matrix_items

    merged = matrix_items.merge(item_features, on="item_id", how="left", suffixes=("_matrix", "_csv"))
    for col in ITEM_FEATURE_COLS:
        csv_col = f"{col}_csv"
        matrix_col = f"{col}_matrix"
        if csv_col in merged.columns:
            if matrix_col in merged.columns:
                merged[col] = merged[csv_col].combine_first(merged[matrix_col])
            else:
                merged[col] = merged[csv_col]
        elif matrix_col in merged.columns:
            merged[col] = merged[matrix_col]
    keep = ["item_id"] + [c for c in ITEM_FEATURE_COLS if c in merged.columns]
    return merged[keep]


def _extend_ui_features(
    ui_features: pd.DataFrame,
    matrix,
    mappings: dict,
) -> pd.DataFrame:
    id2user = mappings["id2user"]
    id2item = mappings["id2item"]
    coo = matrix.tocoo()
    matrix_ui = pd.DataFrame(
        {
            "uid": [id2user[r] for r in coo.row],
            "item_id": [id2item[c] for c in coo.col],
            "interaction_weight": coo.data,
        }
    )

    if ui_features.empty:
        return matrix_ui

    merged = matrix_ui.merge(
        ui_features, on=["uid", "item_id"], how="outer", suffixes=("_matrix", "_csv")
    )
    for col in UI_FEATURE_COLS:
        if col == "interaction_weight":
            merged[col] = merged["interaction_weight_matrix"].combine_first(
                merged.get("interaction_weight_csv")
            )
            continue
        csv_col = f"{col}_csv"
        matrix_col = f"{col}_matrix"
        if csv_col in merged.columns:
            merged[col] = merged[csv_col]
        elif matrix_col in merged.columns:
            merged[col] = merged[matrix_col]
    keep = ["uid", "item_id"] + [c for c in UI_FEATURE_COLS if c in merged.columns]
    return merged[keep]


def _load_engine() -> RecommendationEngine:
    with open(RERANKER_PATH, "rb") as f:
        reranker = pickle.load(f)
    with open(ELSA_PATH, "rb") as f:
        elsa = pickle.load(f)
    with open(MAPPINGS_PATH, "rb") as f:
        mappings = pickle.load(f)

    matrix = load_npz(MATRIX_PATH)
    feature_store = FeatureStore.from_artifacts(matrix, mappings)

    return RecommendationEngine(
        elsa=elsa,
        reranker_model=reranker["model"],
        feature_cols=reranker["feature_cols"],
        best_iteration=reranker["model"].best_iteration,
        user_item_matrix=matrix,
        user2id=mappings["user2id"],
        id2item=mappings["id2item"],
        feature_store=feature_store,
    )


@lru_cache(maxsize=1)
def get_engine() -> RecommendationEngine:
    return _load_engine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_engine()
    yield


app = FastAPI(
    title="Music Recommendation Service",
    description="ELSA retrieval + LightGBM reranker",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "known_users": len(engine.user2id),
        "reranker_best_iteration": engine.best_iteration,
        "candidate_k": CANDIDATE_K,
    }


@app.get("/recommendations/{uid}", response_model=RecommendationResponse)
def recommend_get(
    uid: int,
    k: int = Query(default=DEFAULT_K, ge=1, le=CANDIDATE_K),
):
    return _recommend(uid, k)


@app.post("/recommendations", response_model=RecommendationResponse)
def recommend_post(body: RecommendationRequest):
    return _recommend(body.uid, body.k)


def _recommend(uid: int, k: int) -> RecommendationResponse:
    engine = get_engine()
    try:
        items = engine.recommend(uid, k=k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return RecommendationResponse(uid=uid, k=k, recommendations=items)
