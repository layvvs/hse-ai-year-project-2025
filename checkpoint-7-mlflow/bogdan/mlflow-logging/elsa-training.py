import pickle
import mlflow
from scipy.sparse import load_npz
from dotenv import load_dotenv
import os
from metrics import recall_at_k, precision_at_k, ndcg_at_k, map_at_k, hit_rate_at_k
from models import ELSA


load_dotenv()


mlflow.set_tracking_uri(os.getenv('MLFLOW_URI'))
mlflow.set_experiment("elsa_training")

with open('../data/mappings_item_filters_user_filters.pkl', 'rb') as f:
    loaded_mappings_user_filters = pickle.load(f)

user2id = loaded_mappings_user_filters['user2id']
id2item = loaded_mappings_user_filters['id2item']

with open('../data/test_true_item_filters_user_filters.pkl', 'rb') as f:
    test_true = pickle.load(f)

user_item_weighted_sparse = load_npz('../data/user_item_matrix_item_filters_user_filters.npz')


class ELSAWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        uids = model_input["uid"].values
        return [self.model.predict(uid) for uid in uids]


with mlflow.start_run(run_name="elsa_training_item_filters_user_filters"):

    elsa_weighted = ELSA()
    elsa_weighted.fit(user_item_weighted_sparse)

    mlflow.log_params({'top_k':5000, 'lam': 250})

    common_kwargs = dict(
        test_true=test_true,
        user_item_matrix=user_item_weighted_sparse,
        user2id=user2id,
        id2item=id2item,
        recommend_fn=elsa_weighted.predict,
    )

    metrics_10 = {
        "recall_at_10":    recall_at_k(**common_kwargs, k=10),
        "precision_at_10": precision_at_k(**common_kwargs, k=10),
        "ndcg_at_10":      ndcg_at_k(**common_kwargs, k=10),
        "map_at_10":       map_at_k(**common_kwargs, k=10),
        "hit_rate_at_10":  hit_rate_at_k(**common_kwargs, k=10),
    }
    mlflow.log_metrics(metrics_10)

    metrics_100 = {
        "recall_at_100":    recall_at_k(**common_kwargs, k=100),
        "precision_at_100": precision_at_k(**common_kwargs, k=100),
        "ndcg_at_100":      ndcg_at_k(**common_kwargs, k=100),
        "map_at_100":       map_at_k(**common_kwargs, k=100),
        "hit_rate_at_100":  hit_rate_at_k(**common_kwargs, k=100),
    }
    mlflow.log_metrics(metrics_100)

    with open("elsa_training_item_filters_user_filters.pkl", "wb") as f:
        pickle.dump(elsa_weighted, f)
    mlflow.log_artifact("elsa_training_item_filters_user_filters.pkl")
