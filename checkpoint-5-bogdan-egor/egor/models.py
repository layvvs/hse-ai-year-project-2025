import numpy as np
from scipy.sparse import csr_matrix


def _to_csr(user_item_matrix):
    if hasattr(user_item_matrix, "tocsr"):
        return user_item_matrix.tocsr()
    return csr_matrix(user_item_matrix)


def _positive_only_csr(r):
    r = _to_csr(r).copy().astype("float32")
    r.data = np.clip(r.data, 0.0, None)
    r.eliminate_zeros()
    return r

# взял у Богдана
class iALS:
    def __init__(
        self,
        factors=64,
        regularization=0.08,
        iterations=25,
        random_state=42,
        num_threads=0,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.num_threads = num_threads
        self._model = None

    def fit(self, user_item_matrix):
        from implicit.als import AlternatingLeastSquares

        r = _positive_only_csr(user_item_matrix)
        r.data = np.ones_like(r.data, dtype="float32")

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            num_threads=self.num_threads,
        )
        self._model.fit(r)

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10):
        assert self._model is not None, "fit algorithm!"

        if user_id not in user2id:
            return []

        u = user2id[user_id]
        r = _positive_only_csr(user_item_matrix)
        row = r[u]

        ids, _ = self._model.recommend(
            u,
            row,
            N=k,
            filter_already_liked_items=True,
        )
        return [id2item[int(i)] for i in ids]


class WeightedALS:
    def __init__(
        self,
        factors=64,
        regularization=0.08,
        iterations=25,
        alpha=40.0,
        random_state=42,
        num_threads=0,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        self.num_threads = num_threads
        self._model = None

    def fit(self, user_item_matrix):
        from implicit.als import AlternatingLeastSquares

        r = _positive_only_csr(user_item_matrix)

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            num_threads=self.num_threads,
        )
        self._model.fit(self.alpha * r)

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10):
        assert self._model is not None, "fit algorithm!"

        if user_id not in user2id:
            return []

        u = user2id[user_id]
        r = _positive_only_csr(user_item_matrix)
        row = r[u]

        ids, _ = self._model.recommend(
            u,
            row,
            N=k,
            filter_already_liked_items=True,
        )
        return [id2item[int(i)] for i in ids]


class BPR:
    def __init__(
        self,
        factors=64,
        regularization=0.01,
        iterations=100,
        learning_rate=0.05,
        random_state=42,
        num_threads=0,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.num_threads = num_threads
        self._model = None

    def fit(self, user_item_matrix):
        from implicit.bpr import BayesianPersonalizedRanking

        r = _positive_only_csr(user_item_matrix)

        self._model = BayesianPersonalizedRanking(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            num_threads=self.num_threads,
        )
        self._model.fit(r)

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10):
        assert self._model is not None, "fit algorithm!"

        if user_id not in user2id:
            return []

        u = user2id[user_id]
        r = _positive_only_csr(user_item_matrix)
        row = r[u]

        ids, _ = self._model.recommend(
            u,
            row,
            N=k,
            filter_already_liked_items=True,
        )
        return [id2item[int(i)] for i in ids]