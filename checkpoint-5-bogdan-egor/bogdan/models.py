import os

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _to_csr(user_item_matrix):
    if hasattr(user_item_matrix, "tocsr"):
        return user_item_matrix.tocsr()
    return csr_matrix(user_item_matrix)


class ItemKNN:
    def __init__(self):
        self.similarity_matrix = None

    def fit(self, user_item_matrix):
        self.similarity_matrix = cosine_similarity(user_item_matrix.T)
        np.fill_diagonal(self.similarity_matrix, 0)

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10):
        u = user2id[user_id]

        user_vector = user_item_matrix[u]
        scores = user_vector.dot(self.similarity_matrix)
        scores = np.asarray(scores).ravel()

        seen = user_vector.nonzero()[1]
        scores[seen] = -1

        top_k_idx = np.argsort(scores)[-k:][::-1]

        return [id2item[i] for i in top_k_idx]


class UserKNN:
    def __init__(self):
        self.similarity_matrix = None

    def fit(self, user_item_matrix):
        self.similarity_matrix = cosine_similarity(user_item_matrix)
        np.fill_diagonal(self.similarity_matrix, 0)

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10): 
        u = user2id[user_id]
        sim_vec = self.similarity_matrix[u]
        scores = np.asarray(sim_vec @ user_item_matrix).ravel()

        seen = user_item_matrix[u].nonzero()[1]
        scores[seen] = -1

        top_k_idx = np.argsort(scores)[-k:][::-1]

        return [id2item[i] for i in top_k_idx]


class TopPopular:
    def __init__(self):
        self._item_order = None

    def fit(self, user_item_matrix):
        R = _to_csr(user_item_matrix)
        col_pop = np.array(R.sum(axis=0)).ravel()
        self._item_order = np.argsort(-col_pop)

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10):
        if self._item_order is None:
            raise RuntimeError("fit algorithm!")

        u = user2id[user_id]
        seen = set(user_item_matrix[u].nonzero()[1])
        out = []
        for idx in self._item_order:
            if int(idx) in seen:
                continue
            out.append(id2item[int(idx)])
            if len(out) >= k:
                break
        return out


class TopPersonal(UserKNN):
    ...


class TopPersonalTopPopular:
    def __init__(self, personal=None, personal_fraction=0.5):
        self.personal = personal if personal is not None else TopPersonal()
        self.personal_fraction = personal_fraction
        self.popular = TopPopular()

    def clear(self):
        if hasattr(self.personal, "clear"):
            self.personal.clear()
        if hasattr(self.popular, "clear"):
            self.popular.clear()

    def fit(self, user_item_matrix):
        self.personal.fit(user_item_matrix)
        self.popular.fit(user_item_matrix)

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10):
        u = user2id[user_id]
        seen_ids = {id2item[int(j)] for j in user_item_matrix[u].nonzero()[1]}

        k_pers = max(1, int(round(k * self.personal_fraction)))
        k_pop = max(0, k - k_pers)

        n_cand = min(max(k * 5, k + 20), user_item_matrix.shape[1])
        pers = self.personal.predict(
            user_id, user_item_matrix, user2id, id2item, k=n_cand
        )
        pop = self.popular.predict(
            user_id, user_item_matrix, user2id, id2item, k=n_cand
        )

        out = []
        used = set()

        def take(candidates, target_len):
            for item in candidates:
                if len(out) >= target_len:
                    break
                if item in seen_ids or item in used:
                    continue
                used.add(item)
                out.append(item)

        if k_pop == 0:
            take(pers, k)
        else:
            take(pers, k_pers)
            take(pop, k)

        if len(out) < k:
            take(pop, k)
        if len(out) < k:
            take(pers, k)

        return out[:k]


def _positive_only_csr(R):
    coo = R.tocoo()
    mask = coo.data > 0
    out = csr_matrix(
        (coo.data[mask].astype(np.float32), (coo.row[mask], coo.col[mask])),
        shape=R.shape,
    )
    out.eliminate_zeros()
    return out


def _user_row_ravel(user_item_matrix, u):
    row = user_item_matrix[u]
    if hasattr(row, "toarray"):
        return row.toarray().ravel()
    return np.asarray(row).ravel()


class ELSA:

    def __init__(self, top_k=5000, lam=250.0):
        self.top_k = top_k
        self.lam = lam
        self._B = None
        self._ease_item_cols = None

    def fit(self, user_item_matrix):
        R = _to_csr(user_item_matrix)
        R_pos = _positive_only_csr(R)

        col_pop = np.array(R_pos.sum(axis=0)).ravel()
        n_items = R_pos.shape[1]
        k_ease = min(self.top_k, n_items)
        top_part = np.argpartition(-col_pop, k_ease - 1)[:k_ease]
        ease_cols = top_part[np.argsort(-col_pop[top_part])].astype(np.int64)

        X_bin = (R_pos[:, ease_cols].toarray() > 0).astype(np.float64)
        G = X_bin.T @ X_bin
        B = np.linalg.solve(G + self.lam * np.eye(G.shape[0]), G)
        np.fill_diagonal(B, 0.0)

        self._B = B
        self._ease_item_cols = ease_cols

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10):
        assert self._B is not None and self._ease_item_cols is not None, "fit algorithm!"

        if user_id not in user2id:
            return []

        u = user2id[user_id]
        row = _user_row_ravel(user_item_matrix, u)
        ease_cols = self._ease_item_cols
        x = row[ease_cols].astype(np.float64)
        x_bin = (x > 0).astype(np.float64)
        scores = x_bin @ self._B
        seen = np.flatnonzero(x_bin)
        scores[seen] = -1.0
        n_take = min(k, scores.shape[0])
        top_local = np.argsort(scores)[-n_take:][::-1]
        return [id2item[int(ease_cols[j])] for j in top_local]


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

        R = _to_csr(user_item_matrix)
        R_pos = _positive_only_csr(R)

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            num_threads=self.num_threads,
        )
        self._model.fit(R_pos)

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10):
        assert self._model is not None, "fit algorithm!"

        if user_id not in user2id:
            return []

        u = user2id[user_id]
        R = _to_csr(user_item_matrix)
        row = R[u]
        ids, _ = self._model.recommend(
            u, row, N=k, filter_already_liked_items=True
        )
        return [id2item[int(i)] for i in ids]
