import os

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF as SklearnNMF
from sklearn.linear_model import ElasticNet

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
        u = user2id[user_id]
        R = _to_csr(user_item_matrix)
        row = R[u]
        ids, _ = self._model.recommend(
            u, row, N=k, filter_already_liked_items=True
        )
        return [id2item[int(i)] for i in ids]


class NMFModel:
    def __init__(
        self,
        n_components=64,
        init="nndsvda",
        random_state=42,
        max_iter=300,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    ):
        self.n_components = n_components
        self.init = init
        self.random_state = random_state
        self.max_iter = max_iter
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self._model = None
        self._W = None
        self._H = None

    def fit(self, user_item_matrix):
        R = _to_csr(user_item_matrix)
        X = R.toarray().astype(np.float64)
        X = np.maximum(X, 0.0)

        self._model = SklearnNMF(
            n_components=self.n_components,
            init=self.init,
            random_state=self.random_state,
            max_iter=self.max_iter,
            alpha_W=self.alpha_W,
            alpha_H=self.alpha_H,
            l1_ratio=self.l1_ratio,
        )
        self._W = self._model.fit_transform(X)
        self._H = self._model.components_

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10):
        u = user2id[user_id]
        scores = self._W[u] @ self._H

        R = _to_csr(user_item_matrix)
        seen = R[u].nonzero()[1]
        scores[seen] = -1.0

        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [id2item[int(i)] for i in top_k_idx]


class BPRModel:
    def __init__(
        self,
        factors=64,
        learning_rate=0.05,
        regularization=0.01,
        iterations=100,
        random_state=42,
        num_threads=0,
    ):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.num_threads = num_threads
        self._model = None

    def fit(self, user_item_matrix):
        from implicit.bpr import BayesianPersonalizedRanking

        R = _to_csr(user_item_matrix)
        R_pos = _positive_only_csr(R)

        self._model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            num_threads=self.num_threads,
        )
        self._model.fit(R_pos)

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10):
        u = user2id[user_id]
        R = _to_csr(user_item_matrix)
        row = R[u]
        ids, _ = self._model.recommend(
            u, row, N=k, filter_already_liked_items=True
        )
        return [id2item[int(i)] for i in ids]


class PMFModel:
    def __init__(
        self,
        n_factors=64,
        n_epochs=20,
        lr=0.01,
        reg=0.01,
        random_state=42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state
        self.P = None
        self.Q = None
        self.mu = 0.0

    def fit(self, user_item_matrix):
        R = _to_csr(user_item_matrix).astype(np.float64)
        coo = R.tocoo()

        rng = np.random.default_rng(self.random_state)
        n_users, n_items = R.shape

        self.P = 0.1 * rng.standard_normal((n_users, self.n_factors))
        self.Q = 0.1 * rng.standard_normal((n_items, self.n_factors))
        self.mu = float(coo.data.mean()) if coo.nnz > 0 else 0.0

        rows = coo.row
        cols = coo.col
        vals = coo.data

        for _ in range(self.n_epochs):
            order = rng.permutation(len(vals))
            for idx in order:
                u = rows[idx]
                i = cols[idx]
                r = vals[idx]

                pred = self.mu + self.P[u] @ self.Q[i]
                err = r - pred

                pu = self.P[u].copy()
                qi = self.Q[i].copy()

                self.P[u] += self.lr * (err * qi - self.reg * pu)
                self.Q[i] += self.lr * (err * pu - self.reg * qi)

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10):
        u = user2id[user_id]
        scores = self.mu + self.P[u] @ self.Q.T

        R = _to_csr(user_item_matrix)
        seen = R[u].nonzero()[1]
        scores[seen] = -1.0

        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [id2item[int(i)] for i in top_k_idx]


class LightFMModel:
    def __init__(
        self,
        no_components=64,
        loss="bpr",
        learning_rate=0.05,
        item_alpha=0.0,
        user_alpha=0.0,
        epochs=30,
        random_state=42,
        num_threads=1,
    ):
        self.no_components = no_components
        self.loss = loss
        self.learning_rate = learning_rate
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.epochs = epochs
        self.random_state = random_state
        self.num_threads = num_threads
        self._model = None

    def fit(self, user_item_matrix):
        from lightfm import LightFM

        R = _positive_only_csr(_to_csr(user_item_matrix))

        self._model = LightFM(
            no_components=self.no_components,
            loss=self.loss,
            learning_rate=self.learning_rate,
            item_alpha=self.item_alpha,
            user_alpha=self.user_alpha,
            random_state=self.random_state,
        )
        self._model.fit(R, epochs=self.epochs, num_threads=self.num_threads)

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10):
        u = user2id[user_id]
        R = _to_csr(user_item_matrix)
        n_items = R.shape[1]

        scores = self._model.predict(
            user_ids=np.repeat(u, n_items),
            item_ids=np.arange(n_items),
            num_threads=self.num_threads,
        )

        seen = R[u].nonzero()[1]
        scores[seen] = -1.0

        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [id2item[int(i)] for i in top_k_idx]


class SLIMModel:
    def __init__(
        self,
        alpha=0.001,
        l1_ratio=0.01,
        max_iter=100,
        top_k=200,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.top_k = top_k
        self.W = None

    def fit(self, user_item_matrix):
        R = _to_csr(user_item_matrix).astype(np.float64)
        X = R.toarray()
        n_items = X.shape[1]
        W = np.zeros((n_items, n_items), dtype=np.float64)

        for j in range(n_items):
            y = X[:, j].copy()
            xj = X[:, j].copy()
            X[:, j] = 0.0

            m = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=False,
                positive=True,
                max_iter=self.max_iter,
                selection="random",
            )
            m.fit(X, y)
            coef = m.coef_

            if self.top_k is not None and self.top_k < len(coef):
                idx = np.argpartition(coef, -self.top_k)[-self.top_k:]
                mask = np.zeros_like(coef, dtype=bool)
                mask[idx] = True
                coef[~mask] = 0.0

            W[:, j] = coef
            X[:, j] = xj

        self.W = W

    def predict(self, user_id, user_item_matrix, user2id, id2item, k=10):
        u = user2id[user_id]
        R = _to_csr(user_item_matrix)
        row = R[u]
        scores = np.asarray(row.dot(self.W)).ravel()

        seen = row.nonzero()[1]
        scores[seen] = -1.0

        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [id2item[int(i)] for i in top_k_idx]