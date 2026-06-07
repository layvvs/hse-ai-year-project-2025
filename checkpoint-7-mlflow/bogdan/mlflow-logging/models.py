import os

import numpy as np
from scipy.sparse import csr_matrix

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _to_csr(user_item_matrix):
    if hasattr(user_item_matrix, "tocsr"):
        return user_item_matrix.tocsr()
    return csr_matrix(user_item_matrix)


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
